from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Literal

import os

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md"],
    pythonpath=True,
    dotenv=True,
)

import numpy as np
import pickle
import torch

import smplx
from smplx.lbs import lbs, batch_rodrigues
from smplx.vertex_ids import vertex_ids as VERTEX_IDS
from smplx.utils import Struct, to_tensor, Tensor, MANOOutput

from model.smpl.utils.rot6d import rotation_6d_to_matrix

MANO_PATH = "/data/sunzhihao/mano/mano/model"


class MANOLayer(smplx.SMPL):
    NUM_BODY_JOINTS = 1
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS

    def __init__(
        self,
        model_path: str,
        is_rhand: bool = True,
        data_struct: Optional[Struct] = None,
        flat_hand_mean: bool = False,
        dtype=torch.float32,
        vertex_ids=None,
        use_compressed: bool = True,
        ext: str = 'pkl',
        **kwargs
    ) -> None:
        ''' 
        Extend the official MANO implementation (smplx.MANO) with additional features:
        1)  Removes internal pose/shape/transl parameters.
        2)  Accepts flexible pose input types with unified rotation-matrix representation.
            forward() accepts pose representations specified by pose_type: "pca", "axisangle", "rot6d", "rotmatrix".
            The intent is to internally convert all supported types to rotation matrices before calling LBS (pose2rot=False).
        3)  Include extra joint indices (extra_joints_idxs) to include fingertip joints, totaling 21 joints.
        4)  Reorder joints using openpose_joint_map for consistency.
        '''

        self.is_rhand = is_rhand

        # If no data structure is passed, load the data from the given model folder.
        if data_struct is None:
            # Load the model
            if os.path.isdir(model_path):
                model_fn = 'MANO_{}.{ext}'.format(
                    'RIGHT' if is_rhand else 'LEFT', ext=ext)
                mano_path = os.path.join(model_path, model_fn)
            else:
                mano_path = model_path
                self.is_rhand = True if 'RIGHT' in os.path.basename(
                    model_path) else False
            assert os.path.exists(mano_path), 'Path {} does not exist!'.format(
                mano_path)

            if ext == 'pkl':
                with open(mano_path, 'rb') as mano_file:
                    model_data = pickle.load(mano_file, encoding='latin1')
            elif ext == 'npz':
                model_data = np.load(mano_path, allow_pickle=True)
            else:
                raise ValueError('Unknown extension: {}'.format(ext))
            data_struct = Struct(**model_data)

        if vertex_ids is None:
            vertex_ids = VERTEX_IDS['smplh']

        super(MANOLayer, self).__init__(
            model_path=model_path, data_struct=data_struct,
            vertex_ids=vertex_ids,
            use_compressed=use_compressed, dtype=dtype, ext=ext, 
            create_global_orient=False,
            create_hand_pose=False,
            create_betas=False,
            create_transl=False,
            **kwargs)

        # Add only MANO tips to the extra joints.
        self.vertex_joint_selector.extra_joints_idxs = to_tensor(
            list(VERTEX_IDS['mano'].values()), dtype=torch.long)

        hand_components = data_struct.hands_components
        self.register_buffer('hand_components', torch.tensor(hand_components, dtype=dtype))

        self.flat_hand_mean = flat_hand_mean
        if self.flat_hand_mean:
            hand_mean = np.zeros_like(data_struct.hands_mean)
        else:
            hand_mean = data_struct.hands_mean
        self.register_buffer('hand_mean', to_tensor(hand_mean, dtype=self.dtype))

        # Create the buffer for the mean pose.
        pose_mean = self.create_mean_pose()
        pose_mean_tensor = pose_mean.clone().to(dtype)
        self.register_buffer('pose_mean', pose_mean_tensor)

        openpose_joint_map = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        self.register_buffer('joint_map', torch.tensor(openpose_joint_map, dtype=torch.long))

    def forward(
        self,
        pose_type: Literal["pca", "axisangle", "rot6d", "rotmatrix"] = "rotmatrix",
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
    ) -> MANOOutput:
        device, dtype = self.shapedirs.device, self.shapedirs.dtype

        if global_orient is None:
            batch_size = 1
            global_orient = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        else:
            batch_size = global_orient.shape[0]

            if pose_type == "pca" or pose_type == "axisangle":
                global_orient = batch_rodrigues(global_orient.reshape(-1, 3)).reshape([batch_size, 1, 3, 3])
            elif pose_type == "rot6d":
                global_orient = rotation_6d_to_matrix(global_orient.reshape(-1, 6)).reshape([batch_size, 1, 3, 3])

        if hand_pose is None:
            hand_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        else:
            if pose_type == "pca":
                hand_pose = self.decode_pose_pca(hand_pose)
                hand_pose = batch_rodrigues(hand_pose.reshape(-1, 3)).reshape([batch_size, -1, 3, 3])
            elif pose_type == "axisangle":
                hand_pose = self.decode_pose_axisangle(hand_pose)
                hand_pose = batch_rodrigues(hand_pose.reshape(-1, 3)).reshape([batch_size, -1, 3, 3])
            elif pose_type == "rot6d":
                hand_pose = rotation_6d_to_matrix(hand_pose.reshape(-1, 6)).reshape([batch_size, -1, 3, 3])
            
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas], dtype=dtype, device=device)
            
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        full_pose = torch.cat([global_orient, hand_pose], dim=1)

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=False)
        
        # Add pre-selected extra joints that might be needed.
        joints = self.vertex_joint_selector(vertices, joints)

        if self.joint_map is not None:
            joints = joints[:, self.joint_map, :]
        
        joints = joints + transl.unsqueeze(dim=1)
        vertices = vertices + transl.unsqueeze(dim=1)

        output = MANOOutput(
            vertices=vertices,
            joints=joints,
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose,
            full_pose=full_pose,
        )

        return output

    def name(self) -> str:
        return 'MANOLayer'

    def create_mean_pose(self):
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        pose_mean = torch.cat([global_orient_mean, self.hand_mean], dim=0)

        return pose_mean

    def decode_pose_pca(self, pca_coeffs: torch.Tensor):
        num_pca_comps = pca_coeffs.shape[-1]
        hand_components = self.hand_components[:num_pca_comps]
        
        pose = torch.einsum('bi,ij->bj', [pca_coeffs, hand_components])
        pose += self.hand_mean

        return pose
    
    def decode_pose_axisangle(self, pose: torch.Tensor):
        pose += self.hand_mean

        return pose


if __name__ == "__main__":
    mano = MANOLayer(model_path=MANO_PATH, is_rhand=True)
