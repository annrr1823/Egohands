from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Literal

import os
import json

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from smplx.lbs import batch_rodrigues
import torch
import yaml

from dataset.hand_rgb__2_0 import HandRGBDataset
from model.smpl.utils.rotation import unify_transfer_matrix
from model.smpl.utils.rot6d import matrix_to_rotation_6d
from model.smpl.transfer import transfer_mano_params_A_to_B


class ARCTICEgoRGBDataset(HandRGBDataset):
    def __init__(
        self,
        dataset_folder: str,
        split: Literal["train", "val", "test"] = "train",
        setup: Literal["p1", "p2"] = "p1",
        sequence_length: int = 8,
        camera_cache_file=None,
        camera_source: Literal["auto", "cache_only", "gt_only"] = "auto",
        **kwargs,
    ):
        self.dataset_folder = dataset_folder
        self.split = split
        self.setup = setup
        self.sequence_length = sequence_length
        self.camera_cache_file = camera_cache_file
        self.camera_source = camera_source

        self.raw_image_size = (2800, 2000)

        self.cache_dir = Path(__file__).parent.parent / "_DATA"
        if camera_source == "auto":
            self.sequence_cache_file = self.cache_dir / "sequences_cache" / f"arctic_{setup}_{split}.pkl"
        else:
            self.sequence_cache_file = self.cache_dir / "sequences_cache" / f"arctic_{setup}_{split}_{camera_source}.pkl"
        self.camera_cache = self._load_camera_cache(self.camera_cache_file)

        print(f"[INFO] Loading ARCTIC dataset with setup {setup}-{split}.")

        # data_root = os.path.join(self.dataset_folder, "arctic/unpack/arctic_data/data")
        data_root = self.dataset_folder


        split_meta_path = os.path.join(data_root, "splits", f"{self.setup}_{self.split}.npy")
        split_meta = np.load(split_meta_path, allow_pickle=True).item()
        self.split_meta = split_meta["data_dict"]

        misc_path = os.path.join(data_root, "meta", "misc.json")
        with open(misc_path, "r") as f:
            self.misc_meta = json.load(f)
        
        super().__init__(sequence_length=sequence_length, camera_source=camera_source, **kwargs)
    
    def collect_sequences(self) -> List[dict]:
        """
        Return a list of sequences, each like:
            {
                "seq_id": "S0001",
                "frame_ids": [
                    "/path/to/840412060917/image_0001.jpg",
                    "/path/to/840412060917/image_0002.jpg",
                    ...
                ]
            }
        """
        sequences: List[dict] = []

        if self.setup == "p1":
            if self.split == "train":
                subjects = ["s01", "s02", "s04", "s06", "s07", "s08", "s09", "s10"]
            elif self.split == "val":
                subjects = ["s05"]
            elif self.split == "test":
                subjects = ["s03"]
            elif self.split == "all":
                subjects = ["s01", "s02", "s03", "s04", "s05", "s06", "s07", "s08", "s09", "s10"]
            else:
                raise ValueError(f"The split {self.split} is not supported.")
        elif self.setup == "p2":
            raise NotImplementedError(f"The setup {self.setup} is not implemented yet.")
        else:
            raise ValueError(f"The setup {self.setup} is not supported.")

        # data_root = os.path.join(self.dataset_folder, "arctic/unpack/arctic_data/data")
        data_root = self.dataset_folder

        for subject in subjects:
            seq_ids = os.listdir(os.path.join(data_root, "images", subject))
            index_offset = self.misc_meta[subject]['ioi_offset']

            for seq_id in seq_ids:
                frame_ids = []
                frame_folder = os.path.join(data_root, "images", subject, seq_id, "0")

                all_frame_ids = [frame_name[:-4] for frame_name in os.listdir(frame_folder) if frame_name[-4:] == ".jpg"]
                all_frame_ids = sorted(all_frame_ids)

                seq_meta = self.split_meta[f"{subject}/{seq_id}"]
                cam_coord_meta = seq_meta["cam_coord"]
                left_valid = cam_coord_meta["left_valid"]
                right_valid = cam_coord_meta["right_valid"]

                for frame_id in all_frame_ids:
                    frame_index = int(frame_id) - 1 - index_offset

                    left_valid_frame = left_valid[frame_index][0]
                    right_valid_frame = right_valid[frame_index][0]

                    if left_valid_frame or right_valid_frame:
                        frame_ids.append(os.path.join(frame_folder, f"{frame_id}.jpg"))

                if self.camera_source == "cache_only":
                    if len(frame_ids) == 0:
                        continue

                    if not all(self.has_cached_camera_record({"frame_id": frame_id}) for frame_id in frame_ids):
                        continue

                sequences.append({"seq_id": f"{subject}/{seq_id}", "frame_ids": frame_ids})

        return sequences

    def convert_frame_id_to_image_path(self, frame_id: str):
        image_path = frame_id
        return image_path

    def read_annos(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        anno: Dict[str, Any] = {}

        frame_id = frame["frame_id"]
        image_path = self.convert_frame_id_to_image_path(frame_id)

        subject = Path(image_path).parent.parent.parent.name
        seq_id = Path(image_path).parent.parent.name

        label_path = Path(image_path).parent.parent.parent.parent.parent / "raw_seqs" / subject / f"{seq_id}.mano.npy"
        egocam_path = Path(image_path).parent.parent.parent.parent.parent / "raw_seqs" / subject / f"{seq_id}.egocam.dist.npy"

        index_offset = self.misc_meta[subject]['ioi_offset']
        frame_index = int(Path(image_path).stem) - 1 - index_offset

        seq_meta = self.split_meta[f"{subject}/{seq_id}"]
        cam_coord_meta = seq_meta["cam_coord"]
        left_valid = cam_coord_meta["left_valid"][frame_index][0]
        right_valid = cam_coord_meta["right_valid"][frame_index][0]

        # 0: left, 1: right, 2: both
        hand_side_mask = torch.zeros(2, dtype=torch.float32)

        if left_valid and right_valid:
            hand_side = 2
            hand_side_mask[:] = 1.0
        elif left_valid:
            hand_side = 0
            hand_side_mask[0] = 1.0
        elif right_valid:
            hand_side = 1
            hand_side_mask[1] = 1.0

        anno["hand_side"] = hand_side
        anno["hand_side_mask"] = hand_side_mask

        anno["object"] = seq_id

        cached_camera = self.get_cached_camera_record(frame)
        use_cached_camera = cached_camera is not None and self.camera_source in {"auto", "cache_only"}

        if use_cached_camera:
            intrinsics = cached_camera["intrinsics"]
            anno["intrinsics"] = intrinsics
            anno["intrinsics_resize"] = self._compute_intrinsics_resize(intrinsics)
            extrinsics = cached_camera["extrinsics"]
            anno["extrinsics"] = extrinsics
            T_camera_to_world = np.eye(4, dtype=np.float32)
            T_camera_to_world[:3, :] = extrinsics.squeeze(0).numpy()
            T_world_to_camera = np.linalg.inv(T_camera_to_world)
        else:
            if self.camera_source == "cache_only":
                raise KeyError(f"Camera cache not found for frame {frame_id} while camera_source='cache_only'.")

            # Load camera params.
            egocam_meta = np.load(egocam_path, allow_pickle=True).item()

            intrinsics = egocam_meta['intrinsics']
            fx = intrinsics[0][0]
            fy = intrinsics[1][1]
            cx = intrinsics[0][2]
            cy = intrinsics[1][2]

            fx_resize = fx * self.resize_scale[0]
            fy_resize = fy * self.resize_scale[1]
            cx_resize = cx * self.resize_scale[0]
            cy_resize = cy * self.resize_scale[1]

            anno["intrinsics"] = torch.asarray([fx, fy, cx, cy]).reshape(1, 4)
            anno["intrinsics_resize"] = torch.asarray([fx_resize, fy_resize, cx_resize, cy_resize]).reshape(1, 4)

            R = egocam_meta["R_k_cam_np"][frame_index]
            t = egocam_meta["T_k_cam_np"][frame_index]

            T_world_to_camera = unify_transfer_matrix(rotation_matrix=R, translation_xyz=t[:, 0])
            T_camera_to_world = np.linalg.inv(T_world_to_camera)

            extrinsics = torch.asarray(T_camera_to_world[:3, :]).reshape(1, 3, 4)
            anno["extrinsics"] = extrinsics

        # Load hand pose params.
        label = np.load(label_path, allow_pickle=True).item()
        joints3d_hands = torch.zeros(2, 21, 3)
        
        betas_hands = torch.zeros(2, 10)
        pose_hands = torch.zeros(2, 16, 6)
        transl_hands = torch.zeros(2, 3)
        global_orient_world_hands = torch.zeros(2, 1, 6)
        transl_world_hands = torch.zeros(2, 3)

        if left_valid:
            left_betas = torch.from_numpy(label['left']['shape'])
            betas_hands[0] = left_betas

            left_global_orient = torch.from_numpy(label['left']['rot'][frame_index])
            left_global_orient = batch_rodrigues(left_global_orient.reshape(1, 3)).view(1, 3, 3)
            left_global_orient = matrix_to_rotation_6d(left_global_orient).view(1, 1, 6)
            global_orient_world_hands[0] = left_global_orient

            left_hand_pose_aa45 = torch.from_numpy(label['left']['pose'][frame_index])
            left_hand_pose = self.mano_lhand.decode_pose_axisangle(left_hand_pose_aa45)
            left_hand_pose = batch_rodrigues(left_hand_pose.reshape(15, 3)).view(15, 3, 3)
            left_hand_pose = matrix_to_rotation_6d(left_hand_pose).view(1, 15, 6)

            left_transl = torch.from_numpy(label['left']['trans'][frame_index]).view(1, 3)
            transl_world_hands[0] = left_transl

            left_wrist_offset = self.mano_lhand(
                pose_type="rot6d",
                betas=left_betas[None, :], 
                global_orient=left_global_orient,
                hand_pose=left_hand_pose,
            )["joints"][..., 0, :]

            left_global_orient_camera, left_transl_camera = transfer_mano_params_A_to_B(
                T_world_to_camera, left_global_orient.numpy(), left_transl.numpy(), left_wrist_offset.numpy()
            )
            left_global_orient_camera = torch.from_numpy(left_global_orient_camera).reshape(1, 1, 6)
            left_transl_camera = torch.from_numpy(left_transl_camera).reshape(1, 3)

            pose_hands[0, :1] = left_global_orient_camera
            pose_hands[0, 1:] = left_hand_pose
            transl_hands[0] = left_transl_camera

            # # Check the correctness of the MANO params.
            # left_joints3d = torch.from_numpy(cam_coord_meta["joints.left"][frame_index][0]).view(21, 3)

            left_mano_output = self.mano_lhand(
                pose_type="rot6d",
                betas=left_betas[None, :], 
                global_orient=left_global_orient_camera,
                hand_pose=left_hand_pose,
                transl=left_transl_camera
            )
            left_joints3d_mano = left_mano_output["joints"][0]
            # mpjpe = torch.norm(left_joints3d_mano - left_joints3d, dim=-1).mean()
            # assert mpjpe < 1e-3, f"MPJPE {mpjpe.item()} is unacceptable, there might be some bug in the transfer function."

            joints3d_hands[0] = left_joints3d_mano


        if right_valid:
            right_betas = torch.from_numpy(label['right']['shape'])
            betas_hands[1] = right_betas

            right_global_orient = torch.from_numpy(label['right']['rot'][frame_index])
            right_global_orient = batch_rodrigues(right_global_orient.reshape(1, 3)).view(1, 3, 3)
            right_global_orient = matrix_to_rotation_6d(right_global_orient).view(1, 1, 6)
            global_orient_world_hands[1] = right_global_orient

            right_hand_pose_aa45 = torch.from_numpy(label['right']['pose'][frame_index])
            right_hand_pose = self.mano_rhand.decode_pose_axisangle(right_hand_pose_aa45)
            right_hand_pose = batch_rodrigues(right_hand_pose.reshape(15, 3)).view(15, 3, 3)
            right_hand_pose = matrix_to_rotation_6d(right_hand_pose).view(1, 15, 6)

            right_transl = torch.from_numpy(label['right']['trans'][frame_index]).view(1, 3)
            transl_world_hands[1] = right_transl

            right_wrist_offset = self.mano_rhand(
                pose_type="rot6d",
                betas=right_betas[None, :], 
                global_orient=right_global_orient,
                hand_pose=right_hand_pose,
            )["joints"][..., 0, :]

            right_global_orient_camera, right_transl_camera = transfer_mano_params_A_to_B(
                T_world_to_camera, right_global_orient.numpy(), right_transl.numpy(), right_wrist_offset.numpy()
            )
            right_global_orient_camera = torch.from_numpy(right_global_orient_camera).reshape(1, 1, 6)
            right_transl_camera = torch.from_numpy(right_transl_camera).reshape(1, 3)

            pose_hands[1, :1] = right_global_orient_camera
            pose_hands[1, 1:] = right_hand_pose
            transl_hands[1] = right_transl_camera

            # # Check the correctness of the MANO params.
            # right_joints3d = torch.from_numpy(cam_coord_meta["joints.right"][frame_index][0]).view(21, 3)

            right_mano_output = self.mano_rhand(
                pose_type="rot6d",
                betas=right_betas[None, :], 
                global_orient=right_global_orient_camera,
                hand_pose=right_hand_pose,
                transl=right_transl_camera
            )
            right_joints3d_mano = right_mano_output["joints"][0]
            # mpjpe = torch.norm(right_joints3d_mano - right_joints3d, dim=-1).mean()
            # assert mpjpe < 1e-3, f"MPJPE {mpjpe.item()} is unacceptable, there might be some bug in the transfer function."
            joints3d_hands[1] = right_joints3d_mano

        anno["betas"] = betas_hands
        anno["pose"] = pose_hands
        anno["transl"] = transl_hands
        anno["joints3d"] = joints3d_hands
        anno["global_orient_world"] = global_orient_world_hands
        anno["transl_world"] = transl_world_hands
    
        bbox_hands = torch.zeros(2, 4)

        frame_t = frame["t"]
        parts = Path(image_path).parts

        bbox_path = self.cache_dir / "hand_detection" / "arctic" / parts[-4] / parts[-3] / "bbox.npy"
        bbox_seq = np.load(bbox_path)
        bbox = bbox_seq[:, frame_t]

        if left_valid:
            left_bbox = bbox[0]
            if not np.all(left_bbox == -1):
                x1, y1, x2, y2 = left_bbox[:4]
                center_x = (x1 + x2) * 0.5
                center_y = (y1 + y2) * 0.5
                width = x2 - x1
                height = y2 - y1

                left_bbox = torch.tensor([center_x, center_y, width, height], dtype=torch.float32)
                bbox_hands[0] = left_bbox
        if right_valid:
            right_bbox = bbox[1]
            if not np.all(right_bbox == -1):
                x1, y1, x2, y2 = right_bbox[:4]
                center_x = (x1 + x2) * 0.5
                center_y = (y1 + y2) * 0.5
                width = x2 - x1
                height = y2 - y1

                right_bbox = torch.tensor([center_x, center_y, width, height], dtype=torch.float32)
                bbox_hands[1] = right_bbox

        anno["bbox"] = bbox_hands

        return anno

    def _build_extrinsics_4x4(self, extrinsics: torch.Tensor) -> np.ndarray:
        extrinsics_4x4 = np.eye(4, dtype=np.float32)
        extrinsics_4x4[:3, :] = extrinsics.squeeze(0).cpu().numpy()
        return extrinsics_4x4

    def _transform_hand_world_to_canonical(
        self,
        T_world_to_canonical: np.ndarray,
        anno: Dict[str, Any],
        hand_index: int,
    ) -> None:
        if anno["hand_side_mask"][hand_index] <= 0:
            return

        mano_layer = self.mano_lhand if hand_index == 0 else self.mano_rhand

        betas = anno["betas"][hand_index][None, :]
        global_orient_world = anno["global_orient_world"][hand_index][None]
        hand_pose = anno["pose"][hand_index, 1:][None]
        transl_world = anno["transl_world"][hand_index][None]

        wrist_offset = mano_layer(
            pose_type="rot6d",
            betas=betas,
            global_orient=global_orient_world,
            hand_pose=hand_pose,
        )["joints"][..., 0, :]

        global_orient_canonical, transl_canonical = transfer_mano_params_A_to_B(
            T_world_to_canonical,
            global_orient_world.numpy(),
            transl_world.numpy(),
            wrist_offset.numpy(),
        )
        global_orient_canonical = torch.from_numpy(global_orient_canonical).reshape(1, 1, 6)
        transl_canonical = torch.from_numpy(transl_canonical).reshape(1, 3)

        anno["pose"][hand_index, :1] = global_orient_canonical
        anno["transl"][hand_index] = transl_canonical

        mano_output = mano_layer(
            pose_type="rot6d",
            betas=betas,
            global_orient=global_orient_canonical,
            hand_pose=hand_pose,
            transl=transl_canonical,
        )
        anno["joints3d"][hand_index] = mano_output["joints"][0]

    def convert_sequence_annos_to_canonical(
        self,
        frames: Sequence[Dict[str, Any]],
        annos: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        annos = [dict(anno) for anno in annos]

        if len(annos) == 0:
            return annos

        T_canonical_to_world = self._build_extrinsics_4x4(annos[0]["extrinsics"])
        T_world_to_canonical = np.linalg.inv(T_canonical_to_world)

        for anno in annos:
            curr_extrinsics = self._build_extrinsics_4x4(anno["extrinsics"])
            T_curr_to_canonical = T_world_to_canonical @ curr_extrinsics

            self._transform_hand_world_to_canonical(T_world_to_canonical, anno, hand_index=0)
            self._transform_hand_world_to_canonical(T_world_to_canonical, anno, hand_index=1)

            anno["extrinsics"] = torch.from_numpy(T_curr_to_canonical[:3, :]).reshape(1, 3, 4)
            anno.pop("global_orient_world", None)
            anno.pop("transl_world", None)

        return annos


    def scale_bbox(self, bbox, aspect_ratio=(192.0, 256.0), scale_factor=1.5):
        center_x, center_y, width, height = bbox

        target_ratio = float(aspect_ratio[0]) / float(aspect_ratio[1])
        if width / height > target_ratio:
            height = width / target_ratio
        else:
            width = height * target_ratio

        width = width * scale_factor
        height = height * scale_factor

        return torch.stack([center_x, center_y, width, height], dim=0)
    
    def crop_image_with_bbox(self, image, bbox, crop_size=(192, 256)):
        if torch.all(bbox == 0):
            crop_w, crop_h = crop_size
            hand_image = np.zeros((int(crop_h), int(crop_w), 3), dtype=image.dtype)
            trans = np.zeros((2, 3), dtype=np.float32)
            return hand_image, trans
        bbox = self.scale_bbox(bbox)

        center_x, center_y, width, height = bbox.tolist()
        crop_w, crop_h = crop_size
        src_w = width
        src_h = height

        src = np.array(
            [
                [center_x - src_w * 0.5, center_y - src_h * 0.5],
                [center_x - src_w * 0.5, center_y + src_h * 0.5],
                [center_x + src_w * 0.5, center_y - src_h * 0.5],
            ],
            dtype=np.float32,
        )
        dst = np.array(
            [
                [0.0, 0.0],
                [0.0, crop_h - 1.0],
                [crop_w - 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        trans = cv2.getAffineTransform(src, dst)
        hand_image = cv2.warpAffine(
            image,
            trans,
            (int(crop_w), int(crop_h)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        return hand_image, trans

    def read_image(self, frame: Dict[str, Any], anno: Dict[str, Any]):
        frame_id = frame["frame_id"]
        image_path = self.convert_frame_id_to_image_path(frame_id)

        hand_side = anno["hand_side"]
        bbox_hands = anno["bbox"]

        assert os.path.exists(image_path), \
            f"The image_path must be existed, got {image_path}."
                    
        image = cv2.imread(image_path)
        assert image is not None, \
            f"Failed to read {image_path}."

        crop_size = (192, 256)
        crop_w, crop_h = crop_size

        if hand_side == 0:
            left_bbox = bbox_hands[0]
            left_hand_image, left_crop_trans = self.crop_image_with_bbox(image, left_bbox, crop_size)
            right_hand_image = np.zeros((int(crop_h), int(crop_w), 3), dtype=image.dtype)
            right_crop_trans = np.zeros((2, 3), dtype=np.float32)
        elif hand_side == 1:
            right_bbox = bbox_hands[1]
            right_hand_image, right_crop_trans = self.crop_image_with_bbox(image, right_bbox, crop_size)
            left_hand_image = np.zeros((int(crop_h), int(crop_w), 3), dtype=image.dtype)
            left_crop_trans = np.zeros((2, 3), dtype=np.float32)
        else:
            left_bbox = bbox_hands[0]
            right_bbox = bbox_hands[1]
            left_hand_image, left_crop_trans = self.crop_image_with_bbox(image, left_bbox, crop_size)
            right_hand_image, right_crop_trans = self.crop_image_with_bbox(image, right_bbox, crop_size)
        
        image = cv2.resize(image, self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        left_hand_image = cv2.cvtColor(left_hand_image, cv2.COLOR_BGR2RGB)
        right_hand_image = cv2.cvtColor(right_hand_image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0
        image = (image - self.image_mean) / self.image_std
        image = torch.from_numpy(image.transpose(2, 0, 1))

        left_hand_image = left_hand_image.astype(np.float32) / 255.0
        left_hand_image = (left_hand_image - self.image_mean) / self.image_std
        left_hand_image = torch.from_numpy(left_hand_image.transpose(2, 0, 1))

        right_hand_image = right_hand_image.astype(np.float32) / 255.0
        right_hand_image = (right_hand_image - self.image_mean) / self.image_std
        right_hand_image = torch.from_numpy(right_hand_image.transpose(2, 0, 1))

        left_crop_trans = torch.from_numpy(left_crop_trans).float()
        right_crop_trans = torch.from_numpy(right_crop_trans).float()
        
        image_dict = {
            "image": image,
            "left_hand_image": left_hand_image,
            "right_hand_image": right_hand_image,
            "left_crop_trans": left_crop_trans,
            "right_crop_trans": right_crop_trans,
        }

        return image_dict


if __name__ == "__main__":
    os.makedirs("check_visualization", exist_ok=True)

    arctic_dataset = ARCTICEgoRGBDataset(
        dataset_folder="/data/public/dataset/hand/arctic", 
        extra_anno_keys=["extrinsics"],
        image_size=(512, 512),
        split="train"
    )


    # idx = arctic_dataset.random.randint(0, len(arctic_dataset.frames))
    # sample = arctic_dataset.__getitem__(idx)
    sample = arctic_dataset.__getitem__(200)

    frame_id = sample["meta"]["frame_id"]
    print(f"[INFO] Visualizing sample with frame_id {frame_id}.")

    subject_id = Path(frame_id).parent.parent.parent.name

    image_path = arctic_dataset.convert_frame_id_to_image_path(frame_id)
    image = cv2.imread(image_path)

    # hand_side = sample["hand_side"]
    hand_side = 0

    betas = sample["betas"]
    pose = sample["pose"]
    transl = sample["transl"]

    if hand_side == 0:
        mano_layer = arctic_dataset.mano_lhand
    elif hand_side == 1:
        mano_layer = arctic_dataset.mano_rhand

    betas = betas[hand_side][None]
    global_orient = pose[hand_side, :1][None]
    hand_pose = pose[hand_side, 1:][None]
    transl = transl[hand_side][None]

    print(betas.shape)
    print(global_orient.shape)
    print(hand_pose.shape)
    print(transl.shape)
    
    mano_output = mano_layer(
        pose_type="rot6d",
        betas=betas, 
        global_orient=global_orient,
        hand_pose=hand_pose,
        transl=transl
    )

    joints3d = sample["joints3d"][hand_side]
    print((joints3d - mano_output["joints"]).abs().mean())

    from model.smpl.utils.renderer import MANORenderer
    renderer = MANORenderer()

    intrinsics = sample["intrinsics"][0]

    render_bgr_with_image, render_rgba, render_depth = renderer.render_with_image(
        image=image,
        vertices=mano_output["vertices"].numpy(),
        is_rhand=[hand_side],
        focal_length=intrinsics[:2],
        camera_center=intrinsics[2:]
    )

    cv2.imwrite("check_visualization/render.png", render_bgr_with_image)
