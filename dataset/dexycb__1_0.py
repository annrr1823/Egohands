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

from dataset.hand_rgb__1_0 import HandRGBDataset
from model.smpl.utils.rotation import unify_transfer_matrix
from model.smpl.utils.rot6d import matrix_to_rotation_6d
from model.smpl.transfer import transfer_mano_params_A_to_B

DEXYCB_OBJECTS_DICT = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

DEXYCB_EXTRINSICS_DICT = {
    "20200709-subject-01": "extrinsics_20200702_151821",
    "20200813-subject-02": "extrinsics_20200813_100608",
    "20200820-subject-03": "extrinsics_20200820_091149",
    "20200903-subject-04": "extrinsics_20200903_072753",
    "20200908-subject-05": "extrinsics_20200907_105926",
    "20200918-subject-06": "extrinsics_20200918_092020",
    "20200928-subject-07": "extrinsics_20200928_082347",
    "20201002-subject-08": "extrinsics_20201001_200551",
    "20201015-subject-09": "extrinsics_20201014_215638",
    "20201022-subject-10": "extrinsics_20201022_091549",
}

SUBJECT_LIST = [
    "20200709-subject-01",
    "20200813-subject-02",
    "20200820-subject-03",
    "20200903-subject-04",
    "20200908-subject-05",
    "20200918-subject-06",
    "20200928-subject-07",
    "20201002-subject-08",
    "20201015-subject-09",
    "20201022-subject-10",
]

VIEW_LIST = [
    "840412060917",     # World
    "836212060125",
    "839512060362",
    "841412060263",
    "932122060857",
    "932122060861",
    "932122061900",
    "932122062010",
]


class DexYCBRGBDataset(HandRGBDataset):
    def __init__(
        self,
        dataset_folder: str,
        split: Literal["train", "val", "test"] = "train",
        setup: Literal["s0", "s1", "s2", "s3"] = "s0",
        **kwargs,
    ):
        self.dataset_folder = dataset_folder
        self.split = split
        self.setup = setup

        self.raw_image_size = (640, 480)

        self.cache_dir = Path(__file__).parent.parent / "_DATA"
        self.sequence_cache_file = self.cache_dir / "sequences_cache" / f"dexycb_{setup}_{split}.pkl"

        print(f"[INFO] Loading DexYCB dataset with setup {setup}-{split}.")
        
        super().__init__(**kwargs)
    
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

        data_root = os.path.join(self.dataset_folder, "bop", self.setup, self.split)

        seq_ids = sorted(os.listdir(data_root))
        for seq_id in seq_ids:
            frame_ids = []
            frame_folder = os.path.join(data_root, seq_id, "rgb")
            frame_names = sorted(os.listdir(frame_folder))

            for frame_name in frame_names:
                frame_path = os.path.join(frame_folder, frame_name)
                image_real_path = os.path.realpath(frame_path)

                # Filter invalid frame
                label_path = image_real_path.replace("color_", "labels_").replace(".jpg", ".npz")
                label = np.load(label_path)
                pose_m = label['pose_m']
                joints_3d = label['joint_3d']
                if np.all(pose_m == 0) or np.all(joints_3d == -1):
                    continue

                frame_ids.append(image_real_path)

            sequences.append({"seq_id": seq_id, "frame_ids": frame_ids})

        return sequences

    def convert_frame_id_to_image_path(self, frame_id: str):
        image_path = frame_id

        return image_path

    def read_annos(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        anno: Dict[str, Any] = {}

        frame_id = frame["frame_id"]
        image_path = self.convert_frame_id_to_image_path(frame_id)

        meta_path = Path(image_path).parent.parent / "meta.yml"
        with open(meta_path, 'r') as f:
            meta_info = yaml.load(f, Loader=yaml.FullLoader)

        # 0: left, 1: right, 2: both
        hand_side_mask = torch.zeros(2, dtype=torch.float32)

        hand_side = meta_info['mano_sides'][0]    
        if hand_side == "left":
            hand_side = 0
            hand_side_mask[0] = 1.0
        elif hand_side == "right":
            hand_side = 1
            hand_side_mask[1] = 1.0
        elif hand_side == "both":
            hand_side = 2
            hand_side_mask[:] = 1.0
        else:
            raise ValueError(f"Unknown hand_side type {hand_side} for DexYCB dataset.")

        anno["hand_side"] = hand_side
        anno["hand_side_mask"] = hand_side_mask

        objects = meta_info['ycb_ids']
        object_id = meta_info['ycb_grasp_ind']
        anno["object"] = ' '.join(DEXYCB_OBJECTS_DICT[objects[object_id]].split('_')[1:])

        # Load camera params.
        view_name = Path(image_path).parent.name
        subject_name = Path(image_path).parent.parent.parent.name

        intrinsics_path = os.path.join(self.dataset_folder, 'calibration', 'intrinsics', f'{view_name}_640x480.yml')
        with open(intrinsics_path, 'r') as f:
            cameras_intrinsics = yaml.load(f, Loader=yaml.FullLoader)
        intrinsics = cameras_intrinsics['color']

        fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["ppx"], intrinsics["ppy"]
        fx_resize = fx * self.resize_scale[0]
        fy_resize = fy * self.resize_scale[1]
        cx_resize = cx * self.resize_scale[0]
        cy_resize = cy * self.resize_scale[1]

        anno["intrinsics"] = torch.asarray([fx, fy, cx, cy]).reshape(1, 4)
        anno["intrinsics_resize"] = torch.asarray([fx_resize, fy_resize, cx_resize, cy_resize]).reshape(1, 4)
        
        extrinsics_path = os.path.join(self.dataset_folder, "calibration", DEXYCB_EXTRINSICS_DICT[subject_name], "extrinsics.yml")
        with open(extrinsics_path, 'r') as f:
            cameras_extrinsics = yaml.load(f, Loader=yaml.FullLoader)["extrinsics"]
        extrinsics = cameras_extrinsics[view_name]

        anno["extrinsics"] = torch.asarray(extrinsics).reshape(1, 3, 4)

        # Load hand pose params.
        mano_calib = meta_info['mano_calib'][0]
        mano_calib_path = os.path.join(self.dataset_folder, 'calibration', f'mano_{mano_calib}', 'mano.yml')
        with open(mano_calib_path, 'r') as f:
            mano_calib_infos = yaml.load(f, Loader=yaml.FullLoader)
        betas = torch.asarray(mano_calib_infos['betas']).reshape((10, ))

        label_path = image_path.replace("color_", "labels_").replace(".jpg", ".npz")
        label = np.load(label_path)

        joints3d = torch.from_numpy(label['joint_3d']).view([21, 3])
        
        pose_m = torch.from_numpy(label['pose_m'])
        transl = pose_m[:, 48:51]
        global_orient = pose_m[:, :3]
        hand_pose_pca45 = pose_m[:, 3:48]

        mano_layer = self.mano_lhand if hand_side == 0 else self.mano_rhand
        hand_pose = mano_layer.decode_pose_pca(hand_pose_pca45)

        global_orient = batch_rodrigues(global_orient.reshape(1, 3)).view(1, 3, 3)
        global_orient = matrix_to_rotation_6d(global_orient).view(1, 1, 6)

        hand_pose = batch_rodrigues(hand_pose.reshape(15, 3)).view(15, 3, 3)
        hand_pose = matrix_to_rotation_6d(hand_pose).view(1, 15, 6)

        # Check the correctness of the MANO params.
        mano_output = mano_layer(
            pose_type="rot6d",
            betas=betas[None, :], 
            global_orient=global_orient,
            hand_pose=hand_pose,
            transl=transl
        )
        joints3d_mano = mano_output["joints"][0]
        mpjpe = torch.norm(joints3d_mano - joints3d, dim=-1).mean()
        assert mpjpe < 1e-3, f"MPJPE {mpjpe.item()} is unacceptable, there might be some bug in the transfer function."

        joints3d_hands = torch.zeros(2, 21, 3)
        transl_hands = torch.zeros(2, 3)
        pose_hands = torch.zeros(2, 16, 6)
        betas_hands = torch.zeros(2, 10)

        if hand_side in (0, 1):
            joints3d_hands[hand_side] = joints3d
            transl_hands[hand_side] = transl
            pose_hands[hand_side] = torch.cat([global_orient, hand_pose], dim=1).view(16, 6)
            betas_hands[hand_side] = betas
        else:
            raise ValueError(f"Unknown hand_side type {hand_side} for DexYCB dataset.")
        
        anno["joints3d"] = joints3d_hands
        anno["transl"] = transl_hands
        anno["pose"] = pose_hands
        anno["betas"] = betas_hands

        joints2d_hands = torch.zeros(2, 21, 2)

        joints2d = torch.from_numpy(label['joint_2d']).view([21, 2])
        if hand_side in (0, 1):
            joints2d_hands[hand_side] = joints2d
        else:
            raise ValueError(f"Unknown hand_side type {hand_side} for DexYCB dataset.")
        anno["joints2d"] = joints2d_hands
            
        bbox_hands = torch.zeros(2, 4)

        frame_t = frame["t"]
        parts = Path(image_path).parts

        bbox_path = self.cache_dir / "hand_detection" / "dexycb" / parts[-4] / parts[-3] / parts[-2] / "bbox.npy"
        bbox_seq = np.load(bbox_path)
        bbox = bbox_seq[hand_side, frame_t]

        if hand_side in (0, 1):
            if not np.all(bbox == -1):
                x1, y1, x2, y2 = bbox[:4]
                center_x = (x1 + x2) * 0.5
                center_y = (y1 + y2) * 0.5
                width = x2 - x1
                height = y2 - y1

                bbox = torch.tensor([center_x, center_y, width, height], dtype=torch.float32)
                bbox_hands[hand_side] = bbox
        else:
            raise ValueError(f"Unknown hand_side type {hand_side} for DexYCB dataset.")

        anno["bbox"] = bbox_hands

        return anno


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

        bbox = bbox_hands[hand_side]
        if hand_side == 0:
            left_hand_image, left_crop_trans = self.crop_image_with_bbox(image, bbox, crop_size)
            right_hand_image = np.zeros((int(crop_h), int(crop_w), 3), dtype=image.dtype)
            right_crop_trans = np.zeros((2, 3), dtype=np.float32)
        elif hand_side == 1:
            right_hand_image, right_crop_trans = self.crop_image_with_bbox(image, bbox, crop_size)
            left_hand_image = np.zeros((int(crop_h), int(crop_w), 3), dtype=image.dtype)
            left_crop_trans = np.zeros((2, 3), dtype=np.float32)
        
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

    dexycb_dataset = DexYCBRGBDataset(
        dataset_folder="/data/public/dataset/hand/dexycb", 
        extra_anno_keys=["extrinsics"],
        image_size=(640, 480),
        split="val"
    )


    idx = dexycb_dataset.random.randint(0, len(dexycb_dataset.frames))
    sample = dexycb_dataset.__getitem__(idx)


    frame_id = sample["meta"]["frame_id"]
    print(f"[INFO] Visualizing sample with frame_id {frame_id}.")

    subject_id = Path(frame_id).parent.parent.parent.name

    image_path = dexycb_dataset.convert_frame_id_to_image_path(frame_id)
    image = cv2.imread(image_path)


    hand_side = sample["hand_side"]

    betas = sample["betas"]
    pose = sample["pose"]
    transl = sample["transl"]

    if hand_side == 0:
        mano_layer = dexycb_dataset.mano_lhand
    elif hand_side == 1:
        mano_layer = dexycb_dataset.mano_rhand

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

    intrinsics = sample["intrinsics_resize"][0]

    render_bgr_with_image, render_rgba, render_depth = renderer.render_with_image(
        image=image,
        vertices=mano_output["vertices"].numpy(),
        is_rhand=[hand_side],
        focal_length=intrinsics[:2],
        camera_center=intrinsics[2:]
    )

    cv2.imwrite("check_visualization/render.png", render_bgr_with_image)
