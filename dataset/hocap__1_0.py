from typing import Any, Dict, List, Literal, Optional, Tuple

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pyrootutils
import torch
import yaml
from scipy.spatial.transform import Rotation as SciRotation
from smplx.lbs import batch_rodrigues

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md"],
    pythonpath=True,
    dotenv=True,
)

from dataset.hand_rgb__1_0 import HandRGBDataset
from model.smpl.transfer import transfer_mano_params_A_to_B
from model.smpl.utils.rot6d import matrix_to_rotation_6d

# HO-Cap SequenceRenderer: OpenCV camera in OpenGL frame for HoloLens PV pose.
_HL_CVCAM_IN_GLCAM = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32
)
# frame_id[2] == -1 表示使用 HoloLens 头戴 RGB + poses_pv 外参（第一人称）。
_HOLOLENS_FRAME_CAM_IDX = -1


class HOCapRGBDataset(HandRGBDataset):
    def __init__(
        self,
        dataset_folder: str,
        split: Literal["train", "val", "test"] = "train",
        setup: Literal["default"] = "default",
        task: Literal["hpe", "odet", "ope"] = "hpe",
        view: Literal["hololens_pv", "realsense"] = "hololens_pv",
        realsense_cam_idx: int = 0,
        **kwargs,
    ):
        self.dataset_folder = dataset_folder
        self.split = "valid" if split == "val" else split
        self.setup = setup
        self.task = task
        self.view = view
        self.realsense_cam_idx = int(realsense_cam_idx)

        self.cache_dir = Path(__file__).parent.parent / "_DATA"

        split_config_path = os.path.join(dataset_folder, "config", f"hocap_{task}.json")
        with open(split_config_path, "r") as file:
            self.split_config = json.load(file)

        hocap_info_path = os.path.join(dataset_folder, "config", "hocap_info.yaml")
        with open(hocap_info_path, "r") as file:
            hocap_info = yaml.safe_load(file)
        self.hocap_subject_ids: List[str] = hocap_info["subject_ids"]
        self.hocap_sequence_ids: List[str] = hocap_info["sequence_ids"]
        self.hocap_device_serials: List[str] = hocap_info["device_serials"][:-1]

        if self.view == "hololens_pv":
            self.raw_image_size = (1280, 720)
        else:
            if not (0 <= self.realsense_cam_idx < len(self.hocap_device_serials)):
                raise ValueError(
                    f"realsense_cam_idx must be in [0, {len(self.hocap_device_serials) - 1}], "
                    f"got {self.realsense_cam_idx}"
                )
            self.raw_image_size = (640, 480)

        view_tag = "hololens" if self.view == "hololens_pv" else f"rs{self.realsense_cam_idx}"
        self.sequence_cache_file = (
            self.cache_dir / "sequences_cache" / f"hocap_{setup}_{task}_{split}_{view_tag}.pkl"
        )

        self._poses_pv_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._hl_cam_K_np: Optional[np.ndarray] = None
        if self.view == "hololens_pv":
            self._hl_cam_K_np = self._load_hololens_color_intrinsics_matrix()

        self.subject_folders = self._list_subject_folders()
        self.sequence_folders_by_subject = {
            subject_folder: self._list_sequence_folders(subject_folder)
            for subject_folder in self.subject_folders
        }
        self.seq_idx_to_path = self._build_seq_idx_mapping()

        self._sequence_meta_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._camera_w2c_cache: Dict[str, np.ndarray] = {}

        self._mano_betas_by_subject = self._load_all_subject_betas()
        self._rs_world_to_camera = self._load_rs_cam_RTs()

        print(
            f"[INFO] Loading HOCap view={self.view}"
            + (
                " (HoloLens PV, 第一人称)"
                if self.view == "hololens_pv"
                else f" (RealSense index={self.realsense_cam_idx})"
            )
            + f", setup={setup}-{task}-{split}."
        )

        super().__init__(**kwargs)

    def _load_hololens_color_intrinsics_matrix(self) -> np.ndarray:
        intrin_path = os.path.join(
            self.dataset_folder, "calibration", "intrinsics", "hololens_kv5h72.yaml"
        )
        with open(intrin_path, "r") as file:
            color_meta = yaml.safe_load(file)["color"]
        fx = float(color_meta["fx"])
        fy = float(color_meta["fy"])
        cx = float(color_meta["ppx"])
        cy = float(color_meta["ppy"])
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    def _get_poses_pv(self, subject_folder: str, seq_folder: str) -> np.ndarray:
        key = (subject_folder, seq_folder)
        if key not in self._poses_pv_cache:
            poses_candidates = [
                os.path.join(self.dataset_folder, "poses", subject_folder, seq_folder, "poses_pv.npy"),
                os.path.join(self.dataset_folder, subject_folder, seq_folder, "poses_pv.npy"),
            ]
            poses_path = self._resolve_existing_path(poses_candidates, "poses_pv file")
            self._poses_pv_cache[key] = np.load(poses_path).astype(np.float32)
        return self._poses_pv_cache[key]

    def _hololens_c2w_from_pose7(self, pose7: np.ndarray) -> np.ndarray:
        """poses_pv 单行 (qx,qy,qz,qw,tx,ty,tz) -> 与 HO-Cap SequenceRenderer 一致的 HoloLens PV camera-to-world。"""
        q = pose7[:4].astype(np.float64)
        t = pose7[4:7].astype(np.float64)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = SciRotation.from_quat(q).as_matrix().astype(np.float32)
        mat[:3, 3] = t.astype(np.float32)
        return mat @ _HL_CVCAM_IN_GLCAM

    def _hololens_world_to_camera(self, subject_folder: str, seq_folder: str, frame_idx: int) -> np.ndarray:
        poses_pv = self._get_poses_pv(subject_folder, seq_folder)
        if frame_idx < 0 or frame_idx >= poses_pv.shape[0]:
            raise IndexError(
                f"frame_idx {frame_idx} out of range for poses_pv with shape {poses_pv.shape}."
            )
        c2w = self._hololens_c2w_from_pose7(poses_pv[frame_idx])
        return np.linalg.inv(c2w).astype(np.float32)

    def _list_subject_folders(self) -> List[str]:
        return sorted(
            [
                item
                for item in os.listdir(self.dataset_folder)
                if item.startswith("subject_")
                and os.path.isdir(os.path.join(self.dataset_folder, item))
            ]
        )

    def _list_sequence_folders(self, subject_folder: str) -> List[str]:
        subject_path = os.path.join(self.dataset_folder, subject_folder)
        return sorted(
            [
                item
                for item in os.listdir(subject_path)
                if item.startswith("202") and os.path.isdir(os.path.join(subject_path, item))
            ]
        )

    def _build_seq_idx_mapping(self) -> Dict[Tuple[int, int], Tuple[str, str]]:
        seq_idx_to_path: Dict[Tuple[int, int], Tuple[str, str]] = {}
        split_items = (
            self.split_config.get("train", [])
            + self.split_config.get("valid", [])
            + self.split_config.get("test", [])
        )

        seq_indices_by_subject: Dict[int, List[int]] = {}
        for subject_idx, seq_idx, _cam_idx, _frame_idx in split_items:
            seq_indices_by_subject.setdefault(subject_idx, []).append(seq_idx)

        for subject_idx, seq_indices in seq_indices_by_subject.items():
            subject_folder = self.subject_folders[subject_idx]
            available_sequences = set(self.sequence_folders_by_subject[subject_folder])
            for seq_idx in sorted(set(seq_indices)):
                if seq_idx >= len(self.hocap_sequence_ids):
                    continue
                seq_folder = self.hocap_sequence_ids[seq_idx]
                if seq_folder in available_sequences:
                    seq_idx_to_path[(subject_idx, seq_idx)] = (subject_folder, seq_folder)

        return seq_idx_to_path

    def _load_sequence_meta(self, subject_folder: str, seq_folder: str) -> Dict[str, Any]:
        cache_key = (subject_folder, seq_folder)
        if cache_key not in self._sequence_meta_cache:
            meta_path = os.path.join(self.dataset_folder, subject_folder, seq_folder, "meta.yaml")
            with open(meta_path, "r") as file:
                self._sequence_meta_cache[cache_key] = yaml.safe_load(file)
        return self._sequence_meta_cache[cache_key]

    def _load_all_subject_betas(self) -> Dict[str, torch.Tensor]:
        subject_betas: Dict[str, torch.Tensor] = {}
        for subject_folder in self.subject_folders:
            beta_path = os.path.join(self.dataset_folder, "calibration", "mano", f"{subject_folder}.yaml")
            with open(beta_path, "r") as file:
                beta_data = yaml.safe_load(file)
            subject_betas[subject_folder] = torch.tensor(beta_data["betas"], dtype=torch.float32)
        return subject_betas

    def _load_rs_cam_RTs(self):
        def create_mat(values):
            return np.array(
                [values[0:4], values[4:8], values[8:12], [0, 0, 0, 1]], dtype=np.float32
            )

        extrinsics_path = os.path.join(
            self.dataset_folder, "calibration", "extrinsics", "extrinsics_20231014.yaml"
        )
        with open(extrinsics_path, "r") as file:
            extrinsics = yaml.safe_load(file)["extrinsics"]

        tag_1 = create_mat(extrinsics["tag_1"])
        tag_1_inv = np.linalg.inv(tag_1)

        rs_world_to_camera = []
        for serial in self.hocap_device_serials:
            rs_world_to_camera.append(tag_1_inv @ create_mat(extrinsics[serial]))

        return rs_world_to_camera

    def _get_sequence_path(self, subject_idx: int, seq_idx: int) -> Tuple[str, str]:
        key = (subject_idx, seq_idx)
        if key not in self.seq_idx_to_path:
            raise ValueError(
                f"Cannot find sequence mapping for subject_idx={subject_idx}, seq_idx={seq_idx}."
            )

        subject_folder, seq_folder = self.seq_idx_to_path[key]

        return subject_folder, seq_folder

    def _resolve_existing_path(self, candidates: List[str], path_type: str) -> str:
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        joined = "\n".join(candidates)
        raise FileNotFoundError(f"Cannot find {path_type}. Tried:\n{joined}")

    def collect_sequences(self) -> List[dict]:
        sequences: List[dict] = []
        split_indices = self.split_config[self.split]

        sequence_map: Dict[Tuple[int, int, int], List[int]] = {}
        for subject_idx, seq_idx, cam_idx, frame_idx in split_indices:
            if self.view == "hololens_pv":
                key = (subject_idx, seq_idx, _HOLOLENS_FRAME_CAM_IDX)
            else:
                if cam_idx != self.realsense_cam_idx:
                    continue
                key = (subject_idx, seq_idx, cam_idx)
            sequence_map.setdefault(key, []).append(frame_idx)

        for (subject_idx, seq_idx, cam_key), frame_indices in sequence_map.items():
            frame_indices = sorted(set(frame_indices))
            if self.view == "hololens_pv":
                seq_id = f"sub{subject_idx:02d}_seq{seq_idx:03d}_hl"
                cam_idx_out = _HOLOLENS_FRAME_CAM_IDX
            else:
                seq_id = f"sub{subject_idx:02d}_seq{seq_idx:03d}_cam{cam_key:02d}"
                cam_idx_out = cam_key
            frame_ids = [[subject_idx, seq_idx, cam_idx_out, fi] for fi in frame_indices]
            sequences.append({"seq_id": seq_id, "frame_ids": frame_ids})

        return sequences

    def _resolve_frame(self, frame_id: List[int]) -> Tuple[str, str, str, int, int, int]:
        subject_idx, seq_idx, cam_idx, frame_idx = frame_id
        subject_folder, seq_folder = self._get_sequence_path(subject_idx, seq_idx)
        meta = self._load_sequence_meta(subject_folder, seq_folder)

        if cam_idx == _HOLOLENS_FRAME_CAM_IDX:
            camera_id = str(meta["hololens"]["serial"])
            image_candidates = [
                os.path.join(
                    self.dataset_folder,
                    subject_folder,
                    seq_folder,
                    camera_id,
                    f"color_{frame_idx:06d}.jpg",
                ),
                os.path.join(
                    self.dataset_folder,
                    subject_folder,
                    seq_folder,
                    camera_id,
                    f"color_{frame_idx:06d}.png",
                ),
            ]
            image_path = self._resolve_existing_path(image_candidates, "image file")
            return image_path, subject_folder, seq_folder, camera_id, subject_idx, seq_idx

        camera_serials = list(meta["realsense"]["serials"])
        if cam_idx < 0 or cam_idx >= len(camera_serials):
            raise ValueError(
                f"Camera index {cam_idx} out of range for subject={subject_idx}, seq={seq_idx}."
            )

        camera_id = camera_serials[cam_idx]
        image_candidates = [
            os.path.join(
                self.dataset_folder,
                subject_folder,
                seq_folder,
                camera_id,
                f"color_{frame_idx:06d}.jpg",
            ),
            os.path.join(
                self.dataset_folder,
                subject_folder,
                seq_folder,
                camera_id,
                f"color_{frame_idx:06d}.png",
            ),
        ]
        image_path = self._resolve_existing_path(image_candidates, "image file")
        return image_path, subject_folder, seq_folder, camera_id, subject_idx, seq_idx

    def convert_frame_id_to_image_path(self, frame_id: List[int]):
        image_path, _, _, _, _, _ = self._resolve_frame(frame_id)
        return image_path

    def _load_camera_world_to_camera(self, camera_id: str) -> np.ndarray:
        if camera_id not in self._camera_w2c_cache:
            camera_idx = self.hocap_device_serials.index(camera_id)
            self._camera_w2c_cache[camera_id] = self._rs_world_to_camera[camera_idx]
        return self._camera_w2c_cache[camera_id]

    def read_annos(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        anno: Dict[str, Any] = {}

        frame_id = frame["frame_id"]
        _image_path, subject_folder, seq_folder, camera_id, _subject_idx, _seq_idx = self._resolve_frame(frame_id)
        frame_idx = frame_id[3]
        is_hl = frame_id[2] == _HOLOLENS_FRAME_CAM_IDX

        meta = self._load_sequence_meta(subject_folder, seq_folder)
        mano_sides = meta["mano_sides"]
        object_ids = meta.get("object_ids", [])

        poses_candidates = [
            os.path.join(self.dataset_folder, "poses", subject_folder, seq_folder, "poses_m.npy"),
            os.path.join(self.dataset_folder, subject_folder, seq_folder, "poses_m.npy"),
        ]
        poses_m_path = self._resolve_existing_path(poses_candidates, "poses_m file")
        poses_m = np.load(poses_m_path).astype(np.float32)

        if is_hl:
            assert self._hl_cam_K_np is not None
            cam_K = self._hl_cam_K_np.copy()
            hand_joints_3d: Optional[np.ndarray] = None
            hand_joints_2d: Optional[np.ndarray] = None
        else:
            label_candidates = [
                os.path.join(
                    self.dataset_folder,
                    "labels",
                    subject_folder,
                    seq_folder,
                    camera_id,
                    f"label_{frame_idx:06d}.npz",
                ),
                os.path.join(
                    self.dataset_folder,
                    subject_folder,
                    seq_folder,
                    camera_id,
                    f"label_{frame_idx:06d}.npz",
                ),
            ]
            label_path = self._resolve_existing_path(label_candidates, "label file")
            label_data = np.load(label_path)
            cam_K = label_data["cam_K"].astype(np.float32)
            hand_joints_3d = label_data["hand_joints_3d"].astype(np.float32)
            hand_joints_2d = label_data["hand_joints_2d"].astype(np.float32)

        # HO-Cap source arrays are right-first: idx 0 -> right, idx 1 -> left.
        source_idx_from_side = {"left": 1, "right": 0}
        target_idx_from_side = {"left": 0, "right": 1}

        left_valid = "left" in mano_sides and not np.all(poses_m[source_idx_from_side["left"], frame_idx] == -1)
        right_valid = "right" in mano_sides and not np.all(poses_m[source_idx_from_side["right"], frame_idx] == -1)

        hand_side_mask = torch.zeros(2, dtype=torch.float32)
        if left_valid:
            hand_side_mask[0] = 1.0
        if right_valid:
            hand_side_mask[1] = 1.0

        if left_valid and right_valid:
            hand_side = 2
        elif left_valid:
            hand_side = 0
        elif right_valid:
            hand_side = 1
        else:
            hand_side = 2

        anno["hand_side"] = hand_side
        anno["hand_side_mask"] = hand_side_mask
        anno["object"] = "_".join(object_ids)

        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy = cam_K[0, 2], cam_K[1, 2]

        fx_resize = fx * self.resize_scale[0]
        fy_resize = fy * self.resize_scale[1]
        cx_resize = cx * self.resize_scale[0]
        cy_resize = cy * self.resize_scale[1]

        anno["intrinsics"] = torch.tensor([fx, fy, cx, cy], dtype=torch.float32).reshape(1, 4)
        anno["intrinsics_resize"] = torch.tensor(
            [fx_resize, fy_resize, cx_resize, cy_resize], dtype=torch.float32
        ).reshape(1, 4)

        world_to_camera = (
            self._hololens_world_to_camera(subject_folder, seq_folder, frame_idx)
            if is_hl
            else self._load_camera_world_to_camera(camera_id)
        )
        if is_hl:
            # Make HoloLens camera frame consistent with the OpenCV convention used by other datasets.
            world_to_camera = _HL_CVCAM_IN_GLCAM @ world_to_camera
        camera_to_world = np.linalg.inv(world_to_camera).astype(np.float32)
        anno["extrinsics"] = torch.from_numpy(camera_to_world[:3, :].reshape(1, 3, 4)).float()

        joints3d_hands = torch.zeros(2, 21, 3, dtype=torch.float32)
        joints2d_hands = torch.zeros(2, 21, 2, dtype=torch.float32)
        betas_hands = torch.zeros(2, 10, dtype=torch.float32)
        pose_hands = torch.zeros(2, 16, 6, dtype=torch.float32)
        transl_hands = torch.zeros(2, 3, dtype=torch.float32)

        subject_betas = self._mano_betas_by_subject[subject_folder]

        for side in ["left", "right"]:
            target_idx = target_idx_from_side[side]
            source_idx = source_idx_from_side[side]

            if hand_side_mask[target_idx] <= 0:
                continue

            mano_params = poses_m[source_idx, frame_idx]
            if np.all(mano_params == -1):
                continue

            mano_layer = self.mano_lhand if side == "left" else self.mano_rhand
            if not is_hl:
                assert hand_joints_3d is not None and hand_joints_2d is not None
                joints3d_gt = torch.from_numpy(hand_joints_3d[source_idx])
                joints2d_gt = torch.from_numpy(hand_joints_2d[source_idx])
            betas = subject_betas[None, :]

            global_rot_aa = torch.from_numpy(mano_params[:3].astype(np.float32))
            hand_pose_45 = torch.from_numpy(mano_params[3:48].astype(np.float32))
            transl_world = torch.from_numpy(mano_params[48:51].astype(np.float32)).view(1, 3)

            global_rot_mat = batch_rodrigues(global_rot_aa.reshape(1, 3)).view(1, 3, 3)
            global_rot_6d_world = matrix_to_rotation_6d(global_rot_mat).view(1, 6)
            hand_pose_aa = mano_layer.decode_pose_pca(hand_pose_45.view(1, 45)).view(45)
            hand_pose_mat = batch_rodrigues(hand_pose_aa.reshape(15, 3)).view(15, 3, 3)
            hand_pose_6d = matrix_to_rotation_6d(hand_pose_mat).view(1, 15, 6)

            wrist_offset = mano_layer(
                pose_type="rot6d",
                betas=betas,
                global_orient=global_rot_6d_world.view(1, 1, 6),
                hand_pose=hand_pose_6d,
            )["joints"][..., 0, :]

            global_rot_6d_camera_np, transl_camera_np = transfer_mano_params_A_to_B(
                world_to_camera,
                global_rot_6d_world.numpy(),
                transl_world.numpy(),
                wrist_offset.numpy(),
            )

            global_rot_6d_camera = torch.from_numpy(global_rot_6d_camera_np).view(1, 1, 6)
            transl_camera = torch.from_numpy(transl_camera_np).view(1, 3)

            pose_hands[target_idx, :1] = global_rot_6d_camera
            pose_hands[target_idx, 1:] = hand_pose_6d
            transl_hands[target_idx] = transl_camera[0]
            betas_hands[target_idx] = subject_betas

            if is_hl:
                mano_output = mano_layer(
                    pose_type="rot6d",
                    betas=betas,
                    global_orient=global_rot_6d_camera,
                    hand_pose=hand_pose_6d,
                    transl=transl_camera,
                )
                joints3d_cam = mano_output["joints"][0]
                joints3d_hands[target_idx] = joints3d_cam
                jc = joints3d_cam.detach().numpy()
                z = np.maximum(jc[:, 2], 1e-4)
                u = cam_K[0, 0] * (jc[:, 0] / z) + cam_K[0, 2]
                v = cam_K[1, 1] * (jc[:, 1] / z) + cam_K[1, 2]
                joints2d_hands[target_idx] = torch.from_numpy(np.stack([u, v], axis=-1).astype(np.float32))
            else:
                joints3d_hands[target_idx] = joints3d_gt
                joints2d_hands[target_idx] = joints2d_gt

        anno["betas"] = betas_hands
        anno["pose"] = pose_hands
        anno["transl"] = transl_hands
        anno["joints3d"] = joints3d_hands

        bbox_hands = torch.zeros(2, 4, dtype=torch.float32)
        for hand_idx in range(2):
            if hand_side_mask[hand_idx] <= 0:
                continue

            joints_2d = joints2d_hands[hand_idx]
            if not torch.any(joints_2d != 0):
                continue

            x_min, y_min = joints_2d.min(dim=0).values
            x_max, y_max = joints_2d.max(dim=0).values
            center_x = (x_min + x_max) * 0.5
            center_y = (y_min + y_max) * 0.5
            width = x_max - x_min
            height = y_max - y_min
            bbox_hands[hand_idx] = torch.tensor([center_x, center_y, width, height], dtype=torch.float32)

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

        src = np.array(
            [
                [center_x - width * 0.5, center_y - height * 0.5],
                [center_x - width * 0.5, center_y + height * 0.5],
                [center_x + width * 0.5, center_y - height * 0.5],
            ],
            dtype=np.float32,
        )
        dst = np.array(
            [[0.0, 0.0], [0.0, crop_h - 1.0], [crop_w - 1.0, 0.0]], dtype=np.float32
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

        assert os.path.exists(image_path), f"The image_path must exist, got {image_path}."
        image = cv2.imread(image_path)
        assert image is not None, f"Failed to read {image_path}."

        crop_size = (192, 256)
        crop_w, crop_h = crop_size

        if hand_side == 0:
            left_hand_image, left_crop_trans = self.crop_image_with_bbox(image, bbox_hands[0], crop_size)
            right_hand_image = np.zeros((int(crop_h), int(crop_w), 3), dtype=image.dtype)
            right_crop_trans = np.zeros((2, 3), dtype=np.float32)
        elif hand_side == 1:
            right_hand_image, right_crop_trans = self.crop_image_with_bbox(image, bbox_hands[1], crop_size)
            left_hand_image = np.zeros((int(crop_h), int(crop_w), 3), dtype=image.dtype)
            left_crop_trans = np.zeros((2, 3), dtype=np.float32)
        else:
            left_hand_image, left_crop_trans = self.crop_image_with_bbox(image, bbox_hands[0], crop_size)
            right_hand_image, right_crop_trans = self.crop_image_with_bbox(image, bbox_hands[1], crop_size)

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

        return {
            "image": image,
            "left_hand_image": left_hand_image,
            "right_hand_image": right_hand_image,
            "left_crop_trans": left_crop_trans,
            "right_crop_trans": right_crop_trans,
        }


if __name__ == "__main__":
    os.makedirs("check_visualization", exist_ok=True)

    hocap_dataset = HOCapRGBDataset(
        dataset_folder="/data/turuirui/Hocap_dataset",
        extra_anno_keys=["extrinsics"],
        image_size=(1280, 720),
        split="val",
        task="hpe",
        clip_len=1,
        view="hololens_pv",
    )

    sample = hocap_dataset.__getitem__(395)

    frame_id = sample["meta"]["frame_id"]
    print(f"[INFO] Visualizing sample with frame_id {frame_id}.")

    image_path = hocap_dataset.convert_frame_id_to_image_path(frame_id)
    image = cv2.imread(image_path)

    hand_side_mask = sample["hand_side_mask"]
    if float(hand_side_mask[0]) > 0.5:
        hand_side = 0
    elif float(hand_side_mask[1]) > 0.5:
        hand_side = 1
    else:
        raise RuntimeError(
            f"No valid hand in sample frame_id={frame_id}. "
            f"hand_side={sample['hand_side']}, hand_side_mask={hand_side_mask.tolist()}"
        )

    betas = sample["betas"]
    pose = sample["pose"]
    transl = sample["transl"]
    mano_layer = hocap_dataset.mano_lhand if hand_side == 0 else hocap_dataset.mano_rhand

    betas = betas[hand_side][None]
    global_orient = pose[hand_side, :1][None]
    hand_pose = pose[hand_side, 1:][None]
    transl = transl[hand_side][None]

    mano_output = mano_layer(
        pose_type="rot6d",
        betas=betas,
        global_orient=global_orient,
        hand_pose=hand_pose,
        transl=transl,
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
        camera_center=intrinsics[2:],
    )
    cv2.imwrite("check_visualization/hocap_render.png", render_bgr_with_image)
