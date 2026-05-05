from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import csv
import os

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import h5py
import numpy as np
from pathlib import Path
from collections import OrderedDict
import torch

from dataset.hand_rgb__2_0 import HandRGBDataset


DEFAULT_CAMERA_INTRINSIC = np.array(
    [
        [736.6339, 0.0, 960.0],
        [0.0, 736.6339, 540.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


# Map EgoDex transforms to 21-keypoint hand topology:
# wrist + (index/middle/pinky/ring: 4 joints each) + (thumb: 4 joints).
LEFT_HAND_TRANSFORMS = [
    "leftHand",
    "leftIndexFingerKnuckle",
    "leftIndexFingerIntermediateBase",
    "leftIndexFingerIntermediateTip",
    "leftIndexFingerTip",
    "leftMiddleFingerKnuckle",
    "leftMiddleFingerIntermediateBase",
    "leftMiddleFingerIntermediateTip",
    "leftMiddleFingerTip",
    "leftLittleFingerKnuckle",
    "leftLittleFingerIntermediateBase",
    "leftLittleFingerIntermediateTip",
    "leftLittleFingerTip",
    "leftRingFingerKnuckle",
    "leftRingFingerIntermediateBase",
    "leftRingFingerIntermediateTip",
    "leftRingFingerTip",
    "leftThumbKnuckle",
    "leftThumbIntermediateBase",
    "leftThumbIntermediateTip",
    "leftThumbTip",
]

RIGHT_HAND_TRANSFORMS = [
    "rightHand",
    "rightIndexFingerKnuckle",
    "rightIndexFingerIntermediateBase",
    "rightIndexFingerIntermediateTip",
    "rightIndexFingerTip",
    "rightMiddleFingerKnuckle",
    "rightMiddleFingerIntermediateBase",
    "rightMiddleFingerIntermediateTip",
    "rightMiddleFingerTip",
    "rightLittleFingerKnuckle",
    "rightLittleFingerIntermediateBase",
    "rightLittleFingerIntermediateTip",
    "rightLittleFingerTip",
    "rightRingFingerKnuckle",
    "rightRingFingerIntermediateBase",
    "rightRingFingerIntermediateTip",
    "rightRingFingerTip",
    "rightThumbKnuckle",
    "rightThumbIntermediateBase",
    "rightThumbIntermediateTip",
    "rightThumbTip",
]


class EgoDexRGBDataset(HandRGBDataset):
    def __init__(
        self,
        dataset_folder: str,
        split: Literal["train", "val", "test"] = "train",
        setup: Literal["default"] = "default",
        sequence_length: int = 1,
        **kwargs,
    ):
        self.dataset_folder = dataset_folder
        self.split = split
        self.setup = setup
        self.sequence_length = sequence_length

        # EgoDex RGB is 1920x1080.
        self.raw_image_size = (1920, 1080)

        self.cache_dir = Path(__file__).parent.parent / "_DATA"
        self.sequence_cache_file = self.cache_dir / "sequences_cache" / f"egodex_{setup}_{split}.pkl"
        self.frame_cache_dir = self.cache_dir / "frame_cache" / f"egodex_{setup}_{split}"

        self._prompt_by_hdf5: Dict[str, str] = {}
        # Per-worker handle caches to avoid expensive open/close on every sample.
        # Each DataLoader worker owns its dataset instance, so these caches are process-local.
        self._h5_cache: "OrderedDict[str, h5py.File]" = OrderedDict()
        self._video_cache: "OrderedDict[str, cv2.VideoCapture]" = OrderedDict()
        self._max_open_h5 = 64
        self._max_open_video = 8

        print(f"[INFO] Loading EgoDex dataset with setup {setup}-{split}.")

        super().__init__(sequence_length=sequence_length, **kwargs)

    def _metadata_file_for_split(self) -> Path:
        if self.split == "train":
            return Path("/data/sunzhihao2/turuirui/egodex_metadata/egodex_metadata.csv")
        if self.split in ("val", "test"):
            return Path("/data/sunzhihao2/turuirui/egodex_metadata/egodex_metadata_test.csv")

        raise ValueError(f"Unsupported split: {self.split}")

    def _parse_frame_id(self, frame_id: str) -> Tuple[str, str, int]:
        hdf5_path, frame_idx_str = frame_id.split("::")
        mp4_path = hdf5_path[:-5] + ".mp4"
        return hdf5_path, mp4_path, int(frame_idx_str)

    def collect_sequences(self) -> List[dict]:
        sequences: List[dict] = []

        metadata_file = self._metadata_file_for_split()
        if not metadata_file.exists():
            raise FileNotFoundError(f"EgoDex metadata file does not exist: {metadata_file}")

        with open(metadata_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_rel = row["video"].strip()
                prompt = row.get("prompt", "").strip()

                mp4_path = Path(self.dataset_folder) / video_rel
                hdf5_path = mp4_path.with_suffix(".hdf5")

                if (not mp4_path.exists()) or (not hdf5_path.exists()):
                    continue

                with h5py.File(str(hdf5_path), "r") as root_file:
                    if "/transforms/camera" not in root_file:
                        continue
                    num_frames = int(root_file["/transforms/camera"].shape[0])

                if num_frames <= 0:
                    continue

                frame_ids = [f"{hdf5_path}::{frame_idx}" for frame_idx in range(num_frames)]

                seq_id = str(mp4_path.relative_to(self.dataset_folder)).replace(".mp4", "")
                sequences.append({"seq_id": seq_id, "frame_ids": frame_ids})

                self._prompt_by_hdf5[str(hdf5_path)] = prompt

        return sequences

    def convert_frame_id_to_image_path(self, frame_id: str):
        _, mp4_path, _ = self._parse_frame_id(frame_id)
        return mp4_path

    def _read_video_frame(self, mp4_path: str, frame_idx: int) -> np.ndarray:
        cap = self._get_video_capture(mp4_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()

        if (not ok) or (frame_bgr is None):
            raise RuntimeError(f"Failed to decode frame {frame_idx} from {mp4_path}")

        return frame_bgr

    def _frame_cache_path(self, mp4_path: str, frame_idx: int) -> Path:
        mp4_path_obj = Path(mp4_path)
        try:
            rel_video_path = mp4_path_obj.relative_to(self.dataset_folder)
        except ValueError:
            rel_video_path = Path(mp4_path_obj.name)

        video_stem_rel = rel_video_path.with_suffix("")
        return self.frame_cache_dir / video_stem_rel / f"{frame_idx:08d}.jpg"

    def _read_or_build_cached_frame(self, mp4_path: str, frame_idx: int) -> np.ndarray:
        cache_path = self._frame_cache_path(mp4_path, frame_idx)
        if cache_path.exists():
            cached = cv2.imread(str(cache_path), cv2.IMREAD_COLOR)
            if cached is not None:
                return cached

        frame_bgr = self._read_video_frame(mp4_path, frame_idx)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
        try:
            ok = cv2.imwrite(str(tmp_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ok:
                os.replace(tmp_path, cache_path)
        except Exception:
            # Best-effort cache write; reading path should remain robust even if cache write fails.
            pass
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        return frame_bgr

    def _evict_oldest_h5(self):
        if len(self._h5_cache) <= self._max_open_h5:
            return
        _, handle = self._h5_cache.popitem(last=False)
        try:
            handle.close()
        except Exception:
            pass

    def _evict_oldest_video(self):
        if len(self._video_cache) <= self._max_open_video:
            return
        _, cap = self._video_cache.popitem(last=False)
        try:
            cap.release()
        except Exception:
            pass

    def _get_h5_handle(self, hdf5_path: str) -> h5py.File:
        handle = self._h5_cache.get(hdf5_path, None)
        if handle is not None and handle.id.valid:
            self._h5_cache.move_to_end(hdf5_path)
            return handle

        # Re-open stale/missing handles.
        handle = h5py.File(hdf5_path, "r")
        self._h5_cache[hdf5_path] = handle
        self._h5_cache.move_to_end(hdf5_path)
        self._evict_oldest_h5()
        return handle

    def _get_video_capture(self, mp4_path: str) -> cv2.VideoCapture:
        cap = self._video_cache.get(mp4_path, None)
        if cap is not None and cap.isOpened():
            self._video_cache.move_to_end(mp4_path)
            return cap

        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {mp4_path}")

        self._video_cache[mp4_path] = cap
        self._video_cache.move_to_end(mp4_path)
        self._evict_oldest_video()
        return cap

    def _close_caches(self):
        for _, handle in self._h5_cache.items():
            try:
                handle.close()
            except Exception:
                pass
        self._h5_cache.clear()

        for _, cap in self._video_cache.items():
            try:
                cap.release()
            except Exception:
                pass
        self._video_cache.clear()

    def __del__(self):
        self._close_caches()

    @staticmethod
    def _to_camera_frame(world_tfs: np.ndarray, cam_ext: np.ndarray) -> np.ndarray:
        # Both are in ARKit world frame; convert hand transforms into current camera frame.
        return np.linalg.inv(cam_ext)[None] @ world_tfs

    @staticmethod
    def _project_points(points_cam: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        z = np.maximum(points_cam[:, 2], 1e-6)
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        u = fx * (points_cam[:, 0] / z) + cx
        v = fy * (points_cam[:, 1] / z) + cy
        return np.stack([u, v], axis=-1).astype(np.float32)

    @staticmethod
    def _normalize_intrinsics(cam_int: np.ndarray) -> np.ndarray:
        # Some episodes contain empty/malformed intrinsics; use a safe default in that case.
        if cam_int is None:
            return DEFAULT_CAMERA_INTRINSIC.copy()

        cam_int = np.asarray(cam_int, dtype=np.float32)

        if cam_int.size == 0:
            return DEFAULT_CAMERA_INTRINSIC.copy()

        if cam_int.shape == (3, 3):
            return cam_int

        if cam_int.size >= 4:
            fx, fy, cx, cy = cam_int.reshape(-1)[:4]
            intr = np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            return intr

        return DEFAULT_CAMERA_INTRINSIC.copy()

    @staticmethod
    def _bbox_from_joints2d(joints2d: np.ndarray, image_w: int, image_h: int) -> torch.Tensor:
        x_min, y_min = joints2d.min(axis=0)
        x_max, y_max = joints2d.max(axis=0)

        x_min = np.clip(x_min, 0, image_w - 1)
        x_max = np.clip(x_max, 0, image_w - 1)
        y_min = np.clip(y_min, 0, image_h - 1)
        y_max = np.clip(y_max, 0, image_h - 1)

        width = float(max(0.0, x_max - x_min))
        height = float(max(0.0, y_max - y_min))
        cx = float((x_min + x_max) * 0.5)
        cy = float((y_min + y_max) * 0.5)

        if width < 2.0 or height < 2.0:
            return torch.zeros(4, dtype=torch.float32)

        return torch.tensor([cx, cy, width, height], dtype=torch.float32)

    def read_annos(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        anno: Dict[str, Any] = {}

        frame_id = frame["frame_id"]
        hdf5_path, _, frame_idx = self._parse_frame_id(frame_id)

        root_file = self._get_h5_handle(hdf5_path)
        cam_ext = root_file["/transforms/camera"][frame_idx].astype(np.float32)
        cam_int_raw = root_file["/camera/intrinsic"][:]
        cam_int = self._normalize_intrinsics(cam_int_raw)

        left_world = np.stack(
            [root_file[f"/transforms/{name}"][frame_idx] for name in LEFT_HAND_TRANSFORMS],
            axis=0,
        ).astype(np.float32)
        right_world = np.stack(
            [root_file[f"/transforms/{name}"][frame_idx] for name in RIGHT_HAND_TRANSFORMS],
            axis=0,
        ).astype(np.float32)

        left_conf = 1.0
        right_conf = 1.0
        if "/confidences/leftHand" in root_file:
            left_conf = float(root_file["/confidences/leftHand"][frame_idx])
        if "/confidences/rightHand" in root_file:
            right_conf = float(root_file["/confidences/rightHand"][frame_idx])

        left_cam = self._to_camera_frame(left_world, cam_ext)
        right_cam = self._to_camera_frame(right_world, cam_ext)

        left_joints3d = left_cam[:, :3, 3].astype(np.float32)
        right_joints3d = right_cam[:, :3, 3].astype(np.float32)

        left_joints2d = self._project_points(left_joints3d, cam_int)
        right_joints2d = self._project_points(right_joints3d, cam_int)

        left_valid = np.all(np.isfinite(left_joints3d)) and np.all(np.isfinite(left_joints2d)) and left_conf > 0.1
        right_valid = np.all(np.isfinite(right_joints3d)) and np.all(np.isfinite(right_joints2d)) and right_conf > 0.1

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

        # EgoDex does not provide MANO labels. Keep MANO supervision mask zero so pose/transl losses are skipped.
        anno["mano_side_mask"] = torch.zeros(2, dtype=torch.float32)

        fx = float(cam_int[0, 0])
        fy = float(cam_int[1, 1])
        cx = float(cam_int[0, 2])
        cy = float(cam_int[1, 2])

        fx_resize = fx * self.resize_scale[0]
        fy_resize = fy * self.resize_scale[1]
        cx_resize = cx * self.resize_scale[0]
        cy_resize = cy * self.resize_scale[1]

        anno["intrinsics"] = torch.tensor([fx, fy, cx, cy], dtype=torch.float32).reshape(1, 4)
        anno["intrinsics_resize"] = torch.tensor([fx_resize, fy_resize, cx_resize, cy_resize], dtype=torch.float32).reshape(1, 4)

        anno["extrinsics"] = torch.from_numpy(cam_ext[:3, :]).reshape(1, 3, 4).float()

        joints3d_hands = torch.zeros(2, 21, 3, dtype=torch.float32)
        joints2d_hands = torch.zeros(2, 21, 2, dtype=torch.float32)

        if left_valid:
            joints3d_hands[0] = torch.from_numpy(left_joints3d)
            joints2d_hands[0] = torch.from_numpy(left_joints2d)
        if right_valid:
            joints3d_hands[1] = torch.from_numpy(right_joints3d)
            joints2d_hands[1] = torch.from_numpy(right_joints2d)

        anno["joints3d"] = joints3d_hands
        anno["joints2d"] = joints2d_hands

        # Placeholders required by the shared model interface.
        anno["betas"] = torch.zeros(2, 10, dtype=torch.float32)
        anno["pose"] = torch.zeros(2, 16, 6, dtype=torch.float32)
        anno["transl"] = torch.zeros(2, 3, dtype=torch.float32)

        image_w, image_h = self.raw_image_size
        bbox_hands = torch.zeros(2, 4, dtype=torch.float32)
        if left_valid:
            bbox_hands[0] = self._bbox_from_joints2d(left_joints2d, image_w=image_w, image_h=image_h)
        if right_valid:
            bbox_hands[1] = self._bbox_from_joints2d(right_joints2d, image_w=image_w, image_h=image_h)
        anno["bbox"] = bbox_hands

        seq_prompt = self._prompt_by_hdf5.get(hdf5_path, "")
        if not seq_prompt:
            seq_prompt = Path(hdf5_path).parent.name
        anno["object"] = seq_prompt

        return anno

    @staticmethod
    def _build_extrinsics_4x4(extrinsics: torch.Tensor) -> np.ndarray:
        extrinsics_4x4 = np.eye(4, dtype=np.float32)
        extrinsics_4x4[:3, :] = extrinsics.squeeze(0).cpu().numpy()
        return extrinsics_4x4

    def convert_sequence_annos_to_canonical(
        self,
        frames: Sequence[Dict[str, Any]],
        annos: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        annos = [dict(anno) for anno in annos]

        if len(annos) == 0:
            return annos

        T_canonical_to_world = self._build_extrinsics_4x4(annos[0]["extrinsics"])
        T_world_to_canonical = np.linalg.inv(T_canonical_to_world).astype(np.float32)

        for anno in annos:
            curr_extrinsics = self._build_extrinsics_4x4(anno["extrinsics"])
            T_curr_to_canonical = (T_world_to_canonical @ curr_extrinsics).astype(np.float32)

            joints3d = anno["joints3d"]
            hand_side_mask = anno["hand_side_mask"]

            T_curr_to_canonical_torch = torch.from_numpy(T_curr_to_canonical).to(
                dtype=joints3d.dtype,
                device=joints3d.device,
            )

            for hand_index in range(2):
                if float(hand_side_mask[hand_index].item()) <= 0.0:
                    continue

                joints = joints3d[hand_index]
                joints_homo = torch.cat(
                    [joints, torch.ones((joints.shape[0], 1), dtype=joints.dtype, device=joints.device)],
                    dim=-1,
                )
                joints_canonical = (joints_homo @ T_curr_to_canonical_torch.T)[:, :3]
                joints3d[hand_index] = joints_canonical

            anno["joints3d"] = joints3d
            anno["extrinsics"] = torch.from_numpy(T_curr_to_canonical[:3, :]).reshape(1, 3, 4).float()

        return annos

    @staticmethod
    def scale_bbox(bbox, aspect_ratio=(192.0, 256.0), scale_factor=1.5):
        center_x, center_y, width, height = bbox

        target_ratio = float(aspect_ratio[0]) / float(aspect_ratio[1])
        if width / max(height, 1e-6) > target_ratio:
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
        _, mp4_path, frame_idx = self._parse_frame_id(frame_id)

        hand_side = anno["hand_side"]
        bbox_hands = anno["bbox"]

        image = self._read_or_build_cached_frame(mp4_path, frame_idx)

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

        image_dict = {
            "image": image,
            "left_hand_image": left_hand_image,
            "right_hand_image": right_hand_image,
            "left_crop_trans": left_crop_trans,
            "right_crop_trans": right_crop_trans,
        }

        return image_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick sanity test for EgoDexRGBDataset")
    parser.add_argument("--dataset_folder", type=str, default="/data/public/ego/egodex")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--sequence_length", type=int, default=1)
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--vis_output_dir", type=str, default="check_visualization")
    args = parser.parse_args()

    dataset = EgoDexRGBDataset(
        dataset_folder=args.dataset_folder,
        split=args.split,
        setup="default",
        sequence_length=args.sequence_length,
        image_size=tuple(args.image_size),
        seed=42,
    )

    print(f"[CHECK] dataset len = {len(dataset)}")

    index = max(0, min(args.sample_index, len(dataset) - 1))
    sample = dataset[index]

    print(f"[CHECK] sample index = {index}")
    for key in [
        "image",
        "left_hand_image",
        "right_hand_image",
        "joints3d",
        "intrinsics",
        "extrinsics",
        "hand_side_mask",
        "mano_side_mask",
    ]:
        value = sample[key]
        if isinstance(value, torch.Tensor):
            print(f"[CHECK] {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"[CHECK] {key}: type={type(value)}")

    os.makedirs(args.vis_output_dir, exist_ok=True)

    def _is_temporal_sample(sample_dict: Dict[str, Any]) -> bool:
        meta = sample_dict.get("meta", {})
        return isinstance(meta, dict) and isinstance(meta.get("frame_ids", None), list)

    def _get_frame_tensor(sample_dict: Dict[str, Any], key: str, frame_idx: int):
        value = sample_dict[key]
        if not torch.is_tensor(value):
            return value
        if _is_temporal_sample(sample_dict) and value.ndim >= 1:
            return value[frame_idx]
        return value

    def _project_joints3d(joints3d_np: np.ndarray, intrinsics_np: np.ndarray) -> np.ndarray:
        z = np.maximum(joints3d_np[:, 2], 1e-6)
        fx, fy, cx, cy = intrinsics_np
        u = fx * (joints3d_np[:, 0] / z) + cx
        v = fy * (joints3d_np[:, 1] / z) + cy
        return np.stack([u, v], axis=-1)

    def _get_vis_joints2d(sample_dict: Dict[str, Any], frame_id: str, frame_idx: int):
        if "joints2d" in sample_dict:
            return _get_frame_tensor(sample_dict, "joints2d", frame_idx)
        anno_for_vis = dataset.read_annos({"frame_id": frame_id})
        return anno_for_vis["joints2d"]

    vis_frame_idx = 0
    if _is_temporal_sample(sample):
        vis_frame_id = sample["meta"]["frame_ids"][vis_frame_idx]
        hand_side_mask = _get_frame_tensor(sample, "hand_side_mask", vis_frame_idx)
        joints3d = _get_frame_tensor(sample, "joints3d", vis_frame_idx)
        joints2d = _get_vis_joints2d(sample, vis_frame_id, vis_frame_idx)
        intrinsics = _get_frame_tensor(sample, "intrinsics", vis_frame_idx)
    else:
        vis_frame_id = sample["meta"]["frame_id"]
        hand_side_mask = sample["hand_side_mask"]
        joints3d = sample["joints3d"]
        joints2d = _get_vis_joints2d(sample, vis_frame_id, vis_frame_idx)
        intrinsics = sample["intrinsics"]

    _, vis_mp4_path, vis_frame_number = dataset._parse_frame_id(vis_frame_id)
    vis_image_bgr = dataset._read_or_build_cached_frame(vis_mp4_path, vis_frame_number).copy()

    hand_side_mask_np = hand_side_mask.detach().cpu().numpy()
    joints3d_np = joints3d.detach().cpu().numpy()
    joints2d_np = joints2d.detach().cpu().numpy()
    intrinsics_np = intrinsics.detach().cpu().numpy().reshape(-1)

    for hand_idx in range(2):
        if float(hand_side_mask_np[hand_idx]) <= 0.0:
            continue

        proj_2d = _project_joints3d(joints3d_np[hand_idx], intrinsics_np)
        gt_2d = joints2d_np[hand_idx]
        reproj_error = np.linalg.norm(proj_2d - gt_2d, axis=1).mean()
        print(f"[VIS] hand={hand_idx} reprojection error(px)={reproj_error:.3f}")

        for kp in gt_2d:
            x, y = int(round(kp[0])), int(round(kp[1]))
            if 0 <= x < vis_image_bgr.shape[1] and 0 <= y < vis_image_bgr.shape[0]:
                cv2.circle(vis_image_bgr, (x, y), 3, (0, 255, 0), -1)

        for kp in proj_2d:
            x, y = int(round(kp[0])), int(round(kp[1]))
            if 0 <= x < vis_image_bgr.shape[1] and 0 <= y < vis_image_bgr.shape[0]:
                cv2.circle(vis_image_bgr, (x, y), 2, (0, 0, 255), -1)

    vis_save_path = os.path.join(
        args.vis_output_dir,
        f"egodex_{args.split}_idx{index}_f{vis_frame_number:06d}_kp3d_proj.jpg",
    )
    cv2.imwrite(vis_save_path, vis_image_bgr)
    print(f"[VIS] saved visualization to {vis_save_path}")

    limit = min(args.num_samples, len(dataset))
    for i in range(limit):
        _ = dataset[i]
    print(f"[CHECK] sequential sample read passed for first {limit} samples")

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch,
    )
    _ = next(iter(loader))
    print("[CHECK] dataloader first batch passed")
