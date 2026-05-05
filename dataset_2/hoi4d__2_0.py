from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import os
import pickle

import pyrootutils
root = pyrootutils.setup_root(
	search_from=__file__,
	indicator=["README.md"],
	pythonpath=True,
	dotenv=True,
)

import cv2
import numpy as np
from pathlib import Path
from smplx.lbs import batch_rodrigues
import torch

from dataset.hand_rgb__2_0 import HandRGBDataset
from model.smpl.transfer import transfer_mano_params_A_to_B
from model.smpl.utils.rot6d import matrix_to_rotation_6d


class HOI4DEgoRGBDataset(HandRGBDataset):
	def __init__(
		self,
		dataset_folder: str,
		split: Literal["train", "val", "test"] = "train",
		setup: Literal["default"] = "default",
		sequence_length: int = 8,
		camera_cache_file=None,
		**kwargs,
	):
		self.dataset_folder = dataset_folder
		self.split = split
		self.setup = setup
		self.sequence_length = sequence_length
		self.camera_cache_file = camera_cache_file

		# Keep base resize_scale simple and compute per-frame pseudo intrinsics later.
		self.raw_image_size = None

		self.cache_dir = Path(__file__).parent.parent / "_DATA"
		self.sequence_cache_file = self.cache_dir / "sequences_cache" / f"hoi4d_{setup}_{split}.pkl"
		self.camera_cache = self._load_camera_cache(self.camera_cache_file)

		self._image_path_cache: Dict[str, str] = {}
		self._intrinsics_cache: Dict[str, torch.Tensor] = {}
		self._extrinsics_cache: Dict[str, Dict[int, torch.Tensor]] = {}

		print(f"[INFO] Loading HOI4D dataset with setup {setup}-{split}.")

		super().__init__(sequence_length=sequence_length, **kwargs)

	def _frame_idx_to_image_path(self, seq_rel_path: str, frame_idx: int) -> str:
		if seq_rel_path in self._image_path_cache:
			template = self._image_path_cache[seq_rel_path]
			return template.format(frame_idx)

		align_rgb_dir = Path(self.dataset_folder) / "HOI4D_release" / seq_rel_path / "align_rgb"
		if not align_rgb_dir.exists():
			raise FileNotFoundError(f"Missing align_rgb folder for {seq_rel_path}: {align_rgb_dir}")

		candidate = align_rgb_dir / f"{frame_idx:05d}.jpg"
		if candidate.exists():
			self._image_path_cache[seq_rel_path] = str(align_rgb_dir / f"{{:05d}}.jpg")
			return str(candidate)

		raise FileNotFoundError(f"Missing frame image for {seq_rel_path} frame {frame_idx}: {candidate}")

	def _split_sequences(self, seq_ids: List[str]) -> List[str]:
		if self.split == "train":
			# HOI4D does not provide an official split in this layout.
			# Use all sequences for training by default.
			return seq_ids
		return []

	def collect_sequences(self) -> List[dict]:
		"""
		Return a list of sequences, each like:
			{
				"seq_id": "ZY20210800002/H2/C20/N13/S275/s01/T2",
				"frame_ids": [
					"/path/to/HOI4D_release/.../align_rgb/00155.jpg",
					...
				]
			}
		Only includes frames where at least one hand has valid data.
		"""
		if self.setup != "default":
			raise ValueError(f"The setup {self.setup} is not supported.")

		sequences: List[dict] = []

		left_root = Path(self.dataset_folder) / "Hand_pose" / "handpose_left_hand"
		right_root = Path(self.dataset_folder) / "Hand_pose" / "handpose_right_hand"

		if not left_root.exists() and not right_root.exists():
			raise FileNotFoundError(
				"Cannot find HOI4D hand pose folders. Expected Hand_pose/handpose_left_hand or handpose_right_hand."
			)

		# Track which frames have valid left/right hand data
		seq_to_frames: Dict[str, Dict[int, Tuple[bool, bool]]] = {}

		for side_root in (left_root, right_root):
			if not side_root.exists():
				continue

			side = "left" if "left" in side_root.name else "right"
			for pkl_path in side_root.rglob("*.pickle"):
				rel_seq = str(pkl_path.relative_to(side_root).parent)
				frame_idx = int(pkl_path.stem)

				if rel_seq not in seq_to_frames:
					seq_to_frames[rel_seq] = {}
				if frame_idx not in seq_to_frames[rel_seq]:
					seq_to_frames[rel_seq][frame_idx] = (False, False)

				left_valid, right_valid = seq_to_frames[rel_seq][frame_idx]
				if side == "left":
					seq_to_frames[rel_seq][frame_idx] = (True, right_valid)
				else:
					seq_to_frames[rel_seq][frame_idx] = (left_valid, True)

		seq_ids = sorted(seq_to_frames.keys())
		seq_ids = self._split_sequences(seq_ids)

		for seq_id in seq_ids:
			frame_data = seq_to_frames[seq_id]
			frame_ids = []

			for frame_idx in sorted(frame_data.keys()):
				left_valid, right_valid = frame_data[frame_idx]
				
				# Only include frames where at least one hand is valid
				if not (left_valid or right_valid):
					continue

				try:
					image_path = self._frame_idx_to_image_path(seq_id, frame_idx)
				except FileNotFoundError:
					continue
				frame_ids.append(image_path)

			if len(frame_ids) == 0:
				continue

			sequences.append({"seq_id": seq_id, "frame_ids": frame_ids})

		return sequences

	def convert_frame_id_to_image_path(self, frame_id: str):
		return frame_id

	def _get_pickle_paths(self, seq_id: str, frame_idx: int):
		left_path = Path(self.dataset_folder) / "Hand_pose" / "handpose_left_hand" / seq_id / f"{frame_idx}.pickle"
		right_path = Path(self.dataset_folder) / "Hand_pose" / "handpose_right_hand" / seq_id / f"{frame_idx}.pickle"
		return left_path, right_path

	def _build_bbox_from_kps2d(self, kps2d: np.ndarray) -> torch.Tensor:
		if kps2d is None or kps2d.size == 0:
			return torch.zeros(4, dtype=torch.float32)

		valid = np.isfinite(kps2d).all(axis=1)
		kps = kps2d[valid]
		if kps.shape[0] == 0:
			return torch.zeros(4, dtype=torch.float32)

		x1, y1 = np.min(kps[:, 0]), np.min(kps[:, 1])
		x2, y2 = np.max(kps[:, 0]), np.max(kps[:, 1])

		w = max(float(x2 - x1), 1.0)
		h = max(float(y2 - y1), 1.0)
		cx = float(x1 + x2) * 0.5
		cy = float(y1 + y2) * 0.5

		return torch.tensor([cx, cy, w, h], dtype=torch.float32)

	def _decode_pose_coeff(self, pose_coeff: np.ndarray) -> torch.Tensor:
		pose_coeff = torch.from_numpy(pose_coeff.astype(np.float32)).view(16, 3)
		pose_mat = batch_rodrigues(pose_coeff).view(16, 3, 3)
		pose_6d = matrix_to_rotation_6d(pose_mat).view(16, 6)
		return pose_6d

	def _get_intrinsics(self, seq_id: str, image_path: str):
		seq_root = Path(seq_id).parts[0] if len(Path(seq_id).parts) > 0 else seq_id

		if seq_root in self._intrinsics_cache:
			intrinsics = self._intrinsics_cache[seq_root]
		else:
			intrin_path = Path(self.dataset_folder) / "camera_params" / seq_root / "intrin.npy"
			if not intrin_path.exists():
				raise FileNotFoundError(f"Missing intrinsics file for {seq_root}: {intrin_path}")

			intrin = np.load(intrin_path)
			if intrin.shape != (3, 3):
				raise RuntimeError(f"Unexpected intrinsics shape for {seq_root}: {intrin.shape}")

			fx = float(intrin[0, 0])
			fy = float(intrin[1, 1])
			cx = float(intrin[0, 2])
			cy = float(intrin[1, 2])
			intrinsics = torch.tensor([fx, fy, cx, cy], dtype=torch.float32).reshape(1, 4)
			self._intrinsics_cache[seq_root] = intrinsics

		fx, fy, cx, cy = intrinsics.view(-1).tolist()
		resize_w, resize_h = self.image_size

		image = cv2.imread(image_path)
		if image is None:
			raise RuntimeError(f"Failed to read image for intrinsics resize: {image_path}")
		raw_h, raw_w = image.shape[:2]

		sx = resize_w / float(raw_w)
		sy = resize_h / float(raw_h)
		intrinsics_resize = torch.tensor([fx * sx, fy * sy, cx * sx, cy * sy], dtype=torch.float32).reshape(1, 4)

		return intrinsics, intrinsics_resize

	def _get_output_log_path(self, seq_id: str) -> Path:
		return Path(self.dataset_folder) / "HOI4D_annotations" / seq_id / "3Dseg" / "output.log"

	def _load_sequence_extrinsics_from_log(self, seq_id: str) -> Dict[int, torch.Tensor]:
		output_log_path = self._get_output_log_path(seq_id)
		if not output_log_path.exists():
			return {}

		with open(output_log_path, "r", encoding="utf-8") as f:
			lines = [line.strip() for line in f if line.strip()]

		extrinsics_map: Dict[int, torch.Tensor] = {}
		i = 0
		while i + 4 < len(lines):
			header_tokens = lines[i].split()
			try:
				frame_idx = int(float(header_tokens[0]))
			except (ValueError, IndexError):
				i += 1
				continue

			matrix_rows = []
			valid_block = True
			for row_offset in range(1, 5):
				row_tokens = lines[i + row_offset].split()
				if len(row_tokens) != 4:
					valid_block = False
					break
				try:
					matrix_rows.append([float(v) for v in row_tokens])
				except ValueError:
					valid_block = False
					break

			if valid_block:
				extrinsic_w2c_4x4 = np.asarray(matrix_rows, dtype=np.float32)
				# Open3D trajectory stores world-to-camera extrinsics; convert to camera-to-world.
				extrinsic_c2w_4x4 = np.linalg.inv(extrinsic_w2c_4x4)
				extrinsics_map[frame_idx] = torch.from_numpy(extrinsic_c2w_4x4[:3, :]).reshape(1, 3, 4)
				i += 5
			else:
				i += 1

		return extrinsics_map

	def _get_extrinsics(self, seq_id: str, frame_idx: int) -> torch.Tensor:
		if seq_id not in self._extrinsics_cache:
			self._extrinsics_cache[seq_id] = self._load_sequence_extrinsics_from_log(seq_id)

		extrinsics_map = self._extrinsics_cache[seq_id]
		if frame_idx in extrinsics_map:
			return extrinsics_map[frame_idx]

		# Some sequences can be indexed with an off-by-one frame convention.
		if (frame_idx - 1) in extrinsics_map:
			return extrinsics_map[frame_idx - 1]
		if (frame_idx + 1) in extrinsics_map:
			return extrinsics_map[frame_idx + 1]

		return torch.tensor(
			[[[1.0, 0.0, 0.0, 0.0],
			  [0.0, 1.0, 0.0, 0.0],
			  [0.0, 0.0, 1.0, 0.0]]],
			dtype=torch.float32,
		)

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

	def _validate_pose_coefficients(self, pose_coeff: np.ndarray, hand_name: str, seq_id: str, frame_idx: int) -> bool:
		"""
		Validate that pose coefficients have the expected size (48).
		Returns True if valid, False otherwise.
		"""
		if pose_coeff is None:
			return False
		pose_coeff_array = np.asarray(pose_coeff, dtype=np.float32)
		if pose_coeff_array.size != 48:
			import logging
			logger = logging.getLogger(__name__)
			logger.warning(
				f"[{hand_name}] Invalid pose coefficient size: {pose_coeff_array.size} (expected 48) "
				f"for seq={seq_id}, frame={frame_idx}. Skipping this hand."
			)
			return False
		return True

	def read_annos(self, frame: Dict[str, Any]) -> Dict[str, Any]:
		anno: Dict[str, Any] = {}

		seq_id = frame["seq_id"]
		frame_id = frame["frame_id"]
		image_path = self.convert_frame_id_to_image_path(frame_id)
		frame_idx = int(Path(image_path).stem)

		left_pickle_path, right_pickle_path = self._get_pickle_paths(seq_id, frame_idx)

		left_valid = left_pickle_path.exists()
		right_valid = right_pickle_path.exists()

		# Note: collect_sequences ensures at least one hand is valid per frame.
		# This check serves as a safety net for inconsistent data state.
		if not left_valid and not right_valid:
			import logging
			logger = logging.getLogger(__name__)
			logger.warning(
				f"Both hand pickle files missing for seq={seq_id}, frame={frame_idx}. "
				f"This frame should have been filtered by collect_sequences."
			)
			# Return default annotations with zero-initialized tensors
			hand_side_mask = torch.zeros(2, dtype=torch.float32)
			anno["hand_side"] = 0
			anno["hand_side_mask"] = hand_side_mask
			anno["object"] = Path(seq_id).parts[2] if len(Path(seq_id).parts) > 2 else "unknown"
			anno["intrinsics"] = torch.zeros(1, 4)
			anno["intrinsics_resize"] = torch.zeros(1, 4)
			anno["extrinsics"] = torch.eye(3, 4).unsqueeze(0)
			anno["betas"] = torch.zeros(2, 10)
			anno["pose"] = torch.zeros(2, 16, 6)
			anno["transl"] = torch.zeros(2, 3)
			anno["joints3d"] = torch.zeros(2, 21, 3)
			anno["bbox"] = torch.zeros(2, 4)
			anno["global_orient_world"] = torch.zeros(2, 1, 6)
			anno["transl_world"] = torch.zeros(2, 3)
			return anno

		hand_side_mask = torch.zeros(2, dtype=torch.float32)
		if left_valid and right_valid:
			hand_side = 2
			hand_side_mask[:] = 1.0
		elif left_valid:
			hand_side = 0
			hand_side_mask[0] = 1.0
		else:
			hand_side = 1
			hand_side_mask[1] = 1.0

		anno["hand_side"] = hand_side
		anno["hand_side_mask"] = hand_side_mask

		# Use object class token from sequence path, e.g. C20.
		seq_parts = Path(seq_id).parts
		anno["object"] = seq_parts[2] if len(seq_parts) > 2 else "unknown"

		cached_camera = self.get_cached_camera_record(frame)
		if cached_camera is not None:
			intrinsics = cached_camera["intrinsics"]
			intrinsics_resize = cached_camera["intrinsics_resize"]
			extrinsics = cached_camera["extrinsics"]
		else:
			intrinsics, intrinsics_resize = self._get_intrinsics(seq_id, image_path)
			extrinsics = self._get_extrinsics(seq_id, frame_idx)

		anno["intrinsics"] = intrinsics
		anno["intrinsics_resize"] = intrinsics_resize
		anno["extrinsics"] = extrinsics

		T_camera_to_world = self._build_extrinsics_4x4(extrinsics)
		T_world_to_camera = np.linalg.inv(T_camera_to_world)

		joints3d_hands = torch.zeros(2, 21, 3)
		betas_hands = torch.zeros(2, 10)
		pose_hands = torch.zeros(2, 16, 6)
		transl_hands = torch.zeros(2, 3)
		bbox_hands = torch.zeros(2, 4)
		global_orient_world_hands = torch.zeros(2, 1, 6)
		transl_world_hands = torch.zeros(2, 3)

		if left_valid:
			with open(left_pickle_path, "rb") as f:
				left_anno = pickle.load(f)

			# Validate pose coefficients
			if not self._validate_pose_coefficients(left_anno.get("poseCoeff"), "LEFT", seq_id, frame_idx):
				left_valid = False
		
		if left_valid:
			left_pose_coeff = np.asarray(left_anno["poseCoeff"], dtype=np.float32).reshape(48)
			left_beta = np.asarray(left_anno["beta"], dtype=np.float32).reshape(10)
			left_trans = np.asarray(left_anno["trans"], dtype=np.float32).reshape(3)
			left_kps2d = np.asarray(left_anno["kps2D"], dtype=np.float32).reshape(21, 2)

			left_pose_6d = self._decode_pose_coeff(left_pose_coeff)
			left_global_orient_world = left_pose_6d[:1].reshape(1, 1, 6)
			left_hand_pose = left_pose_6d[1:].reshape(1, 15, 6)
			left_transl_world = torch.from_numpy(left_trans).reshape(1, 3)
			left_betas = torch.from_numpy(left_beta)

			global_orient_world_hands[0] = left_global_orient_world
			transl_world_hands[0] = left_transl_world

			left_wrist_offset = self.mano_lhand(
				pose_type="rot6d",
				betas=left_betas[None, :],
				global_orient=left_global_orient_world,
				hand_pose=left_hand_pose,
			)["joints"][..., 0, :]

			left_global_orient_camera, left_transl_camera = transfer_mano_params_A_to_B(
				T_world_to_camera,
				left_global_orient_world.numpy(),
				left_transl_world.numpy(),
				left_wrist_offset.numpy(),
			)
			left_global_orient_camera = torch.from_numpy(left_global_orient_camera).reshape(1, 1, 6)
			left_transl_camera = torch.from_numpy(left_transl_camera).reshape(1, 3)

			left_mano_output = self.mano_lhand(
				pose_type="rot6d",
				betas=left_betas[None, :],
				global_orient=left_global_orient_camera,
				hand_pose=left_hand_pose,
				transl=left_transl_camera,
			)

			joints3d_hands[0] = left_mano_output["joints"][0]
			betas_hands[0] = left_betas
			pose_hands[0, :1] = left_global_orient_camera
			pose_hands[0, 1:] = left_pose_6d[1:]
			transl_hands[0] = left_transl_camera[0]
			bbox_hands[0] = self._build_bbox_from_kps2d(left_kps2d)

		if right_valid:
			with open(right_pickle_path, "rb") as f:
				right_anno = pickle.load(f)

			# Validate pose coefficients
			if not self._validate_pose_coefficients(right_anno.get("poseCoeff"), "RIGHT", seq_id, frame_idx):
				right_valid = False

		if right_valid:
			right_pose_coeff = np.asarray(right_anno["poseCoeff"], dtype=np.float32).reshape(48)
			right_beta = np.asarray(right_anno["beta"], dtype=np.float32).reshape(10)
			right_trans = np.asarray(right_anno["trans"], dtype=np.float32).reshape(3)
			right_kps2d = np.asarray(right_anno["kps2D"], dtype=np.float32).reshape(21, 2)

			right_pose_6d = self._decode_pose_coeff(right_pose_coeff)
			right_global_orient_world = right_pose_6d[:1].reshape(1, 1, 6)
			right_hand_pose = right_pose_6d[1:].reshape(1, 15, 6)
			right_transl_world = torch.from_numpy(right_trans).reshape(1, 3)
			right_betas = torch.from_numpy(right_beta)

			global_orient_world_hands[1] = right_global_orient_world
			transl_world_hands[1] = right_transl_world

			right_wrist_offset = self.mano_rhand(
				pose_type="rot6d",
				betas=right_betas[None, :],
				global_orient=right_global_orient_world,
				hand_pose=right_hand_pose,
			)["joints"][..., 0, :]

			right_global_orient_camera, right_transl_camera = transfer_mano_params_A_to_B(
				T_world_to_camera,
				right_global_orient_world.numpy(),
				right_transl_world.numpy(),
				right_wrist_offset.numpy(),
			)
			right_global_orient_camera = torch.from_numpy(right_global_orient_camera).reshape(1, 1, 6)
			right_transl_camera = torch.from_numpy(right_transl_camera).reshape(1, 3)

			right_mano_output = self.mano_rhand(
				pose_type="rot6d",
				betas=right_betas[None, :],
				global_orient=right_global_orient_camera,
				hand_pose=right_hand_pose,
				transl=right_transl_camera,
			)

			joints3d_hands[1] = right_mano_output["joints"][0]
			betas_hands[1] = right_betas
			pose_hands[1, :1] = right_global_orient_camera
			pose_hands[1, 1:] = right_pose_6d[1:]
			transl_hands[1] = right_transl_camera[0]
			bbox_hands[1] = self._build_bbox_from_kps2d(right_kps2d)

		# Warn if both hands failed validation (despite being in collect_sequences)
		if not left_valid and not right_valid:
			import logging
			logger = logging.getLogger(__name__)
			logger.warning(
				f"Both hands failed validation for seq={seq_id}, frame={frame_idx}. "
				f"Possible corruption in pickle files. Using zero-initialized tensors."
			)

		anno["betas"] = betas_hands
		anno["pose"] = pose_hands
		anno["transl"] = transl_hands
		anno["joints3d"] = joints3d_hands
		anno["bbox"] = bbox_hands
		anno["global_orient_world"] = global_orient_world_hands
		anno["transl_world"] = transl_world_hands

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

		assert os.path.exists(image_path), f"The image_path must be existed, got {image_path}."

		image = cv2.imread(image_path)
		assert image is not None, f"Failed to read {image_path}."

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

	hoi4d_dataset = HOI4DEgoRGBDataset(
		dataset_folder="/data/turuirui/HOI4D_dataset",
		extra_anno_keys=["extrinsics"],
		image_size=(1280, 720),
		split="train",
		sequence_length=8,
	)

	sample = hoi4d_dataset.__getitem__(65)

	hand_side = int(sample["hand_side"][0])
	if hand_side == 2:
		hand_side = 0

	if hand_side == 0:
		mano_layer = hoi4d_dataset.mano_lhand
	elif hand_side == 1:
		mano_layer = hoi4d_dataset.mano_rhand
	else:
		raise ValueError(f"Invalid hand_side {hand_side}.")

	betas = sample["betas"][0, hand_side][None]
	global_orient = sample["pose"][0, hand_side, :1][None]
	hand_pose = sample["pose"][0, hand_side, 1:][None]
	transl = sample["transl"][0, hand_side][None]

	mano_output = mano_layer(
		pose_type="rot6d",
		betas=betas,
		global_orient=global_orient,
		hand_pose=hand_pose,
		transl=transl,
	)

	joints3d = sample["joints3d"][0, hand_side]
	print(f"[INFO] MANO joints mpjpe on canonical frame0: {(joints3d - mano_output['joints']).abs().mean().item():.6f}")
