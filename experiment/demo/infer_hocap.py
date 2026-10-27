import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import pyrootutils

root = pyrootutils.setup_root(
	search_from=__file__,
	indicator=["README.md"],
	pythonpath=True,
	dotenv=True,
)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.hand_rgb__2_0 import collate_fn_rgb
from dataset.hocap__1_0 import HOCapRGBDataset as HOCapRGBDataset__1_0
from model.egohands__1_0 import EgoHands as EgoHands__1_0

# CUDA_VISIBLE_DEVICES=1 python /data/sunzhihao2/turuirui/egohands/experiment/demo/infer_hocap.py \
#   --checkpoint_path /data/sunzhihao2/turuirui/egohands/experiment/ckpt/v__11_0__none_temporal_wo_body_backbone/best_mpjpe.pth \
#   --output_dir /data/sunzhihao2/turuirui/hocap_pipeline/outputs/egohands_wo_bodybackbone \
#   --dataset_name hocap \
#   --dataset_version 1_0 \
#   --split val \
#   --task hpe \
#   --view hololens_pv \
#   --device cuda \
#   --save_summary \

# CUDA_VISIBLE_DEVICES=1 python /data/sunzhihao2/turuirui/egohands/experiment/demo/infer_hocap.py \
#   --checkpoint_path /data/sunzhihao2/turuirui/egohands/experiment/ckpt/v__11_0__none_temporal_wo_mask_attention/best_mpjpe.pth \
#   --output_dir /data/sunzhihao2/turuirui/hocap_pipeline/outputs/egohands_wo_maskattention \
#   --dataset_name hocap \
#   --dataset_version 1_0 \
#   --split val \
#   --task hpe \
#   --view hololens_pv \
#   --device cuda \
#   --save_summary \

# CUDA_VISIBLE_DEVICES=1 python /data/sunzhihao2/turuirui/egohands/experiment/demo/infer_hocap.py \
#   --checkpoint_path /data/sunzhihao2/turuirui/egohands/experiment/ckpt/v__11_0__none_temporal_copy/best_mpjpe.pth \
#   --output_dir /data/sunzhihao2/turuirui/hocap_pipeline/outputs/egohands_none_temporal_smoketest \
#   --dataset_name hocap \
#   --dataset_version 1_0 \
#   --split val \
#   --task hpe \
#   --view hololens_pv \
#   --device cuda \
#   --save_summary \
#   --max_samples 100


def parse_args():
	parser = argparse.ArgumentParser(description="Run EgoHands on HoCap and export evalable predictions.")

	parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to trained checkpoint (.pth).")
	parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .npz prediction files.")
	parser.add_argument(
		"--dataset_folder",
		type=str,
		default=None,
		help="Override dataset root folder. If omitted, use built-in defaults by dataset_name.",
	)

	parser.add_argument(
		"--dataset_name",
		type=str,
		default="hocap",
		choices=["hocap"],
		help="Dataset to run inference on.",
	)
	parser.add_argument(
		"--dataset_version",
		type=str,
		default="2_0",
		choices=["1_0", "2_0"],
		help="Dataset version.",
	)
	parser.add_argument(
		"--model_version",
		type=str,
		default="1_0",
		choices=["1_0"],
		help="Model version to instantiate.",
	)
	parser.add_argument(
		"--use_plucker_ray",
		action="store_true",
		help="Fuse patch-level Plucker ray geometry into the body token stream.",
	)
	parser.add_argument(
		"--split",
		type=str,
		default="val",
		choices=["train", "val", "test"],
		help="Dataset split for inference.",
	)
	parser.add_argument(
		"--task",
		type=str,
		default="hpe",
		choices=["hpe", "odet", "ope"],
		help="HoCap task used to choose split config file.",
	)
	parser.add_argument(
		"--view",
		type=str,
		default="hololens_pv",
		choices=["hololens_pv", "realsense"],
		help="Input view type.",
	)
	parser.add_argument(
		"--realsense_cam_idx",
		type=int,
		default=0,
		help="RealSense camera index used when --view realsense.",
	)

	parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
	parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
	parser.add_argument("--seed", type=int, default=618, help="Random seed for dataset internals.")
	parser.add_argument(
		"--sequence_length",
		type=int,
		default=1,
		help="Temporal window length for sequence datasets (set 1 for single-frame).",
	)
	parser.add_argument(
		"--sequence_frame_stride",
		type=int,
		default=1,
		help="Frame stride inside a temporal window for sequence datasets.",
	)

	parser.add_argument(
		"--device",
		type=str,
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device string, e.g., cuda, cuda:0, cpu.",
	)
	parser.add_argument(
		"--mixed_precision",
		type=str,
		default="no",
		choices=["no", "fp16", "bf16"],
		help="Autocast precision for CUDA inference.",
	)

	parser.add_argument(
		"--max_samples",
		type=int,
		default=-1,
		help="If > 0, only run this many samples for quick checks.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing prediction files if they already exist.",
	)
	parser.add_argument(
		"--save_summary",
		action="store_true",
		help="Save a small summary.json alongside the predictions.",
	)
	parser.add_argument(
		"--canonical_to_current_camera",
		type=str,
		default="on",
		choices=["on", "off"],
		help=(
			"For hocap 2_0 temporal clips, convert predicted 3D joints from "
			"canonical(first-frame camera) coordinates to each frame's camera coordinates."
		),
	)
	parser.add_argument(
		"--model_camera_source",
		type=str,
		default="dataset",
		choices=["dataset", "cache", "gt"],
		help=(
			"Which camera parameters are fed to EgoHands. "
			"dataset: use dataset default tensors; cache: force *_cache; gt: force *_gt."
		),
	)

	return parser.parse_args()


def build_model(args):
	if args.model_version == "1_0":
		return EgoHands__1_0(use_plucker_ray=args.use_plucker_ray)
	raise ValueError(f"Unsupported model_version: {args.model_version}")


def build_dataset(args, split):
	dataset_folder_map = {
		"hocap": "/data/sunzhihao2/turuirui/Hocap_dataset",
	}

	dataset_folder = args.dataset_folder or dataset_folder_map[args.dataset_name]
	dataset_args = dict(
		dataset_folder=dataset_folder,
		image_size=(512, 512),
		split=split,
		setup="default",
		task=args.task,
		view=args.view,
		realsense_cam_idx=int(args.realsense_cam_idx),
		seed=args.seed,
	)
	if args.dataset_version == "1_0":
		return HOCapRGBDataset__1_0(**dataset_args)
	
	raise ValueError(
		f"Unsupported dataset config: dataset_name={args.dataset_name}, dataset_version={args.dataset_version}"
	)


def batch_to_device(batch, device):
	out = {}
	for key, value in batch.items():
		if isinstance(value, torch.Tensor):
			out[key] = value.to(device)
		elif isinstance(value, list):
			out[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
		else:
			out[key] = value
	return out


def select_model_camera_params(batch_device: dict, model_camera_source: str, dataset_version: str) -> None:
	if model_camera_source == "dataset":
		return

	if model_camera_source == "cache":
		need = ("intrinsics_cache", "intrinsics_resize_cache", "extrinsics_cache")
		if any(key not in batch_device for key in need):
			if dataset_version == "1_0":
				print(
					"[infer_hocap] Warning: --model_camera_source cache is not supported for dataset_version 1_0. "
					"Falling back to dataset defaults."
				)
				return
			raise KeyError(
				"Camera cache tensors are missing in batch. "
				"Please use dataset 2_0 loaders with camera cache enabled."
			)
		batch_device["intrinsics"] = batch_device["intrinsics_cache"]
		batch_device["intrinsics_resize"] = batch_device["intrinsics_resize_cache"]
		batch_device["extrinsics"] = batch_device["extrinsics_cache"]
		return

	if model_camera_source == "gt":
		need = ("intrinsics_gt", "intrinsics_resize_gt", "extrinsics_gt")
		if any(key not in batch_device for key in need):
			if dataset_version == "1_0":
				print(
					"[infer_hocap] Warning: --model_camera_source gt is not supported for dataset_version 1_0. "
					"Falling back to dataset defaults."
				)
				return
			raise KeyError(
				"GT camera tensors are missing in batch. "
				"Please use dataset 2_0 loaders that provide *_gt camera fields."
			)
		batch_device["intrinsics"] = batch_device["intrinsics_gt"]
		batch_device["intrinsics_resize"] = batch_device["intrinsics_resize_gt"]
		batch_device["extrinsics"] = batch_device["extrinsics_gt"]
		return

	raise ValueError(f"Unsupported model_camera_source: {model_camera_source}")


def build_autocast_and_dtype(device, mixed_precision):
	device_type = torch.device(device).type
	if device_type != "cuda" or mixed_precision == "no":
		return nullcontext, torch.float32
	if mixed_precision == "fp16":
		return (lambda: torch.autocast(device_type="cuda", dtype=torch.float16)), torch.float16
	if mixed_precision == "bf16":
		return (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)), torch.bfloat16
	raise ValueError(f"Unsupported mixed_precision: {mixed_precision}")


def load_checkpoint(checkpoint_path: str):
	path = str(checkpoint_path)
	if path.endswith(".safetensors"):
		try:
			from safetensors.torch import load_file as _load_safetensors
		except Exception as e:
			raise RuntimeError(
				"safetensors package is required to load .safetensors files. "
				"Install with `pip install safetensors`"
			) from e
		return _load_safetensors(path, device="cpu")

	try:
		return torch.load(path, map_location="cpu", weights_only=False)
	except TypeError:
		return torch.load(path, map_location="cpu")


def checkpoint_has_plucker_keys(state_dict: dict) -> bool:
	return any(
		key.startswith("plucker_proj.") or key.startswith("plucker_ray.")
		for key in state_dict.keys()
	)


def ensure_temporal_outputs(outputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
	pose_pred = outputs["pose"]
	transl_pred = outputs["transl"]

	if pose_pred.dim() == 4:
		pose_pred = pose_pred.unsqueeze(1)
	if transl_pred.dim() == 3:
		transl_pred = transl_pred.unsqueeze(1)

	return pose_pred, transl_pred


def ensure_temporal_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
	if extrinsics.dim() == 3:
		extrinsics = extrinsics.unsqueeze(1)
	elif extrinsics.dim() == 4 and extrinsics.shape[1] == 3 and extrinsics.shape[2] == 4:
		extrinsics = extrinsics.unsqueeze(1)
	if extrinsics.dim() == 5 and extrinsics.shape[2] == 1:
		extrinsics = extrinsics.squeeze(2)

	if extrinsics.dim() != 4 or tuple(extrinsics.shape[-2:]) != (3, 4):
		raise ValueError(f"Unexpected extrinsics shape: {tuple(extrinsics.shape)}")

	batch_size, seq_len = extrinsics.shape[:2]
	extrinsics_4x4 = torch.eye(
		4,
		dtype=extrinsics.dtype,
		device=extrinsics.device,
	).view(1, 1, 4, 4).expand(batch_size, seq_len, 4, 4).clone()
	extrinsics_4x4[..., :3, :] = extrinsics
	return extrinsics_4x4


def ensure_temporal_hand_side_mask(hand_side_mask: torch.Tensor) -> torch.Tensor:
	if hand_side_mask.dim() == 2:
		hand_side_mask = hand_side_mask.unsqueeze(1)
	if hand_side_mask.dim() != 3 or hand_side_mask.shape[-1] != 2:
		raise ValueError(f"Unexpected hand_side_mask shape: {tuple(hand_side_mask.shape)}")
	return hand_side_mask


def ensure_temporal_betas(betas: torch.Tensor) -> torch.Tensor:
	if betas.dim() == 3:
		betas = betas.unsqueeze(1)
	if betas.dim() != 4 or tuple(betas.shape[-2:]) != (2, 10):
		raise ValueError(f"Unexpected betas shape: {tuple(betas.shape)}")
	return betas


def transform_kp3d_with_4x4(kp3d_pred: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
	out = kp3d_pred.copy()
	valid_mask = ~np.all(out == -1.0, axis=1)
	if not np.any(valid_mask):
		return out

	points = out[valid_mask]
	points_homo = np.concatenate(
		[points, np.ones((points.shape[0], 1), dtype=points.dtype)],
		axis=1,
	)
	points_transformed = (transform_4x4 @ points_homo.T).T[:, :3]
	out[valid_mask] = points_transformed.astype(np.float32)
	return out


def build_output_name(frame_id) -> str:
	if isinstance(frame_id, (list, tuple)) and len(frame_id) >= 4:
		subject_idx = int(frame_id[0])
		seq_idx = int(frame_id[1])
		frame_idx = int(frame_id[3])
		return f"sub{subject_idx:02d}_seq{seq_idx:03d}_f{frame_idx:06d}.npz"
	return f"frame_{str(frame_id)}.npz"


def get_meta_frame_id(meta: dict, time_index: int):
	frame_ids = meta.get("frame_ids")
	if isinstance(frame_ids, list):
		return frame_ids[time_index]
	return meta.get("frame_id", "")


def save_prediction(save_path: Path, image_path: str, kp3d_pred: np.ndarray, extra: Optional[dict] = None) -> None:
	payload = {
		"image_path": str(image_path),
		"kp3d_pred": kp3d_pred.astype(np.float32),
	}
	if extra:
		payload.update(extra)
	np.savez_compressed(save_path, **payload)


def build_eval_kp3d_pred(
	model,
	pose_pred: torch.Tensor,
	transl_pred: torch.Tensor,
	hand_side_mask: torch.Tensor,
	betas: torch.Tensor,
) -> np.ndarray:
	kp3d_pred = np.full((42, 3), -1.0, dtype=np.float32)

	if hand_side_mask[0] > 0:
		left_global_orient = pose_pred[0:1, :1]
		left_hand_pose = pose_pred[0:1, 1:]
		left_transl = transl_pred[0:1]

		left_output = model.mano_lhand(
			pose_type="rot6d",
			betas=betas[0:1],
			global_orient=left_global_orient,
			hand_pose=left_hand_pose,
			transl=left_transl,
		)
		kp3d_pred[:21] = left_output["joints"][0].detach().cpu().numpy().astype(np.float32)

	if hand_side_mask[1] > 0:
		right_global_orient = pose_pred[1:2, :1]
		right_hand_pose = pose_pred[1:2, 1:]
		right_transl = transl_pred[1:2]

		right_output = model.mano_rhand(
			pose_type="rot6d",
			betas=betas[1:2],
			global_orient=right_global_orient,
			hand_pose=right_hand_pose,
			transl=right_transl,
		)
		kp3d_pred[21:] = right_output["joints"][0].detach().cpu().numpy().astype(np.float32)

	return kp3d_pred


def main():
	args = parse_args()
	if args.view == "realsense" and args.realsense_cam_idx < 0:
		raise ValueError("--realsense_cam_idx must be >= 0 when --view realsense.")

	apply_canonical_to_current = args.canonical_to_current_camera == "on"
	apply_canonical_to_current_for_v2 = apply_canonical_to_current and args.dataset_version == "2_0" and args.sequence_length > 1
	if apply_canonical_to_current and args.dataset_version == "2_0" and args.sequence_length <= 1:
		print(
			"[infer_hocap] Warning: --canonical_to_current_camera only applies to temporal dataset_version 2_0 clips. "
			"Single-frame samples are already in camera coordinates, so the extra transform is skipped."
		)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	checkpoint = load_checkpoint(args.checkpoint_path)
	state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
	if isinstance(state_dict, dict) and any(key.startswith("module.") for key in state_dict.keys()):
		state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

	ckpt_has_plucker = isinstance(state_dict, dict) and checkpoint_has_plucker_keys(state_dict)
	if ckpt_has_plucker and not args.use_plucker_ray:
		print("[infer_hocap] Detected Plucker-ray weights in checkpoint; enabling --use_plucker_ray automatically.")
		args.use_plucker_ray = True

	model = build_model(args)

	strict_loading = True
	if args.use_plucker_ray and not ckpt_has_plucker:
		strict_loading = False
		print("[infer_hocap] --use_plucker_ray is enabled but checkpoint has no Plucker weights; loading non-strictly.")

	incompatible = model.load_state_dict(state_dict, strict=strict_loading)
	if not strict_loading and (incompatible.missing_keys or incompatible.unexpected_keys):
		print(
			"[infer_hocap] Non-strict load details: "
			f"missing_keys={len(incompatible.missing_keys)}, "
			f"unexpected_keys={len(incompatible.unexpected_keys)}"
		)

	device = torch.device(args.device)
	model = model.to(device)
	model.eval()

	dataset_raw = build_dataset(args, split=args.split)
	if hasattr(dataset_raw, "sequence_frame_stride"):
		dataset_raw.sequence_frame_stride = int(args.sequence_frame_stride)
		print(f"[infer_hocap] Using sequence_frame_stride={dataset_raw.sequence_frame_stride}.")

	dataset = dataset_raw
	if args.max_samples > 0:
		dataset = torch.utils.data.Subset(dataset_raw, list(range(min(args.max_samples, len(dataset_raw)))))

	dataloader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=collate_fn_rgb,
	)

	autocast_ctx, weight_dtype = build_autocast_and_dtype(args.device, args.mixed_precision)

	total_samples = 0
	saved_files = 0
	skipped_files = 0

	pbar = tqdm(dataloader, desc="EgoHands -> HoCap", dynamic_ncols=True)

	with torch.no_grad():
		for batch in pbar:
			batch_size = batch["image"].shape[0]
			batch_device = batch_to_device(batch, device)
			select_model_camera_params(batch_device, args.model_camera_source, args.dataset_version)

			extrinsics_seq = ensure_temporal_extrinsics(batch["extrinsics"]).detach().cpu().numpy()

			with autocast_ctx():
				_, _, outputs = model(batch_device, weight_dtype=weight_dtype)

			pose_pred_t, transl_pred_t = ensure_temporal_outputs(outputs)
			pose_pred = pose_pred_t.detach().cpu().numpy()
			transl_pred = transl_pred_t.detach().cpu().numpy()

			hand_side_mask_t = ensure_temporal_hand_side_mask(batch["hand_side_mask"])
			hand_side_mask = hand_side_mask_t.detach().cpu().numpy()

			betas_t = ensure_temporal_betas(batch_device["betas"])

			metas = batch["meta"]
			seq_len = pose_pred_t.shape[1]
			for i in range(batch_size):
				meta = metas[i]
				for t in range(seq_len):
					frame_id = get_meta_frame_id(meta, t)
					image_path = dataset_raw.convert_frame_id_to_image_path(frame_id)
					save_name = build_output_name(frame_id)
					save_path = output_dir / save_name

					if save_path.exists() and not args.overwrite:
						skipped_files += 1
						continue

					kp3d_pred = build_eval_kp3d_pred(
						model=model,
						pose_pred=pose_pred_t[i, t],
						transl_pred=transl_pred_t[i, t],
						hand_side_mask=torch.as_tensor(hand_side_mask[i, t]),
						betas=betas_t[i, t],
					)

					should_transform = apply_canonical_to_current_for_v2
					if should_transform:
						t_curr_to_canonical = extrinsics_seq[i, t].astype(np.float32)
						t_canonical_to_curr = np.linalg.inv(t_curr_to_canonical).astype(np.float32)
						kp3d_pred = transform_kp3d_with_4x4(kp3d_pred, t_canonical_to_curr)

					extra = {
						"pose_pred": pose_pred[i, t].astype(np.float32),
						"transl_pred": transl_pred[i, t].astype(np.float32),
						"hand_side_mask": hand_side_mask[i, t].astype(np.float32),
						"coord_semantics": (
							"camera_current_frame"
							if should_transform or args.dataset_version == "1_0"
							else "camera_canonical_first_frame"
						),
						"canonical_to_current_camera_applied": bool(should_transform),
						"joint_order": "openpose",
						"frame_id": str(frame_id),
						"seq_id": str(meta.get("seq_id", "")),
						"temporal_index": int(t),
						"view": args.view,
						"task": args.task,
					}

					save_prediction(save_path, image_path, kp3d_pred, extra=extra)
					saved_files += 1

			total_samples += batch_size
			pbar.set_postfix(saved=saved_files, skipped=skipped_files)

	summary = {
		"dataset_name": args.dataset_name,
		"dataset_version": args.dataset_version,
		"sequence_length": args.sequence_length,
		"model_version": args.model_version,
		"use_plucker_ray": args.use_plucker_ray,
		"dataset_folder": str(args.dataset_folder) if args.dataset_folder else "<default>",
		"split": args.split,
		"task": args.task,
		"view": args.view,
		"realsense_cam_idx": int(args.realsense_cam_idx),
		"checkpoint_path": args.checkpoint_path,
		"output_dir": str(output_dir),
		"num_samples_processed": total_samples,
		"num_files_saved": saved_files,
		"num_files_skipped": skipped_files,
		"canonical_to_current_camera": args.canonical_to_current_camera,
		"canonical_to_current_camera_applied_for_v2": bool(apply_canonical_to_current_for_v2),
		"model_camera_source": args.model_camera_source,
		"note": "Run /data/turuirui/hocap_pipeline/evaluation/eval.py on this output_dir for HoCap metrics.",
	}

	if args.save_summary:
		summary_path = output_dir / "summary.json"
		with open(summary_path, "w", encoding="utf-8") as f:
			json.dump(summary, f, indent=2, ensure_ascii=False)

	print("=" * 80)
	print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
