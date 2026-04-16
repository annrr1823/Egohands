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
from dataset.arctic__1_0 import ARCTICEgoRGBDataset as ARCTICEgoRGBDataset__1_0
from dataset.arctic__2_0 import ARCTICEgoRGBDataset as ARCTICEgoRGBDataset__2_0
from dataset.dexycb__1_0 import DexYCBRGBDataset as DexYCBRGBDataset__1_0
from dataset.h2o__1_0 import H2OEgoRGBDataset as H2OEgoRGBDataset__1_0
from model.egohands__1_0 import EgoHands as EgoHands__1_0


# python /data/sunzhihao2/turuirui/egohands/experiment/demo/infer_arctic.py \
#   --checkpoint_path /data/sunzhihao2/turuirui/egohands/experiment/ckpt/v__4_0__arctic_1_0/best_mpjpe.pth \
#   --output_dir /data/sunzhihao2/turuirui/arctic_pipeline/outputs/egohands_3 \
#   --dataset_name arctic \
#   --split val \
#   --batch_size 16 \
#   --device cuda \
#   --save_summary

def parse_args():
    parser = argparse.ArgumentParser(description="Run EgoHands on ARCTIC and export evalable predictions.")

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
        default="arctic",
        choices=["dexycb", "arctic", "h2o"],
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

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=618, help="Random seed for dataset internals.")
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=8,
        help="Temporal window length for sequence datasets (e.g., arctic 2_0).",
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

    return parser.parse_args()


def build_model(args):
    if args.model_version == "1_0":
        return EgoHands__1_0(use_plucker_ray=args.use_plucker_ray)
    raise ValueError(f"Unsupported model_version: {args.model_version}")


def build_dataset(args, split):
    dataset_folder_map = {
        "dexycb": "/data/public/dataset/hand/dexycb",
        "arctic": "/data/sunzhihao2/turuirui/dataset/arctic/data/data",
        "h2o": "/data/public/dataset/hand/h2o",
    }

    dataset_folder = args.dataset_folder or dataset_folder_map[args.dataset_name]

    if args.dataset_name == "dexycb":
        dataset_args = dict(
            dataset_folder=dataset_folder,
            image_size=(512, 512),
            split=split,
            setup="s0",
            seed=args.seed,
        )
        if args.dataset_version == "1_0":
            return DexYCBRGBDataset__1_0(**dataset_args)
    elif args.dataset_name == "arctic":
        dataset_args = dict(
            dataset_folder=dataset_folder,
            image_size=(512, 512),
            split=split,
            setup="p1",
            seed=args.seed,
        )
        if args.dataset_version == "1_0":
            return ARCTICEgoRGBDataset__1_0(**dataset_args)
        if args.dataset_version == "2_0":
            return ARCTICEgoRGBDataset__2_0(sequence_length=args.sequence_length, **dataset_args)
    elif args.dataset_name == "h2o":
        dataset_args = dict(
            dataset_folder=dataset_folder,
            image_size=(512, 512),
            split=split,
            setup="default",
            seed=args.seed,
        )
        if args.dataset_version == "1_0":
            return H2OEgoRGBDataset__1_0(**dataset_args)

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


def build_autocast_and_dtype(device, mixed_precision):
    device_type = torch.device(device).type
    if device_type != "cuda" or mixed_precision == "no":
        return nullcontext, torch.float32
    if mixed_precision == "fp16":
        return (lambda: torch.autocast(device_type="cuda", dtype=torch.float16)), torch.float16
    if mixed_precision == "bf16":
        return (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)), torch.bfloat16
    raise ValueError(f"Unsupported mixed_precision: {mixed_precision}")


def build_output_name(image_path: str) -> str:
    path = Path(image_path)
    parts = path.parts
    if len(parts) >= 4:
        setup = parts[-4]
        seq_name = parts[-3]
        frame_id = path.stem
        return f"{setup}_{seq_name}_{frame_id}.npz"
    return f"{path.stem}.npz"


def save_prediction(save_path: Path, image_path: str, kp3d_pred: np.ndarray, extra: Optional[dict] = None) -> None:
    payload = {
        "image_path": str(image_path),
        "kp3d_pred": kp3d_pred.astype(np.float32),
    }
    if extra:
        payload.update(extra)
    np.savez_compressed(save_path, **payload)


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


def get_meta_frame_path(meta: dict, time_index: int) -> str:
    frame_ids = meta.get("frame_ids")
    if isinstance(frame_ids, list):
        return str(frame_ids[time_index])
    return str(meta.get("frame_id", ""))


def build_eval_kp3d_pred(model, pose_pred: torch.Tensor, transl_pred: torch.Tensor, hand_side_mask: torch.Tensor, betas: torch.Tensor) -> np.ndarray:
    # ARCTIC eval expects full-camera MANO joints in [left(21), right(21)] order.
    # If a future model emits crop-space translation, the crop->full recovery belongs here.
    kp3d_pred = np.full((42, 3), -1.0, dtype=np.float32)

    if hand_side_mask[0] > 0:
        left_output = model.mano_lhand(
            pose_type="rot6d",
            betas=betas[0:1],
            global_orient=pose_pred[0:1, :1],
            hand_pose=pose_pred[0:1, 1:],
            transl=transl_pred[0:1],
        )
        kp3d_pred[:21] = left_output["joints"][0].detach().cpu().numpy().astype(np.float32)

    if hand_side_mask[1] > 0:
        right_output = model.mano_rhand(
            pose_type="rot6d",
            betas=betas[1:2],
            global_orient=pose_pred[1:2, :1],
            hand_pose=pose_pred[1:2, 1:],
            transl=transl_pred[1:2],
        )
        kp3d_pred[21:] = right_output["joints"][0].detach().cpu().numpy().astype(np.float32)

    return kp3d_pred


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if isinstance(state_dict, dict) and any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    ckpt_has_plucker = isinstance(state_dict, dict) and checkpoint_has_plucker_keys(state_dict)
    if ckpt_has_plucker and not args.use_plucker_ray:
        print("[infer_arctic] Detected Plucker-ray weights in checkpoint; enabling --use_plucker_ray automatically.")
        args.use_plucker_ray = True

    model = build_model(args)

    strict_loading = True
    if args.use_plucker_ray and not ckpt_has_plucker:
        strict_loading = False
        print("[infer_arctic] --use_plucker_ray is enabled but checkpoint has no Plucker weights; loading non-strictly.")

    incompatible = model.load_state_dict(state_dict, strict=strict_loading)
    if not strict_loading and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(
            "[infer_arctic] Non-strict load details: "
            f"missing_keys={len(incompatible.missing_keys)}, "
            f"unexpected_keys={len(incompatible.unexpected_keys)}"
        )

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    dataset = build_dataset(args, split=args.split)

    if args.max_samples > 0:
        dataset = torch.utils.data.Subset(dataset, list(range(min(args.max_samples, len(dataset)))))

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

    pbar = tqdm(dataloader, desc="EgoHands -> ARCTIC", dynamic_ncols=True)

    with torch.no_grad():
        for batch in pbar:
            batch_size = batch["image"].shape[0]
            batch_device = batch_to_device(batch, device)

            with autocast_ctx():
                _, _, outputs = model(batch_device, weight_dtype=weight_dtype)

            pose_pred_t, transl_pred_t = ensure_temporal_outputs(outputs)
            pose_pred = pose_pred_t.detach().cpu().numpy()
            transl_pred = transl_pred_t.detach().cpu().numpy()

            hand_side_mask_t = batch["hand_side_mask"]
            if hand_side_mask_t.dim() == 2:
                hand_side_mask_t = hand_side_mask_t.unsqueeze(1)
            hand_side_mask = hand_side_mask_t.detach().cpu().numpy()

            metas = batch["meta"]
            seq_len = pose_pred_t.shape[1]
            for i in range(batch_size):
                meta = metas[i]
                for t in range(seq_len):
                    image_path = get_meta_frame_path(meta, t)
                    save_name = build_output_name(image_path)
                    save_path = output_dir / save_name

                    if save_path.exists() and not args.overwrite:
                        skipped_files += 1
                        continue

                    kp3d_pred = build_eval_kp3d_pred(
                        model=model,
                        pose_pred=pose_pred_t[i, t],
                        transl_pred=transl_pred_t[i, t],
                        hand_side_mask=torch.as_tensor(hand_side_mask[i, t]),
                        betas=batch_device["betas"][i, t],
                    )

                    extra = {
                        "pose_pred": pose_pred[i, t].astype(np.float32),
                        "transl_pred": transl_pred[i, t].astype(np.float32),
                        "hand_side_mask": hand_side_mask[i, t].astype(np.float32),
                        "coord_semantics": "camera",
                        "joint_order": "openpose",
                        "seq_id": str(meta.get("seq_id", "")),
                        "frame_id": str(image_path),
                        "temporal_index": int(t),
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
        "checkpoint_path": args.checkpoint_path,
        "output_dir": str(output_dir),
        "num_samples_processed": total_samples,
        "num_files_saved": saved_files,
        "num_files_skipped": skipped_files,
        "note": "Run /data/turuirui/arctic_pipeline/evaluation/eval.py on this output_dir for ARCTIC metrics.",
    }

    if args.save_summary:
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
