import argparse
import json
import os
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

from dataset.hand_rgb__1_0 import collate_fn_rgb
from dataset.arctic__1_0 import ARCTICEgoRGBDataset as ARCTICEgoRGBDataset__1_0
from dataset.dexycb__1_0 import DexYCBRGBDataset as DexYCBRGBDataset__1_0
from dataset.h2o__1_0 import H2OEgoRGBDataset as H2OEgoRGBDataset__1_0
from model.egohands__1_0 import EgoHands as EgoHands__1_0


# python /data/turuirui/egohands/experiment/demo/infer_arctic.py \
#   --checkpoint_path /data/turuirui/egohands/experiment/ckpt/v__1_0__arctic_1_0/best_mpjpe.pth \
#   --output_dir /data/turuirui/arctic_pipeline/outputs/egohands \
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
        "--dataset_name",
        type=str,
        default="arctic",
        choices=["dexycb", "arctic", "h2o"],
        help="Dataset to run inference on.",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        default="1_0",
        choices=["1_0"],
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
        return EgoHands__1_0()
    raise ValueError(f"Unsupported model_version: {args.model_version}")


def build_dataset(args, split):
    if args.dataset_name == "dexycb":
        dataset_args = dict(
            dataset_folder="/data/public/dataset/hand/dexycb",
            image_size=(512, 512),
            split=split,
            setup="s0",
            seed=args.seed,
        )
        if args.dataset_version == "1_0":
            return DexYCBRGBDataset__1_0(**dataset_args)
    elif args.dataset_name == "arctic":
        dataset_args = dict(
            dataset_folder="/data/public/dataset/hand/arctic",
            image_size=(512, 512),
            split=split,
            setup="p1",
            seed=args.seed,
        )
        if args.dataset_version == "1_0":
            return ARCTICEgoRGBDataset__1_0(**dataset_args)
    elif args.dataset_name == "h2o":
        dataset_args = dict(
            dataset_folder="/data/public/dataset/hand/h2o",
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

    model = build_model(args)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)

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

            pose_pred = outputs["pose"].detach().cpu().numpy()
            transl_pred = outputs["transl"].detach().cpu().numpy()
            hand_side_mask = batch["hand_side_mask"].detach().cpu().numpy()

            metas = batch["meta"]
            for i in range(batch_size):
                meta = metas[i]
                image_path = meta["frame_id"]
                save_name = build_output_name(image_path)
                save_path = output_dir / save_name

                if save_path.exists() and not args.overwrite:
                    skipped_files += 1
                    continue

                kp3d_pred = build_eval_kp3d_pred(
                    model=model,
                    pose_pred=outputs["pose"][i],
                    transl_pred=outputs["transl"][i],
                    hand_side_mask=torch.as_tensor(hand_side_mask[i]),
                    betas=batch_device["betas"][i],
                )

                extra = {
                    "pose_pred": pose_pred[i].astype(np.float32),
                    "transl_pred": transl_pred[i].astype(np.float32),
                    "hand_side_mask": hand_side_mask[i].astype(np.float32),
                    "coord_semantics": "camera",
                    "joint_order": "openpose",
                    "seq_id": str(meta.get("seq_id", "")),
                    "frame_id": str(meta.get("frame_id", "")),
                }

                save_prediction(save_path, image_path, kp3d_pred, extra=extra)
                saved_files += 1

            total_samples += batch_size
            pbar.set_postfix(saved=saved_files, skipped=skipped_files)

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_version": args.dataset_version,
        "model_version": args.model_version,
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
