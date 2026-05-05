import argparse
import math
import os

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md"],
    pythonpath=True,
    dotenv=True,
)

import cv2
from pathlib import Path
import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset.hand_rgb__1_0 import collate_fn_rgb
from dataset.arctic__1_0 import ARCTICEgoRGBDataset as ARCTICEgoRGBDataset__1_0
from dataset.h2o__1_0 import H2OEgoRGBDataset as H2OEgoRGBDataset__1_0
from dataset.egodex__2_0 import EgoDexRGBDataset as EgoDexRGBDataset__2_0
from model.egohands__1_0 import EgoHands as EgoHands__1_0


def batch_to_device(batch, device):
    batch_out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_out[key] = value.to(device)
        elif isinstance(value, list):
            batch_out[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
        else:
            batch_out[key] = value
    return batch_out

def transform_points(points, T):
    ones = torch.ones(*points.shape[:-1], 1, device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=-1)
    points_out = points_h @ T.T
    return points_out[..., :3]


def export_trimesh(renderer, vertices, is_rhand, save_folder, file_name, mesh_base_color=None):
    os.makedirs(save_folder, exist_ok=True)
    if mesh_base_color is None:
        mesh_base_color = (1.0, 1.0, 0.9)

    if torch.is_tensor(vertices):
        vertices_np = vertices.detach().cpu().numpy()
    else:
        vertices_np = np.asarray(vertices)

    if vertices_np.ndim == 3 and vertices_np.shape[0] == 1:
        vertices_np = vertices_np[0]
    elif vertices_np.ndim != 2:
        raise ValueError(f"Expected vertices with shape [V, 3] or [1, V, 3], got {vertices_np.shape}")

    mesh = renderer.vertices_to_trimesh(
        vertices=vertices_np,
        is_rhand=is_rhand,
        mesh_base_color=mesh_base_color,
    )
    export_path = os.path.join(save_folder, file_name)
    mesh.export(export_path)
    return export_path


def joints_to_numpy(joints):
    if torch.is_tensor(joints):
        return joints.detach().cpu().numpy()
    return np.asarray(joints)


def set_axes_equal(ax, points):
    points = np.asarray(points)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def save_3d_joints_figure(gt_joints, pred_joints, save_path, title):
    gt = joints_to_numpy(gt_joints)
    pred = joints_to_numpy(pred_joints)
    all_points = np.concatenate([gt, pred], axis=0)

    fig = plt.figure(figsize=(15, 5))
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]

    views = [
        (gt, "GT 3D keypoints", "tab:blue", "GT"),
        (pred, "Pred 3D keypoints", "tab:orange", "Pred"),
        (None, "Overlay", None, None),
    ]

    for ax, (points, subplot_title, color, label) in zip(axes, views):
        if points is not None:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=28, label=label)
            for idx, (x, y, z) in enumerate(points):
                ax.text(x, y, z, str(idx), fontsize=6, color=color)
        else:
            ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c="tab:blue", s=24, label="GT")
            ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c="tab:orange", s=24, label="Pred")
            for idx, (x, y, z) in enumerate(gt):
                ax.text(x, y, z, str(idx), fontsize=6, color="tab:blue")
            for idx, (x, y, z) in enumerate(pred):
                ax.text(x, y, z, str(idx), fontsize=6, color="tab:orange")
            ax.legend(loc="upper left", fontsize=8)

        ax.set_title(subplot_title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_equal(ax, all_points)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def build_model(args):
    if args.model_version == "1_0":
        return EgoHands__1_0()
    else:
        raise ValueError(f"Unsupported model_version: {args.model_version}")


def build_dataset(args, split):
    if args.val_dataset_name == "arctic":
        dataset_args = dict(
            dataset_folder="/data/sunzhihao2/turuirui/dataset/arctic/data/data",
            image_size=(512, 512),
            split=split,
            setup="p1",
            seed=args.seed,
        )
        if args.dataset_version == "1_0":
            return ARCTICEgoRGBDataset__1_0(**dataset_args)
    elif args.val_dataset_name == "h2o":
        dataset_args = dict(
            dataset_folder="/data/public/hand/h2o",
            image_size=(512, 512),
            split=split,
            setup="default",
            seed=args.seed,
        )
        if args.dataset_version == "1_0":
            return H2OEgoRGBDataset__1_0(**dataset_args)
    elif args.val_dataset_name == "egodex":
        dataset_args = dict(
            dataset_folder="/data/public/ego/egodex",
            image_size=(512, 512),
            split=split,
            setup="default",
            seed=args.seed,
        )
        if args.dataset_version == "2_0":
            return EgoDexRGBDataset__2_0(**dataset_args)

    raise ValueError(
        f"Unsupported dataset config: dataset_name={args.val_dataset_name}, dataset_version={args.dataset_version}"
    )


def resolve_hand_side(sample, preferred_hand_side=None):
    hand_side_mask = sample["hand_side_mask"]

    if preferred_hand_side in (0, 1) and float(hand_side_mask[preferred_hand_side].item()) > 0.0:
        return preferred_hand_side

    for hand_index in range(2):
        if float(hand_side_mask[hand_index].item()) > 0.0:
            return hand_index

    return 0


def run_demo_for_dataset(args, model, device):
    dataset = build_dataset(args, split="val")
    sample = dataset[args.sample_index]

    sample0 = sample
    if isinstance(sample.get("meta"), list):
        sample0 = {key: (value[0] if isinstance(value, torch.Tensor) and value.ndim > 0 else value) for key, value in sample.items()}
        sample0["meta"] = sample["meta"][0]

    frame_id = sample0["meta"]["frame_id"]
    print(f"[INFO] Visualizing {args.val_dataset_name} sample with frame_id {frame_id}.")

    batch = collate_fn_rgb([sample])
    batch = batch_to_device(batch, device)

    with torch.no_grad():
        losses, metrics, outputs = model(batch, weight_dtype=torch.float32)

    loss_summary = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
    metric_summary = {k: float(v.detach().cpu().item()) for k, v in metrics.items()}
    total_loss = sum(loss_summary.get(name, 0.0) for name in ("pose", "transl", "joints3d"))

    intrinsics = sample0["intrinsics"][0]

    hand_side = resolve_hand_side(sample0, args.hand_side)
    joints3d = sample0["joints3d"][hand_side]

    if hand_side == 0:
        mano_layer = dataset.mano_lhand
    elif hand_side == 1:
        mano_layer = dataset.mano_rhand

    # mano_layer = dataset.mano_lhand

    betas = batch["betas"][0, hand_side]
    pose = batch["pose"][0, hand_side]
    transl = batch["transl"][0, hand_side]

    betas = betas.unsqueeze(0).cpu()
    global_orient = pose[:1].unsqueeze(0).detach().cpu()
    hand_pose = pose[1:].unsqueeze(0).detach().cpu()
    transl = transl.unsqueeze(0).detach().cpu()

    print(betas.shape)
    print(global_orient.shape)
    print(hand_pose.shape)
    print(transl.shape)

    mano_output = mano_layer(
        pose_type="rot6d",
        betas=betas,
        global_orient=global_orient,
        hand_pose=hand_pose,
        transl=transl,
    )

    joints3d_mano = mano_output["joints"]

    mano_side_mask = sample0.get("mano_side_mask", None)
    has_mano_supervision = args.val_dataset_name != "egodex"
    if mano_side_mask is not None and args.val_dataset_name != "egodex":
        has_mano_supervision = float(mano_side_mask[hand_side].item()) > 0.0

    # if has_mano_supervision:
    #     assert (joints3d - joints3d_mano[0]).abs().mean() < 1e-2
    # else:
    #     gt_mano_diff = float((joints3d - joints3d_mano[0]).abs().mean().item())
    #     print(
    #         f"[INFO] No MANO supervision for this sample (dataset={args.val_dataset_name}, hand={hand_side}); "
    #         f"skipping GT MANO consistency assert. mean|joints3d-mano|={gt_mano_diff:.6f}"
    #     )

    
    from model.smpl.utils.renderer import MANORenderer

    renderer = MANORenderer()
    mesh_save_dir = os.path.join("visualization", f"{args.val_dataset_name}_{args.dataset_version}_i{args.sample_index}_s{hand_side}")
    os.makedirs(mesh_save_dir, exist_ok=True)

    if args.val_dataset_name == "egodex":
        _, image_path, frame_idx = dataset._parse_frame_id(frame_id)
        image = dataset._read_or_build_cached_frame(image_path, frame_idx).copy()
    else:
        image_path = dataset.convert_frame_id_to_image_path(frame_id)
        image = cv2.imread(image_path)

    if has_mano_supervision:
        render_bgr_with_image, render_rgba, render_depth = renderer.render_with_image(
            image=image,
            vertices=mano_output["vertices"].numpy(),
            is_rhand=[hand_side],
            focal_length=intrinsics[:2],
            camera_center=intrinsics[2:],
        )

        cv2.imwrite(
            f"visualization/{args.val_dataset_name}_i{args.sample_index}_s{hand_side}__{args.model_version}_{args.dataset_version}__mano_render__GT.png",
            render_bgr_with_image,
        )

    pose_pred = outputs["pose"][0, hand_side]
    transl_pred = outputs["transl"][0, hand_side]

    global_orient_pred = pose_pred[:1].unsqueeze(0).detach().cpu()
    hand_pose_pred = pose_pred[1:].unsqueeze(0).detach().cpu()
    transl_pred = transl_pred.unsqueeze(0).detach().cpu()

    mano_output_pred = mano_layer(
        pose_type="rot6d",
        betas=betas,
        global_orient=global_orient_pred,
        hand_pose=hand_pose_pred,
        transl=transl_pred,
    )

    joints3d_mano_pred = mano_output_pred["joints"]
    pred_vertices_mean_pos = mano_output_pred["vertices"].mean(1).detach().cpu().numpy()
    joint_l1_error = float((joints3d - joints3d_mano_pred[0]).abs().mean().item())
    joint_l2_error = float(torch.norm(joints3d - joints3d_mano_pred[0], dim=-1).mean().item())

    # Diagnostic prints (放在预测结果已就绪的位置)
    print("=== DIAGNOSTIC ===")
    print("dataset:", args.val_dataset_name, "sample_index:", args.sample_index)
    print("loss summary:", loss_summary)
    print("metric summary:", metric_summary)
    print(f"total loss (pose + transl + joints3d): {total_loss:.6f}")
    print("intrinsics (raw):", sample0["intrinsics"])            # fx,fy,cx,cy
    print("intrinsics_resize:", sample0["intrinsics_resize"])
    print("resize_scale (dataset):", getattr(dataset, "resize_scale", None))
    print("extrinsics (3x4):", sample0.get("extrinsics"))
    print("hand_side_mask:", sample0.get("hand_side_mask"))
    print("mano_side_mask:", sample0.get("mano_side_mask"))
    print("hand_side_chosen:", hand_side)
    print("GT joints3d (selected) mean:", joints3d.mean(0))
    print("GT MANO joints mean:", joints3d_mano[0].mean(0))
    print("Pred pose/transl (model): transl_pred (raw tensor):", transl_pred.detach().cpu().numpy())
    print("Pred MANO vertices mean pos:", pred_vertices_mean_pos)
    print(f"GT vs Pred joints: mean_abs={joint_l1_error:.6f}, mean_l2={joint_l2_error:.6f}")
    if has_mano_supervision:
        print("GT MANO transl:", transl.detach().cpu().numpy())
        print("GT MANO vertices mean pos:", mano_output["vertices"].mean(1).detach().cpu().numpy())
    print("==================")

    pred_mesh_path = export_trimesh(
        renderer=renderer,
        vertices=mano_output_pred["vertices"],
        is_rhand=hand_side,
        save_folder=mesh_save_dir,
        file_name="pred_mesh.obj",
    )
    print(f"[VIS] saved predicted mesh to {pred_mesh_path}")

    render_bgr_with_image, render_rgba, render_depth = renderer.render_with_image(
        image=image,
        vertices=mano_output_pred["vertices"].numpy(),
        is_rhand=[hand_side],
        focal_length=intrinsics[:2],
        camera_center=intrinsics[2:],
    )

    cv2.imwrite(
        f"visualization/{args.val_dataset_name}_i{args.sample_index}_s{hand_side}__{args.model_version}_{args.dataset_version}__mano_render.png",
        render_bgr_with_image,
    )

    keypoints_save_path = os.path.join(
        mesh_save_dir,
        "keypoints_3d_gt_pred.png",
    )
    save_3d_joints_figure(
        gt_joints=joints3d,
        pred_joints=joints3d_mano_pred[0],
        save_path=keypoints_save_path,
        title=f"{args.val_dataset_name} sample={args.sample_index} hand={hand_side} | L={total_loss:.6f} | L1={joint_l1_error:.6f}",
    )
    print(f"[VIS] saved 3D keypoint figure to {keypoints_save_path}")

    if has_mano_supervision:
        gt_mesh_path = export_trimesh(
            renderer=renderer,
            vertices=mano_output["vertices"],
            is_rhand=hand_side,
            save_folder=mesh_save_dir,
            file_name="gt_mesh.obj",
        )
        print(f"[VIS] saved GT mesh to {gt_mesh_path}")


def load_state_dict_from_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        safetensors_path = checkpoint_path / "model.safetensors"
        if safetensors_path.exists():
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    "Loading a directory checkpoint requires safetensors to be installed."
                ) from exc

            state_dict = load_file(str(safetensors_path), device="cpu")
        else:
            pkl_candidates = sorted(checkpoint_path.glob("*.pkl"))
            if not pkl_candidates:
                raise FileNotFoundError(f"No supported checkpoint file found in {checkpoint_path}.")
            checkpoint = torch.load(pkl_candidates[0], map_location="cpu")
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    elif checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Loading .safetensors checkpoints requires safetensors to be installed.") from exc

        state_dict = load_file(str(checkpoint_path), device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    if isinstance(state_dict, dict):
        normalized_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            if new_key.startswith("model."):
                new_key = new_key[len("model.") :]
            normalized_state_dict[new_key] = value
        state_dict = normalized_state_dict

    return state_dict


if __name__ == "__main__":
    os.makedirs("visualization", exist_ok=True)

    parser = argparse.ArgumentParser(description="Demo of HandRGBD.")
    args = parser.parse_args()

    args.train_dataset_name = "arctic"
    args.dataset_version = "1_0"
    args.model_version = "1_0"
    args.pretrained_path = "/data/sunzhihao2/turuirui/egohands/experiment/ckpt/v__8_0__arctic_egodex_stage1_single_frame/checkpoint-step-15000"
    args.seed = 618

    model = build_model(args)
    print(f"Loading pretrained weights from {args.pretrained_path}.")
    state_dict = load_state_dict_from_checkpoint(args.pretrained_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    device = torch.device("cuda:0")
    model = model.to(device)

    demo_jobs = [
        dict(val_dataset_name="arctic", dataset_version="1_0", sample_index=720, hand_side=1),
        dict(val_dataset_name="egodex", dataset_version="2_0", sample_index=2020, hand_side=1),
    ]

    for job in demo_jobs:
        args.val_dataset_name = job["val_dataset_name"]
        args.dataset_version = job["dataset_version"]
        args.sample_index = job["sample_index"]
        args.hand_side = job["hand_side"]
        run_demo_for_dataset(args, model, device)

    
    


