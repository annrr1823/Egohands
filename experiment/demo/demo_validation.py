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

from dataset.hand_rgb__1_0 import collate_fn_rgb
from dataset.arctic__1_0 import ARCTICEgoRGBDataset as ARCTICEgoRGBDataset__1_0
from dataset.h2o__1_0 import H2OEgoRGBDataset as H2OEgoRGBDataset__1_0
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


def build_model(args):
    if args.model_version == "1_0":
        return EgoHands__1_0()
    else:
        raise ValueError(f"Unsupported model_version: {args.model_version}")


def build_dataset(args, split):
    if args.val_dataset_name == "arctic":
        dataset_args = dict(
            dataset_folder="/data/public/dataset/hand/arctic",
            image_size=(512, 512),
            split=split,
            setup="p1",
            seed=args.seed,
        )
        if args.dataset_version == "1_0":
            return ARCTICEgoRGBDataset__1_0(**dataset_args)
    elif args.val_dataset_name == "h2o":
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
        f"Unsupported dataset config: dataset_name={args.val_dataset_name}, dataset_version={args.dataset_version}"
    )


if __name__ == "__main__":
    os.makedirs("visualization", exist_ok=True)

    parser = argparse.ArgumentParser(description="Demo of HandRGBD.")
    args = parser.parse_args()

    args.train_dataset_name = "h2o"
    args.dataset_version = "1_0"
    args.model_version = "1_0"
    args.pretrained_path = f"/data/sunzhihao/egohands/experiment/ckpt/v__{args.model_version}__{args.train_dataset_name}_{args.dataset_version}/best_mpjpe.pth"
    args.seed = 618

    args.val_dataset_name = "h2o"
    args.sample_index = 620
    args.hand_side = 1

    dataset = build_dataset(args, split="val")
    sample = dataset[args.sample_index]

    frame_id = sample["meta"]["frame_id"]
    print(f"[INFO] Visualizing sample with frame_id {frame_id}.")
    
    model = build_model(args)
    print(f"Loading pretrained weights from {args.pretrained_path}.")
    checkpoint = torch.load(args.pretrained_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    device = torch.device("cuda:0")
    model = model.to(device)

    batch = collate_fn_rgb([sample])
    batch = batch_to_device(batch, device)
    
    with torch.no_grad():
        losses, metrics, outputs = model(batch, weight_dtype=torch.float32)

    print(losses)
    print(metrics)
    
    intrinsics = sample["intrinsics"][0]

    # hand_side = sample["hand_side"]
    hand_side = args.hand_side
    joints3d = sample["joints3d"][hand_side]

    if hand_side == 0:
        mano_layer = dataset.mano_lhand
    elif hand_side == 1:
        mano_layer = dataset.mano_rhand

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
        transl=transl
    )

    joints3d_mano = mano_output["joints"]
    vertices = mano_output["vertices"]

    assert (joints3d - joints3d_mano[0]).abs().mean() < 1e-2

    from model.smpl.utils.renderer import MANORenderer
    renderer = MANORenderer()

    subject_id = Path(frame_id).parent.parent.parent.name

    image_path = dataset.convert_frame_id_to_image_path(frame_id)
    image = cv2.imread(image_path)

    render_bgr_with_image, render_rgba, render_depth = renderer.render_with_image(
        image=image,
        vertices=mano_output["vertices"].numpy(),
        is_rhand=[hand_side],
        focal_length=intrinsics[:2],
        camera_center=intrinsics[2:]
    )

    cv2.imwrite(f"visualization/{args.val_dataset_name}_i{args.sample_index}_s{args.hand_side}__{args.model_version}_{args.dataset_version}__mano_render__GT.png", render_bgr_with_image)

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
        transl=transl_pred
    )

    joints3d_mano_pred = mano_output_pred["joints"]

    print((joints3d - joints3d_mano_pred[0]).abs().mean())

    render_bgr_with_image, render_rgba, render_depth = renderer.render_with_image(
        image=image,
        vertices=mano_output_pred["vertices"].numpy(),
        is_rhand=[hand_side],
        focal_length=intrinsics[:2],
        camera_center=intrinsics[2:]
    )

    cv2.imwrite(f"visualization/{args.val_dataset_name}_i{args.sample_index}_s{args.hand_side}__{args.model_version}_{args.dataset_version}__mano_render.png", render_bgr_with_image)

    

    
    
