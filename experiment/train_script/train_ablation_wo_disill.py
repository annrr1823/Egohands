import logging
import math
import os
from typing import List, Tuple

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md"],
    pythonpath=True,
    dotenv=True,
)

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import datasets
import torch
from tqdm.auto import tqdm
import transformers
from safetensors.torch import load_file as load_safetensors

from dataset.hand_rgb__2_0 import collate_fn_rgb
from dataset.dexycb__1_0 import DexYCBRGBDataset as DexYCBRGBDataset__1_0
from dataset.arctic__1_0 import ARCTICEgoRGBDataset as ARCTICEgoRGBDataset__1_0
from dataset.arctic__2_0 import ARCTICEgoRGBDataset as ARCTICEgoRGBDataset__2_0
from dataset.h2o__1_0 import H2OEgoRGBDataset as H2OEgoRGBDataset__1_0
from dataset.h2o__2_0 import H2OEgoRGBDataset as H2OEgoRGBDataset__2_0
from model.egohands__1_0 import EgoHands as EgoHands__1_0
from experiment.utils.args import parse_args_training

logger = get_logger(__name__, log_level="INFO")


def build_model(args):
    if args.model_version == "1_0":
        return EgoHands__1_0(use_plucker_ray=args.use_plucker_ray)
    raise ValueError(f"Unsupported model_version: {args.model_version}")


def _load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str, strict: bool = True):
    if checkpoint_path.endswith(".safetensors"):
        state_dict = load_safetensors(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    model.load_state_dict(state_dict, strict=strict)


def _parse_multi_dataset_args(args, split: str) -> List[Tuple[str, str]]:
    dataset_name_value = args.dataset_name
    dataset_version_value = args.dataset_version

    if split == "val":
        if getattr(args, "val_dataset_name", None):
            dataset_name_value = args.val_dataset_name
        if getattr(args, "val_dataset_version", None):
            dataset_version_value = args.val_dataset_version

    dataset_names = [item.strip() for item in dataset_name_value.split(",") if item.strip()]
    dataset_versions = [item.strip() for item in dataset_version_value.split(",") if item.strip()]

    if len(dataset_names) == 0:
        raise ValueError(f"dataset_name for split='{split}' is empty.")

    if len(dataset_versions) == 1 and len(dataset_names) > 1:
        dataset_versions = dataset_versions * len(dataset_names)

    if len(dataset_names) != len(dataset_versions):
        raise ValueError(
            f"The number of dataset names ({len(dataset_names)}) must match the number of dataset versions ({len(dataset_versions)})."
        )

    return list(zip(dataset_names, dataset_versions))


def _build_single_dataset(dataset_name: str, dataset_version: str, args, split: str, camera_source: str = "auto"):
    if dataset_name == "arctic":
        dataset_args = dict(
            dataset_folder="/data/sunzhihao2/turuirui/dataset/arctic/data/data",
            image_size=(512, 512),
            split=split,
            setup="p1",
            camera_cache_file="/data/sunzhihao2/turuirui/arctic_extrinsics/arctic_camera_cache_with_extrinsics.pkl",
            camera_source=camera_source,
            seed=args.seed,
        )
        if dataset_version == "1_0":
            return ARCTICEgoRGBDataset__1_0(**dataset_args)
        if dataset_version == "2_0":
            return ARCTICEgoRGBDataset__2_0(
                sequence_length=args.sequence_length,
                **dataset_args,
            )
    elif dataset_name == "h2o":
        dataset_args = dict(
            dataset_folder="/data/public/hand/h2o",
            image_size=(512, 512),
            split=split,
            setup="default",
            seed=args.seed,
        )
        if dataset_version == "1_0":
            return H2OEgoRGBDataset__1_0(**dataset_args)
        if dataset_version == "2_0":
            return H2OEgoRGBDataset__2_0(
                sequence_length=args.sequence_length,
                **dataset_args,
            )

    raise ValueError(
        f"Unsupported dataset config: dataset_name={dataset_name}, dataset_version={dataset_version}"
    )


def build_dataset(args, split):
    camera_source = args.camera_source if split == "train" else (args.val_camera_source or "auto")
    dataset_configs = _parse_multi_dataset_args(args, split=split)
    datasets_to_concat = [
        _build_single_dataset(
            dataset_name=name,
            dataset_version=version,
            args=args,
            split=split,
            camera_source=camera_source,
        )
        for name, version in dataset_configs
    ]

    if len(datasets_to_concat) == 1:
        return datasets_to_concat[0], dataset_configs

    return torch.utils.data.ConcatDataset(datasets_to_concat), dataset_configs


def main():
    args = parse_args_training()

    if getattr(args, "teacher_pretrained_path", None):
        logger.warning("teacher_pretrained_path is set but this ablation script does not use teacher distillation. It will be ignored.")
    if getattr(args, "distill_weight", 0) > 0:
        logger.warning("distill_weight > 0 is ignored in this ablation script (no distillation).")

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    train_dataset, train_dataset_configs = build_dataset(args, split="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn_rgb,
    )

    val_dataset = None
    val_dataset_configs = None
    val_dataloader = None
    if args.do_eval:
        val_dataset, val_dataset_configs = build_dataset(args, split="val")
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            collate_fn=collate_fn_rgb,
        )

    model = build_model(args)
    if args.pretrained_path:
        logger.info(f"Loading pretrained weights from {args.pretrained_path}.")
        _load_model_checkpoint(model, args.pretrained_path, strict=True)

    model.train()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    backbone_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("body_backbone.") or name.startswith("hand_backbone."):
            backbone_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decoder_params,
                "lr": args.learning_rate,
            },
            {
                "params": backbone_params,
                "lr": args.learning_rate * args.learning_rate_scale,
            },
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    model = accelerator.prepare(model)
    train_dataloader = accelerator.prepare(train_dataloader)
    optimizer = accelerator.prepare(optimizer)
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=[args.learning_rate, args.learning_rate * args.learning_rate_scale],
        total_steps=max_train_steps,
        pct_start=args.scheduler_pct_start,
        anneal_strategy=args.scheduler_anneal_strategy,
        div_factor=args.scheduler_div_factor,
        final_div_factor=args.scheduler_final_div_factor,
    )
    accelerator.register_for_checkpointing(scheduler)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    logger.info(f"Train datasets = {train_dataset_configs}")
    logger.info(f"Val datasets = {val_dataset_configs}")
    logger.info(f"Num examples = {len(train_dataset)}")
    logger.info(f"Num epochs = {args.num_train_epochs}")
    logger.info(f"Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            first_epoch = int(path.split("-")[-1])

            global_step = first_epoch * num_update_steps_per_epoch
            initial_global_step = global_step
            accelerator.print(
                f"Restored training state to epoch {first_epoch}, global step {global_step}, scheduler step {scheduler.last_epoch}."
            )
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Training Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss_epoch = 0.0
        train_loss_pose_epoch, train_loss_transl_epoch, train_loss_joints3d_epoch = 0.0, 0.0, 0.0
        train_mpjpe_epoch = 0.0
        train_pa_mpjpe_epoch = 0.0
        train_steps_epoch = 0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    losses, metrics, _ = model(batch, weight_dtype, return_intermediate=True)

                    pose_loss = losses["pose"]
                    transl_loss = losses["transl"]
                    joints3d_loss = losses["joints3d"]

                    loss = pose_loss + transl_loss + joints3d_loss

                    mpjpe = metrics["mpjpe"]
                    pa_mpjpe = metrics["pa_mpjpe"]

                    distill_loss_weighted = torch.zeros_like(loss)
                    mano_token_distill = torch.zeros_like(loss)

                statistics = accelerator.gather(
                    torch.stack(
                        [
                            loss,
                            pose_loss,
                            transl_loss,
                            joints3d_loss,
                            distill_loss_weighted,
                            mano_token_distill,
                            mpjpe,
                            pa_mpjpe,
                        ]
                    ).unsqueeze(0)
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    logs = {
                        "L_p": statistics[:, 1].mean().item(),
                        "L_t": statistics[:, 2].mean().item(),
                        "L_j": statistics[:, 3].mean().item(),
                        "L_distill": statistics[:, 4].mean().item(),
                        "L_distill_mano_token": statistics[:, 5].mean().item(),
                        "mpjpe": statistics[:, 6].mean().item(),
                        "pa_mpjpe": statistics[:, 7].mean().item(),
                        "lr_base": scheduler.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)

                    train_loss_epoch += statistics[:, 0].mean().item()
                    train_loss_pose_epoch += statistics[:, 1].mean().item()
                    train_loss_transl_epoch += statistics[:, 2].mean().item()
                    train_loss_joints3d_epoch += statistics[:, 3].mean().item()
                    train_mpjpe_epoch += statistics[:, 6].mean().item()
                    train_pa_mpjpe_epoch += statistics[:, 7].mean().item()
                    train_steps_epoch += 1

                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log(logs, step=global_step)

        model.train()

        if accelerator.is_main_process:
            logger.info("")
            logger.info(f"Statistic of epoch {epoch + 1}:")
            logger.info(f"[TRAIN] loss: {(train_loss_epoch / train_steps_epoch):.4f}")
            logger.info(
                f"[TRAIN] Loss_pose: {(train_loss_pose_epoch / train_steps_epoch):.4f}, "
                f"Loss_transl: {(train_loss_transl_epoch / train_steps_epoch):.4f}, "
                f"Loss_joints3d: {(train_loss_joints3d_epoch / train_steps_epoch):.4f}, "
            )
            logger.info("[TRAIN] Loss_distill(weighted): 0.0000 (ablation: no teacher/distillation)")
            logger.info(f"[TRAIN] MPJPE: {(train_mpjpe_epoch / train_steps_epoch):.4f}")
            logger.info(f"[TRAIN] PA-MPJPE: {(train_pa_mpjpe_epoch / train_steps_epoch):.4f}")
            logger.info("")

            if args.do_eval and val_dataloader is not None:
                logger.info("[VAL] Skipped in train-only mode is disabled.")

            _ = accelerator.unwrap_model(model)

            if (epoch + 1) % args.checkpointing_epochs == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}.")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)

        model_path = os.path.join(args.output_dir, "model.pth")
        torch.save(model.state_dict(), model_path)

    accelerator.end_training()


if __name__ == "__main__":
    main()
