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
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import datasets
import torch
from tqdm.auto import tqdm
import transformers

from dataset.arctic__2_0 import ARCTICEgoRGBDataset as ARCTICEgoRGBDataset__2_0
from dataset.hand_rgb__2_0 import collate_fn_rgb
from dataset.egodex__2_0 import EgoDexRGBDataset as EgoDexRGBDataset__2_0
from model.egohands__1_0 import EgoHands as EgoHands__1_0
from experiment.utils.args import parse_args_training

logger = get_logger(__name__, log_level="INFO")


def build_model(args):
    if args.model_version == "1_0":
        return EgoHands__1_0(use_plucker_ray=args.use_plucker_ray)
    else:
        raise ValueError(f"Unsupported model_version: {args.model_version}")


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            value = checkpoint.get(key, None)
            if isinstance(value, dict):
                return value

        # Some checkpoints are already plain state_dict.
        if all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

    raise ValueError("Unsupported checkpoint format. Cannot find a valid state_dict.")


def _normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        normalized[new_key] = value
    return normalized


def _compute_loss(losses):
    return losses["pose"] + losses["transl"] + losses["joints3d"]


class BalancedTwoDatasetBatchSampler:
    def __init__(self, dataset_lengths: List[int], batch_size: int, seed: int | None = None):
        if len(dataset_lengths) != 2:
            raise ValueError("BalancedTwoDatasetBatchSampler expects exactly two datasets.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size % 2 != 0:
            raise ValueError("train_batch_size must be even to keep a strict 1:1 mix per batch.")

        self.dataset_lengths = dataset_lengths
        self.batch_size = batch_size
        self.half_batch_size = batch_size // 2
        self.seed = 0 if seed is None else int(seed)
        self.epoch = 0
        self.num_batches = max(
            math.ceil(dataset_lengths[0] / self.half_batch_size),
            math.ceil(dataset_lengths[1] / self.half_batch_size),
        )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        dataset_a_length, dataset_b_length = self.dataset_lengths
        dataset_a_offset = 0
        dataset_b_offset = dataset_a_length

        for _ in range(self.num_batches):
            indices_a = torch.randint(
                low=0,
                high=dataset_a_length,
                size=(self.half_batch_size,),
                generator=generator,
            ).tolist()
            indices_b = torch.randint(
                low=0,
                high=dataset_b_length,
                size=(self.half_batch_size,),
                generator=generator,
            ).tolist()

            batch_indices = [dataset_a_offset + index for index in indices_a]
            batch_indices.extend(dataset_b_offset + index for index in indices_b)
            permutation = torch.randperm(len(batch_indices), generator=generator).tolist()
            yield [batch_indices[index] for index in permutation]


def _validate_stage1_args(args):
    if args.model_version != "1_0":
        raise ValueError("train_with_egodex.py only supports --model_version 1_0.")

    if args.sequence_length != 1:
        raise ValueError(
            "Stage-1 mixed training must be single-frame. Please set --sequence_length 1."
        )


def _parse_multi_dataset_args(args, split: str) -> List[Tuple[str, str]]:
    return [("arctic", "2_0"), ("egodex", "2_0")]


def _build_single_dataset(dataset_name: str, dataset_version: str, args, split: str):
    if dataset_name == "arctic" and dataset_version == "2_0":
        dataset_args = dict(
            dataset_folder="/data/sunzhihao2/turuirui/dataset/arctic/data/data",
            image_size=(512, 512),
            split=split,
            setup="p1",
            seed=args.seed,
        )
        return ARCTICEgoRGBDataset__2_0(
            sequence_length=args.sequence_length,
            **dataset_args,
        )

    if dataset_name == "egodex" and dataset_version == "2_0":
        dataset_args = dict(
            dataset_folder="/data/public/ego/egodex",
            image_size=(512, 512),
            split=split,
            setup="default",
            seed=args.seed,
        )
        return EgoDexRGBDataset__2_0(
            sequence_length=args.sequence_length,
            **dataset_args,
        )

    raise ValueError(
        f"Unsupported dataset config for train_with_egodex.py: dataset_name={dataset_name}, dataset_version={dataset_version}"
    )


def _is_body_backbone_param(name: str) -> bool:
    return name.startswith("body_backbone.") or ".body_backbone." in name


def _is_hand_backbone_param(name: str) -> bool:
    return name.startswith("hand_backbone.") or ".hand_backbone." in name


def _freeze_hamer_encoder_only(model):
    trainable_params = 0
    frozen_hamer_encoder_params = 0
    for name, param in model.named_parameters():
        is_hand = _is_hand_backbone_param(name)
        # Stage-1: keep single-frame model and only freeze HaMeR encoder.
        is_trainable = not is_hand

        param.requires_grad = is_trainable
        if is_trainable:
            trainable_params += param.numel()
        else:
            frozen_hamer_encoder_params += param.numel()

    return trainable_params, frozen_hamer_encoder_params


def _collect_trainable_param_groups(model):
    body_backbone_params = []
    other_trainable_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_body_backbone_param(name):
            body_backbone_params.append(param)
        else:
            other_trainable_params.append(param)
    return other_trainable_params, body_backbone_params


def _run_validation(model, val_dataloader, accelerator, weight_dtype):
    model.eval()

    val_loss_epoch = 0.0
    val_loss_pose_epoch = 0.0
    val_loss_transl_epoch = 0.0
    val_loss_joints3d_epoch = 0.0
    val_mpjpe_epoch = 0.0
    val_pa_mpjpe_epoch = 0.0
    val_steps_epoch = 0

    validation_progress_bar = tqdm(
        val_dataloader,
        desc="Validation Steps",
        leave=False,
        disable=not accelerator.is_local_main_process,
    )

    for batch in validation_progress_bar:
        with torch.no_grad():
            with accelerator.autocast():
                losses, metrics, _ = model(batch, weight_dtype)

                loss = _compute_loss(losses)
                pose_loss = losses["pose"]
                transl_loss = losses["transl"]
                joints3d_loss = losses["joints3d"]

                mpjpe = metrics["mpjpe"]
                pa_mpjpe = metrics["pa_mpjpe"]

            statistics = accelerator.gather(
                torch.stack([loss, pose_loss, transl_loss, joints3d_loss, mpjpe, pa_mpjpe]).unsqueeze(0)
            )

            logs = {
                "L_p": statistics[:, 1].mean().item(),
                "L_t": statistics[:, 2].mean().item(),
                "L_j": statistics[:, 3].mean().item(),
                "mpjpe": statistics[:, 4].mean().item(),
                "pa_mpjpe": statistics[:, 5].mean().item(),
            }
            validation_progress_bar.set_postfix(**logs)

            val_loss_epoch += statistics[:, 0].mean().item()
            val_loss_pose_epoch += statistics[:, 1].mean().item()
            val_loss_transl_epoch += statistics[:, 2].mean().item()
            val_loss_joints3d_epoch += statistics[:, 3].mean().item()
            val_mpjpe_epoch += statistics[:, 4].mean().item()
            val_pa_mpjpe_epoch += statistics[:, 5].mean().item()
            val_steps_epoch += 1

    model.train()

    return {
        "loss": val_loss_epoch / max(val_steps_epoch, 1),
        "loss_pose": val_loss_pose_epoch / max(val_steps_epoch, 1),
        "loss_transl": val_loss_transl_epoch / max(val_steps_epoch, 1),
        "loss_joints3d": val_loss_joints3d_epoch / max(val_steps_epoch, 1),
        "mpjpe": val_mpjpe_epoch / max(val_steps_epoch, 1),
        "pa_mpjpe": val_pa_mpjpe_epoch / max(val_steps_epoch, 1),
    }


def build_dataset(args, split):
    dataset_configs = _parse_multi_dataset_args(args, split=split)
    datasets_to_concat = [
        _build_single_dataset(dataset_name=name, dataset_version=version, args=args, split=split)
        for name, version in dataset_configs
    ]

    if len(datasets_to_concat) == 1:
        return datasets_to_concat[0], dataset_configs

    return torch.utils.data.ConcatDataset(datasets_to_concat), dataset_configs


def main():
    args = parse_args_training()

    _validate_stage1_args(args)

    # [STRUCTURE] Set up training environment.
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # [STRUCTURE] Build dataloaders.
    # Build datasets in a main-process-first section to avoid cache file races
    # when multiple ranks create/read the same sequence cache concurrently.
    with accelerator.main_process_first():
        train_dataset, train_dataset_configs = build_dataset(args, split="train")
    train_batch_sampler = BalancedTwoDatasetBatchSampler(
        dataset_lengths=[len(dataset) for dataset in train_dataset.datasets],
        batch_size=args.train_batch_size,
        seed=args.seed,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn_rgb,
    )

    with accelerator.main_process_first():
        val_dataset, val_dataset_configs = build_dataset(args, split="val")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn_rgb,
    )
    
    # [STRUCTURE] Build model and optimization components.
    model = build_model(args)
    # Load pretrained weights only when the user explicitly provided a valid path.
    if getattr(args, "pretrained_path", None):
        if os.path.exists(args.pretrained_path):
            logger.info(f"Loading pretrained weights from {args.pretrained_path}.")
            checkpoint = torch.load(args.pretrained_path, map_location="cpu")
            state_dict = _extract_state_dict(checkpoint)
            state_dict = _normalize_state_dict_keys(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            logger.info(
                "Pretrained load done. missing_keys=%d, unexpected_keys=%d",
                len(missing_keys),
                len(unexpected_keys),
            )
        else:
            logger.warning(
                "Pretrained checkpoint path provided but not found: %s. Continuing with random init.",
                args.pretrained_path,
            )
    else:
        logger.info("No pretrained checkpoint provided; initializing model randomly.")
    model.train()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    trainable_params, frozen_hamer_encoder_params = _freeze_hamer_encoder_only(model)
    logger.info(
        "HaMeR encoder frozen for stage-1. trainable_params=%d, frozen_hamer_encoder_params=%d",
        trainable_params,
        frozen_hamer_encoder_params,
    )

    # Initialize optimizer for stage-1:
    # - base lr for non-backbone trainable params
    # - scaled lr only for trainable body backbone params
    other_trainable_params, body_backbone_params = _collect_trainable_param_groups(model)
    optimizer_param_groups = []
    if len(other_trainable_params) > 0:
        optimizer_param_groups.append(
            {
                "params": other_trainable_params,
                "lr": args.learning_rate,
            }
        )
    if len(body_backbone_params) > 0:
        optimizer_param_groups.append(
            {
                "params": body_backbone_params,
                "lr": args.learning_rate * args.learning_rate_scale,
            }
        )
    if len(optimizer_param_groups) == 0:
        raise ValueError("No trainable parameters found after freezing HaMeR encoder.")

    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Prepare everything with our `accelerator`.
    model = accelerator.prepare(model)
    train_dataloader = accelerator.prepare(train_dataloader)
    val_dataloader = accelerator.prepare(val_dataloader)
    optimizer = accelerator.prepare(optimizer)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        if args.num_train_epochs != 1:
            logging.warning(
                "train_with_egodex.py uses one full epoch when --max_train_steps is not set. "
                "Overriding --num_train_epochs=%s to 1.",
                args.num_train_epochs,
            )
        args.num_train_epochs = 1
        max_train_steps = num_update_steps_per_epoch
    else:
        if args.max_train_steps <= 0:
            raise ValueError("--max_train_steps must be a positive integer when provided.")
        max_train_steps = args.max_train_steps
        args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

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

    if args.checkpointing_steps is not None and args.checkpointing_steps <= 0:
        raise ValueError("--checkpointing_steps must be a positive integer when provided.")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    logger.info(f"Train datasets = {train_dataset_configs}")
    logger.info(f"Val datasets = {val_dataset_configs}")
    logger.info(f"Num examples = {len(train_dataset)}")
    logger.info(f"Train dataloader batches per epoch = {len(train_dataloader)}")
    logger.info(f"Update steps per epoch = {num_update_steps_per_epoch}")
    logger.info(f"Num epochs = {args.num_train_epochs}")
    logger.info(f"Max train steps (effective) = {max_train_steps}")
    logger.info(f"Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0
    # [STRUCTURE] Resume training state if needed.
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
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
            accelerator.print(f"Restored training state to epoch {first_epoch}, global step {global_step}, scheduler step {scheduler.last_epoch}.")
    
    else:
        initial_global_step = 0

    # [STRUCTURE] Run the training and validation loop.
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Training Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    best_validation_mpjpe = float("inf")

    for epoch in range(first_epoch, args.num_train_epochs):
        train_batch_sampler.set_epoch(epoch)

        # Train for one epoch.
        train_loss_epoch = 0.0
        train_loss_pose_epoch = 0.0
        train_loss_transl_epoch = 0.0
        train_loss_joints3d_epoch = 0.0
        train_mpjpe_epoch = 0.0
        train_pa_mpjpe_epoch = 0.0
        train_steps_epoch = 0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    losses, metrics, _ = model(batch, weight_dtype)

                    loss = _compute_loss(losses)
                    pose_loss = losses["pose"]
                    transl_loss = losses["transl"]
                    joints3d_loss = losses["joints3d"]

                    mpjpe = metrics["mpjpe"]
                    pa_mpjpe = metrics["pa_mpjpe"]

                # Gather the losses across all processes for logging (if we use distributed training).
                statistics = accelerator.gather(
                    torch.stack([loss, pose_loss, transl_loss, joints3d_loss, mpjpe, pa_mpjpe]).unsqueeze(0)
                )

                # Backpropagate
                accelerator.backward(loss)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    logs = {
                        "L_p": statistics[:, 1].mean().item(),
                        "L_t": statistics[:, 2].mean().item(),
                        "L_j": statistics[:, 3].mean().item(),
                        "mpjpe": statistics[:, 4].mean().item(),
                        "pa_mpjpe": statistics[:, 5].mean().item(),
                        "lr_base": scheduler.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)

                    train_loss_epoch += statistics[:, 0].mean().item()
                    train_loss_pose_epoch += statistics[:, 1].mean().item()
                    train_loss_transl_epoch += statistics[:, 2].mean().item()
                    train_loss_joints3d_epoch += statistics[:, 3].mean().item()
                    train_mpjpe_epoch += statistics[:, 4].mean().item()
                    train_pa_mpjpe_epoch += statistics[:, 5].mean().item()
                    train_steps_epoch += 1

                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log(logs, step=global_step)

                    if (
                        accelerator.is_main_process
                        and args.checkpointing_steps is not None
                        and global_step % args.checkpointing_steps == 0
                    ):
                        save_path = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved step checkpoint to {save_path}.")

                    if global_step >= max_train_steps:
                        break

        val_metrics = _run_validation(model, val_dataloader, accelerator, weight_dtype)

        if accelerator.is_main_process:
            logger.info("")
            logger.info(f"Statistic of epoch {epoch + 1}:")
            logger.info(f"[TRAIN] loss: {(train_loss_epoch / max(train_steps_epoch, 1)):.4f}")
            logger.info(f"[TRAIN] Loss_pose: {(train_loss_pose_epoch / max(train_steps_epoch, 1)):.4f}")
            logger.info(f"[TRAIN] Loss_transl: {(train_loss_transl_epoch / max(train_steps_epoch, 1)):.4f}")
            logger.info(f"[TRAIN] Loss_joints3d: {(train_loss_joints3d_epoch / max(train_steps_epoch, 1)):.4f}")
            logger.info(f"[TRAIN] MPJPE: {(train_mpjpe_epoch / max(train_steps_epoch, 1)):.4f}")
            logger.info(f"[TRAIN] PA-MPJPE: {(train_pa_mpjpe_epoch / max(train_steps_epoch, 1)):.4f}")
            logger.info("")
            logger.info(f"[VAL] Loss: {val_metrics['loss']:.4f}")
            logger.info(f"[VAL] Loss_pose: {val_metrics['loss_pose']:.4f}")
            logger.info(f"[VAL] Loss_transl: {val_metrics['loss_transl']:.4f}")
            logger.info(f"[VAL] Loss_joints3d: {val_metrics['loss_joints3d']:.4f}")
            logger.info(f"[VAL] MPJPE: {val_metrics['mpjpe']:.4f}")
            logger.info(f"[VAL] PA-MPJPE: {val_metrics['pa_mpjpe']:.4f}")
            logger.info("")

            unwrapped_model = accelerator.unwrap_model(model)
            validation_mpjpe = val_metrics["mpjpe"]

            if validation_mpjpe < best_validation_mpjpe:
                best_validation_mpjpe = validation_mpjpe
                best_mpjpe_path = os.path.join(args.output_dir, "best_mpjpe.pth")
                stage1_best_mpjpe_path = os.path.join(args.output_dir, "stage1_best_mpjpe.pth")

                torch.save(unwrapped_model.state_dict(), best_mpjpe_path)
                torch.save(unwrapped_model.state_dict(), stage1_best_mpjpe_path)
                logger.info(f"Saved best checkpoint to {best_mpjpe_path}.")
                logger.info(f"Saved stage-1 best checkpoint to {stage1_best_mpjpe_path}.")

        if accelerator.is_main_process and (epoch + 1) % args.checkpointing_epochs == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}.")

        if global_step >= max_train_steps:
            break
        
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)

        model_path = os.path.join(args.output_dir, "model.pth")
        stage1_model_path = os.path.join(args.output_dir, "stage1_model.pth")
        torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), stage1_model_path)
        logger.info(f"Saved final checkpoint to {model_path}.")
        logger.info(f"Saved stage-1 final checkpoint to {stage1_model_path}.")

    accelerator.end_training()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        rank = os.environ.get("RANK", "unknown")
        local_rank = os.environ.get("LOCAL_RANK", "unknown")
        logger.exception("Fatal error on rank=%s local_rank=%s", rank, local_rank)
        raise
