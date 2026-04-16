import logging
import math
import os

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
import torch.nn.functional as F
from tqdm.auto import tqdm
import transformers

from dataset.hand_rgb__2_0 import collate_fn_rgb
from dataset.dexycb__1_0 import DexYCBRGBDataset as DexYCBRGBDataset__1_0
from dataset.arctic__1_0 import ARCTICEgoRGBDataset as ARCTICEgoRGBDataset__1_0
from dataset.arctic__2_0 import ARCTICEgoRGBDataset as ARCTICEgoRGBDataset__2_0
from dataset.h2o__1_0 import H2OEgoRGBDataset as H2OEgoRGBDataset__1_0
from model.egohands__1_0 import EgoHands as EgoHands__1_0
from experiment.utils.args import parse_args_training

logger = get_logger(__name__, log_level="INFO")


def build_model(args):
    if args.model_version == "1_0":
        return EgoHands__1_0(use_plucker_ray=args.use_plucker_ray)
    else:
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
            dataset_folder="/data/sunzhihao2/turuirui/dataset/arctic/data/data",
            image_size=(512, 512),
            split=split,
            setup="p1",
            seed=args.seed,
        )
        if args.dataset_version == "1_0":
            return ARCTICEgoRGBDataset__1_0(**dataset_args)
        if args.dataset_version == "2_0":
            return ARCTICEgoRGBDataset__2_0(
                sequence_length=args.sequence_length,
                **dataset_args,
            )
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


def main():
    args = parse_args_training()

    # [STRUCTURE] Set up training environment.
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
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
    train_dataset = build_dataset(args, split="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn_rgb,
    )

    val_dataset = build_dataset(args, split="val")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn_rgb,
    )
    
    # [STRUCTURE] Build model and optimization components.
    model = build_model(args)
    if args.pretrained_path:
        logger.info(f"Loading pretrained weights from {args.pretrained_path}.")
        checkpoint = torch.load(args.pretrained_path, map_location="cpu")
        # checkpoint = torch.load("/data/sunzhihao2/turuirui/egohands/third_party/ckpt/model_dino.ckpt", map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=not args.use_plucker_ray)
    model.train()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
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

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    logger.info(f"Num examples = {len(train_dataset)}")
    logger.info(f"Num epochs = {args.num_train_epochs}")
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
        # Train for one epoch.
        train_loss_epoch = 0.0
        train_loss_pose_epoch, train_loss_transl_epoch, train_loss_joints3d_epoch = 0.0, 0.0, 0.0
        train_mpjpe_epoch = 0.0
        train_pa_mpjpe_epoch = 0.0
        train_steps_epoch = 0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    losses, metrics, _ = model(batch, weight_dtype)

                    pose_loss = losses["pose"]
                    transl_loss = losses["transl"]
                    joints3d_loss = losses["joints3d"]
                    
                    loss = pose_loss + transl_loss + joints3d_loss

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

        # Validate after the training epoch.
        model.eval()

        val_loss_epoch = 0.0
        val_loss_pose_epoch, val_loss_transl_epoch, val_loss_joints3d_epoch = 0.0, 0.0, 0.0
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

                    pose_loss = losses["pose"]
                    transl_loss = losses["transl"]
                    joints3d_loss = losses["joints3d"]
                    
                    loss = pose_loss + transl_loss + joints3d_loss

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

        if accelerator.is_main_process:
            logger.info('')
            logger.info(f"Statistic of epoch {epoch + 1}:")
            logger.info(f"[TRAIN] loss: {(train_loss_epoch / train_steps_epoch):.4f}")
            logger.info(f"[TRAIN] Loss_pose: {(train_loss_pose_epoch / train_steps_epoch):.4f}, " \
                        f"Loss_transl: {(train_loss_transl_epoch / train_steps_epoch):.4f}, " \
                        f"Loss_joints3d: {(train_loss_joints3d_epoch / train_steps_epoch):.4f}, " )
            logger.info(f"[TRAIN] MPJPE: {(train_mpjpe_epoch / train_steps_epoch):.4f}")
            logger.info(f"[TRAIN] PA-MPJPE: {(train_pa_mpjpe_epoch / train_steps_epoch):.4f}")
            logger.info('')
            logger.info(f"[VAL] Loss: {(val_loss_epoch / val_steps_epoch):.4f}")
            logger.info(f"[VAL] Loss_pose: {(val_loss_pose_epoch / val_steps_epoch):.4f}, " \
                        f"Loss_transl: {(val_loss_transl_epoch / val_steps_epoch):.4f}, " \
                        f"Loss_joints3d: {(val_loss_joints3d_epoch / val_steps_epoch):.4f}, " )
            logger.info(f"[VAL] MPJPE: {(val_mpjpe_epoch / val_steps_epoch):.4f}")
            logger.info(f"[VAL] PA-MPJPE: {(val_pa_mpjpe_epoch / val_steps_epoch):.4f}")
            logger.info('')

            unwrapped_model = accelerator.unwrap_model(model)

            validation_mpjpe = val_mpjpe_epoch / val_steps_epoch

            if validation_mpjpe < best_validation_mpjpe:
                best_validation_mpjpe = validation_mpjpe
                best_mpjpe_path = os.path.join(args.output_dir, "best_mpjpe.pth")

                torch.save(unwrapped_model.state_dict(), best_mpjpe_path)
                logger.info(f"Saved best checkpoint to {best_mpjpe_path}.")

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
