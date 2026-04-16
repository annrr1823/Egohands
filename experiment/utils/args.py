import argparse
import os


def parse_args_training():
    parser = argparse.ArgumentParser(description="Training script of EgoHand.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="hands_net",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # accelerate and torch
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="handsnet-train",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    # seed
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    # dataset
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # model
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        # default="/data/sunzhihao2/turuirui/egohands/third_party/ckpt/model_dino.ckpt",
        required=False,
        help="Path to pretrained model.",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="1_0",
        choices=["1_0", "2_0", "3_0"],
        help="Model version to instantiate.",
    )
    parser.add_argument(
        "--use_plucker_ray",
        action="store_true",
        help="Fuse patch-level Plucker ray geometry into the body token stream.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="h2o",
        choices=["h2o", "arctic"],
        help="Dataset source to instantiate.",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        default="1_0",
        choices=["1_0", "2_0"],
        help="Dataset version to instantiate.",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=8,
        help="Temporal window length for sequence datasets (e.g., arctic 2_0).",
    )
    parser.add_argument(
        "--point_dropout_prob",
        type=float,
        default=0.1,
        help="Random point dropout probability for augmented dataset.",
    )
    parser.add_argument(
        "--point_translation_std",
        type=float,
        default=0.01,
        help="Translation std in canonical space for augmented dataset.",
    )
    parser.add_argument(
        "--point_rotation_deg",
        type=float,
        default=10.0,
        help="Max absolute z-axis rotation angle in degrees for augmented dataset.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_epochs`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    # optimizer
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_scale",
        type=float,
        default=0.5,
        help="Learning rate scale for backbone params.",
    )
    
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")

    # scheduler
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="onecycle",
        choices=["onecycle"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--scheduler_pct_start",
        type=float,
        default=0.05,
        help="Fraction of total training steps used for the warmup/increase phase in OneCycleLR.",
    )
    parser.add_argument(
        "--scheduler_anneal_strategy",
        type=str,
        default="cos",
        choices=["cos", "linear"],
        help="Annealing strategy for OneCycleLR.",
    )
    parser.add_argument(
        "--scheduler_div_factor",
        type=float,
        default=10.0,
        help="Initial learning rate divisor for OneCycleLR.",
    )
    parser.add_argument(
        "--scheduler_final_div_factor",
        type=float,
        default=1000.0,
        help="Final learning rate divisor for OneCycleLR.",
    )

    parser.add_argument("--num_train_epochs", type=int, default=10)

    # save
    parser.add_argument(
        "--checkpointing_epochs",
        type=int,
        default=1,
        help=(
            "Save a checkpoint of the training state every X epochs. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
