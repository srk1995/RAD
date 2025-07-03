import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
import datasets
import torch
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from opencv_transforms import transforms
import cv2
from tqdm.auto import tqdm

import diffusers
from diffusers.loaders.dataloaders import MaskDataset, InpaintDataset
# from diffusers.pipelines.localddpm.pipeline_localddpm import LocalDDPMPipeline
from diffusers.pipelines.localddpm.pipeline_localddim import LocalDDIMPipeline
from diffusers.pipelines.localddpm.inpainting_pipeline_localddim import InPaintLocalDDIMPipeline
from diffusers.pipelines.localddpm.inpainting_pipeline_localddpm import InPaintLocalDDPMPipeline
from diffusers.models.unets.unet_2d_local import LocalUNet2DModel
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_localddpm import LocalDDPMScheduler
from diffusers.schedulers.scheduling_localddim import LocalDDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from diffusers.utils.noise_gen import gen_mask_parallel, get_It, generate_perlin_noise_2d
from diffusers.utils.set_random_steps import set_random_steps
# from metrics.fid import masked_fid_score as fid_score
import matplotlib.pyplot as plt

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="merkol/ffhq-256",               ## "merkol/ffhq-256", "Andron00e/Places365-custom-train", "mattymchen/celeba-hq", "tglcourse/lsun_church_train"
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="/home/srk1995/pub/db/lsun_church",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default=f"ddpm-model-Places-256",
    #     help="The output directory where the model predictions and checkpoints will be written.",
    # )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=True,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=2, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--save_images_epochs", type=int, default=1, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=2000)
    parser.add_argument("--ddpm_mask_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=100)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument(
        "--blur_sigma", type=int, default=0
    )

    parser.add_argument(
        "--random_steps", action="store_true", default=False
    )

    parser.add_argument(
        "--use_b_bar", action="store_true", default=True
    )
    parser.add_argument(
        "--noise", type=str, default="perlin"  ## "blob"
    )
    parser.add_argument(
        "--exp_name", type=str, default="",
    )
    parser.add_argument(
        "--time_enc", action="store_true"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args


def main(args):
    args.exp_name += f"Local-{args.dataset_name.split('/')[-1]}-{args.ddpm_num_steps}-{args.ddpm_mask_num_steps}-{args.blur_sigma if args.blur_sigma is not None else 'random-blur'}{'-random' if args.random_steps else ''}{'-b_bar' if args.use_b_bar else ''}-{args.noise}"
    args.output_dir = f"ddpm-model-{args.dataset_name}-{args.resolution}/{args.exp_name}"
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), LocalUNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = model.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Initialize the model
    if args.time_enc:
        mod = UNet2DModel
    else:
        mod = LocalUNet2DModel
    if args.model_config_name_or_path is None:

        if 'imagenet' in args.dataset_name:
            model = mod(
                sample_size=args.resolution,
                in_channels=3,
                out_channels=3,
                layers_per_block=2,
                block_out_channels=(256, 256, 512, 512, 1024, 1024),
                norm_eps=1e-6,
                down_block_types=(
                    "ResnetDownsampleBlock2D",  # 256
                    "ResnetDownsampleBlock2D",  # 128
                    "ResnetDownsampleBlock2D",  # 64
                    "AttnDownBlock2D",  # 32
                    "AttnDownBlock2D",  # 16
                    "AttnDownBlock2D",  # 8
                ),
                up_block_types=(
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "ResnetUpsampleBlock2D",
                    "ResnetUpsampleBlock2D",
                    "ResnetUpsampleBlock2D",
                ),
                attention_head_dim=64,
                resnet_time_scale_shift="scale_shift",
                upsample_type="resnet",
                downsample_type="resnet",
            )
        elif 'lsun' in args.dataset_name:
            model = mod(
                sample_size=args.resolution,
                in_channels=3,
                out_channels=3,
                layers_per_block=2,
                block_out_channels=(256, 256, 512, 512, 1024, 1024),
                norm_eps=1e-6,
                down_block_types=(
                    "ResnetDownsampleBlock2D",  # 256
                    "ResnetDownsampleBlock2D",  # 128
                    "ResnetDownsampleBlock2D",  # 64
                    "AttnDownBlock2D",  # 32
                    "AttnDownBlock2D",  # 16
                    "AttnDownBlock2D",  # 8
                ),
                up_block_types=(
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "ResnetUpsampleBlock2D",
                    "ResnetUpsampleBlock2D",
                    "ResnetUpsampleBlock2D",
                ),
                attention_head_dim=64,
                dropout=0.1,
                resnet_time_scale_shift="scale_shift",
                upsample_type="resnet",
                downsample_type="resnet",
            )
        elif 'ffhq' in args.dataset_name:
            model = mod(
                sample_size=args.resolution,
                in_channels=3,
                out_channels=3,
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 256, 256, 256),
                flip_sin_to_cos=True,
                freq_shift=0,
                downsample_padding=1,
                norm_eps=1e-6,
                mid_block_scale_factor= 1.41421356237,
                down_block_types=(
                    "SkipDownBlock2D",  # 256
                    "SkipDownBlock2D",  # 128
                    "SkipDownBlock2D",  # 64
                    "SkipDownBlock2D",  # 32
                    "AttnSkipDownBlock2D",  # 16
                    "SkipDownBlock2D",  # 8
                    "SkipDownBlock2D",  # 4
                ),
                up_block_types=(
                    "SkipUpBlock2D",
                    "SkipUpBlock2D",
                    "AttnSkipUpBlock2D",
                    "SkipUpBlock2D",
                    "SkipUpBlock2D",
                    "SkipUpBlock2D",
                    "SkipUpBlock2D",
                ),
                attention_head_dim=8,
            )

    else:
        config = mod.load_config(args.model_config_name_or_path)
        model = mod.from_config(config)

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=LocalUNet2DModel,
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(LocalDDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = LocalDDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            num_mask_timesteps=args.ddpm_mask_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = LocalDDPMScheduler(num_train_timesteps=args.ddpm_num_steps,
                                             num_mask_timesteps=args.ddpm_mask_num_steps,
                                             beta_schedule=args.ddpm_beta_scheduler
                                             )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )

        tran_val_dataset = dataset.train_test_split(test_size=1000, seed=42)
        dataset = tran_val_dataset["train"]
        val_dataset = tran_val_dataset["test"]

    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets and DataLoaders creation.
    augmentations = transforms.Compose(
        [
            transforms.Resize(256, interpolation=cv2.INTER_LINEAR),
            transforms.CenterCrop(192) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.Resize(args.resolution, interpolation=cv2.INTER_NEAREST),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        if 'crop' in args.exp_name else
        [
            transforms.Resize(args.resolution, interpolation=cv2.INTER_LINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    def transform_images(examples):
        images = [augmentations(np.array(image.convert("RGB"))) for image in examples["image"]]
        return {"input": images}

    logger.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(transform_images)
    val_dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = "Local diffusion"
        accelerator.init_trackers(project_name=run, config=vars(args), init_kwargs={"rum_name": args.exp_name,})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    noise_scheduler.blur_sigma = args.blur_sigma


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["input"].to(weight_dtype)
            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            random_steps, m_steps, om_steps = set_random_steps(args)

            noise_scheduler.random_steps = random_steps
            noise_scheduler.m_steps = m_steps
            noise_scheduler.om_steps = om_steps

            if args.noise == "perlin":
                mask = generate_perlin_noise_2d((bsz, args.resolution, args.resolution), rand=True, sig=args.blur_sigma).to(clean_images.device)
            else:
                mask = gen_mask_parallel(noise.shape, k=args.blur_sigma * 4 + 1 if args.blur_sigma != None else 0,
                                         sig=args.blur_sigma, expectation=0.2 * (256 // args.resolution), device=clean_images.device)

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)


            noise_scheduler.set_variables(mask, timesteps)
            noise_scheduler.mask_idx = noise_scheduler.b_t.nonzero(as_tuple=True)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                if args.use_b_bar:
                    timesteps = noise_scheduler.b_t_bar
                else:
                    B, _, H, W = noise_scheduler.b_t_bar.shape
                    beta_bar = 1 - noise_scheduler.alphas_cumprod
                    beta_bar = torch.hstack((torch.tensor([0.0]), beta_bar[:noise_scheduler.mask_steps])).to(
                        clean_images.device)
                    dist = torch.abs(noise_scheduler.b_t_bar.view(-1, 1) - beta_bar[None, :])
                    idx_low = torch.argmin(dist, axis=1)

                    idx_high = torch.where(idx_low == 0, 1, idx_low)
                    idx_low = torch.where(idx_low == len(beta_bar) - 1, idx_low - 1, idx_low)

                    idx_high = torch.where((idx_low > 0) & (idx_low < len(beta_bar) - 1) & (
                            beta_bar[idx_low] > noise_scheduler.b_t_bar.view(-1)), idx_low, idx_high)
                    idx_low = torch.where((idx_low > 0) & (idx_low < len(beta_bar) - 1) & (
                            beta_bar[idx_low] > noise_scheduler.b_t_bar.view(-1)), idx_high - 1, idx_low)

                    idx_high = torch.where((idx_low > 0) & (idx_low < len(beta_bar) - 1) & (
                            beta_bar[idx_low] < noise_scheduler.b_t_bar.view(-1)), idx_low + 1, idx_high)

                    low_val = torch.take(beta_bar, idx_low)
                    high_val = torch.take(beta_bar, idx_high)
                    interpolation_factor = (noise_scheduler.b_t_bar.view(-1) - low_val) / (high_val - low_val)
                    interpolation_factor[high_val == low_val] = 0
                    timesteps = idx_low + interpolation_factor
                    timesteps = timesteps.view(B, -1, H, W)
                model_output = model(noisy_images, timesteps).sample

                if args.prediction_type == "epsilon":
                    # loss = torch.where(noise_scheduler.b_t == 0, torch.zeros_like(model_output),
                    #                    F.mse_loss(model_output.float(), noise.float()))
                    # model_output = get_It(noise_scheduler.mask_idx, model_output)
                    # noise = get_It(noise_scheduler.mask_idx, noise)
                    loss = F.mse_loss(model_output.float(), noise.float())  # this could have different weights!
                    loss = loss.mean()
                elif args.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    # use SNR weighting from distillation paper
                    loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
                    loss = loss.mean()
                elif args.prediction_type =="adm":
                    model_output, predicted_variance = torch.split(model_output, 3, dim=1)
                    mse_loss = F.mse_loss(model_output.float(), noise.float())
                    model_output = torch.cat([model_output.detach(), predicted_variance], dim=1)
                    vlb_loss = noise_scheduler.get_vlb_loss(clean_images, noisy_images, model_output, timesteps)
                    loss = mse_loss + vlb_loss

                else:
                    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                if global_step % args.checkpointing_steps == 0:

                    # Handle the repository creation
                    args.img_path = os.path.join(args.output_dir, "generated_images", f"{global_step}")
                    os.makedirs(args.img_path, exist_ok=True)

                    unet = accelerator.unwrap_model(model)
                    pipeline = InPaintLocalDDIMPipeline(
                        unet=unet,
                        scheduler=noise_scheduler
                    )

                    if args.use_ema:
                        ema_model.store(unet.parameters())
                        ema_model.copy_to(unet.parameters())

                    # Generate samples
                    mask_path = [
                        "datasets/gt_keep_masks/extreme",
                        "datasets/gt_keep_masks/box",
                        "datasets/gt_keep_masks/thick"
                    ]
                    for f in mask_path:
                        save_path = f"{args.img_path}/{f.split('/')[-1]}"
                        os.makedirs(save_path, exist_ok=True)
                        os.makedirs(save_path + "/inpainted", exist_ok=True)
                        os.makedirs(save_path + "/mask", exist_ok=True)
                        os.makedirs(save_path + "/original", exist_ok=True)
                        eval_dataset = InpaintDataset(val_dataset, f, transform=transforms.ToTensor())
                        eval_loader = torch.utils.data.DataLoader(
                            eval_dataset, batch_size=args.train_batch_size, shuffle=True,
                            num_workers=args.dataloader_num_workers, drop_last=True
                        )
                        for m, (img, mask) in enumerate(eval_loader):
                            images = img.to(weight_dtype)
                            if m == 0:
                                images = pipeline(
                                    # generator=generator,
                                    clean_image=images,
                                    mask=mask,
                                    num_inference_steps=args.ddpm_num_inference_steps,
                                    output_type="np",
                                    args=args,
                                ).images

                                images_processed = (images * 255).round().astype("uint8")
                                img_processed = (img / 2 + 0.5).clamp(0, 1)
                                img_processed = (img_processed.permute(0, 2, 3,
                                                              1).cpu().detach().numpy() * 255).round().astype(
                                    "uint8")
                                for i, image in enumerate(images_processed):
                                    plt.imsave(
                                        f"{save_path}/inpainted/{args.eval_batch_size * accelerator.process_index + i:04d}.png",
                                        image)
                                    plt.imsave(
                                        f"{save_path}/mask/{args.eval_batch_size * accelerator.process_index + i:04d}.png",
                                        pipeline.mask[i, 0].cpu().detach().numpy())
                                    plt.imsave(
                                        f"{save_path}/original/{args.eval_batch_size * accelerator.process_index + i:04d}.png",
                                        img_processed[i])

                            else:
                                break

                    if args.use_ema:
                        ema_model.restore(unet.parameters())

                    # denormalize the images and save to tensorboard
                    # images_processed = (images * 255).round().astype("uint8")

                    # if args.logger == "tensorboard":
                    #     if is_accelerate_version(">=", "0.17.0.dev0"):
                    #         tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    #     else:
                    #         tracker = accelerator.get_tracker("tensorboard")
                    #     tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                    # elif args.logger == "wandb":
                    #     # # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                    #     # accelerator.get_tracker("wandb").log(
                    #     #     {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                    #     #     step=global_step,
                    #     # )
                    #     for i, image in enumerate(images_processed):
                    #         plt.imsave(
                    #             f"{args.img_path}/sample_{args.eval_batch_size * accelerator.process_index + i}.png",
                    #             image)
                    #         plt.imsave(
                    #             f"{args.img_path}/mask_{args.eval_batch_size * accelerator.process_index + i:}.png",
                    #             pipeline.mask[i, 0].cpu().numpy())

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = InPaintLocalDDIMPipeline(
                    unet=unet,
                    scheduler=LocalDDPMScheduler(num_train_timesteps=args.ddpm_num_steps,
                                                 num_mask_timesteps=args.ddpm_mask_num_steps,
                                                 beta_schedule=args.ddpm_beta_schedule)
                    ,
                )

                pipeline.save_pretrained(args.output_dir)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
