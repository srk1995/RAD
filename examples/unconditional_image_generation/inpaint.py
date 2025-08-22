# Copyright (C) 2024 Sora Kim
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import warnings

warnings.filterwarnings(action="ignore")
import argparse
import logging
import math
import os
from pathlib import Path
import accelerate

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo
from packaging import version
from peft import LoraConfig
from opencv_transforms import transforms
from diffusers.loaders.dataloaders import InpaintDataset, GivenDataset
from glob import glob
from diffusers.training_utils import EMAModel

import diffusers
from diffusers.models.unets.unet_2d_local import LocalUNet2DModel
from diffusers.schedulers.scheduling_localddpm import LocalDDPMScheduler
from diffusers.pipelines.localddpm.inpainting_pipeline_localddpm import InPaintLocalDDPMPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from diffusers.utils.set_random_steps import set_random_steps

from torch.utils.data import Subset, DataLoader
import cv2


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def save_model_card(
    repo_id: str,
    images: list = None,
    base_model: str = None,
    dataset_name: str = None,
    repo_folder: str = None,
):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "lora",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="xutongda/adm_ffhq_256x256",  # xutongda/adm_lsun_bedroom_256x256, xutongda/adm_imagenet_256x256_unconditional, xutongda/adm_ffhq_256x256
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="merkol/ffhq-256", # evanarlian/imagenet_1k_resized_256,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=500)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
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
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
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
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
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
        default="checkpoint-300000",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--exp_name", type=str, default="",
    )
    parser.add_argument(
        "--noise", type=str, default="perlin",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=2000)
    parser.add_argument("--ddpm_mask_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=100)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument(
        "--blur_sigma", type=int, default=0
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000
    )

    parser.add_argument(
        "--random_steps", action="store_true", default=False
    )

    parser.add_argument(
        "--multiple", action="store_true"
    )
    parser.add_argument(
        "--val_data_path", type=str, required=True, help=("path to your validation data"),
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args



def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    args.exp_name += f"Local-{args.dataset_name.split('/')[-1]}-{args.ddpm_num_steps}-{args.ddpm_mask_num_steps}"
    args.output_dir = f"ddpm-model-{args.dataset_name}-{args.resolution}/{args.exp_name}_lora_rank_{args.rank}"
    # args.output_dir += f"{args.rank}" if args.rank is not None else f"min_{}"
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

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
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    # Load scheduler, tokenizer and models.
    noise_scheduler = LocalDDPMScheduler(num_train_timesteps=args.ddpm_num_steps,
                                         num_mask_timesteps=args.ddpm_mask_num_steps,
                                         variance_type="learned_range"
                                         )

    mod = LocalUNet2DModel
    model = mod.from_pretrained(
        args.pretrained_model_name_or_path, revision=args.revision, variant=args.variant, low_cpu_mem_usage=False,
    )

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained_lora(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            import safetensors
            if args.use_ema:
                state_dict = safetensors.torch.load_file(os.path.join(input_dir, "unet_ema", "diffusion_pytorch_model.safetensors"), device="cpu")
                ema_model.load_state_dict(state_dict)
                ema_model.to(accelerator.device)
                del state_dict

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                # load_model = model.from_pretrained(input_dir, subfolder="unet")
                # model.register_to_config(**load_model.config)

                state_dict = safetensors.torch.load_file(
                    os.path.join(input_dir, "unet", "diffusion_pytorch_model.safetensors"), device="cpu")
                model.load_state_dict(state_dict)
                del state_dict

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    args.img_path = os.path.join(args.output_dir,
                             f"Inpainting images_FID_{os.path.basename(args.resume_from_checkpoint).split('-')[1]}")

    os.makedirs(args.img_path, exist_ok=True)

    def adaptive_rank_by_feature_dimension(layer, name):
        """
        This function assigns a rank based on the feature dimension (input or output channels).
        For convolution layers, use output channels.
        For attention layers (key, query, value), use the hidden size (input size).
        """
        base_rank = 4  # Minimum rank

        # Conv2d: Feature dimension = out_channels
        if isinstance(layer, torch.nn.Conv2d):
            feature_dim = layer.out_channels
            adaptive_rank_value = max(base_rank, feature_dim // 4)  # Adjust the divisor as needed
            return adaptive_rank_value

        # Linear: Handling attention's key, query, value
        elif isinstance(layer, torch.nn.Linear):
            # Check if it's an attention layer by name matching (assuming transformers-like structure)
            if any(k in name for k in ["to_k", "to_q", "to_v", "to_out.0"]):
                feature_dim = layer.out_features  # Hidden size for attention layers
                adaptive_rank_value = max(base_rank, feature_dim // 4)  # Adjust the divisor as needed
                return adaptive_rank_value

        # Default rank for other layers
        return base_rank

    if args.rank is None:
        # Get layer names and corresponding feature dimensions for UNet
        layer_names = [name for name, layer in model.named_modules() if
                       isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear))]
        layer_feature_dims = {name: layer for name, layer in model.named_modules() if
                              isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear))}

        # Create adaptive rank mapping for each layer based on feature dimension
        adaptive_layer_rank_mapping = {
            layer_name: adaptive_rank_by_feature_dimension(model.get_submodule(layer_name), layer_name) for layer_name in
            layer_names
        }

        # Print out the assigned ranks for verification
        print("Adaptive ranks assigned to each layer based on feature dimensions:")
        for layer_name, rank in adaptive_layer_rank_mapping.items():
            print(f"{layer_name}: Rank {rank}")

        # Apply LoRA to each layer with adaptive rank based on feature dimension
        lora_configs = {}
        for layer_name, rank in adaptive_layer_rank_mapping.items():
            # Configure LoRA for each layer with the corresponding adaptive rank
            lora_config = LoraConfig(
                r=rank,  # Adaptive rank for this specific layer
                lora_alpha=32,  # You can adjust alpha, which scales the LoRA weights
                lora_dropout=0.1,  # You can adjust dropout as necessary
                target_modules=[layer_name]  # Specify which layer to apply LoRA
            )

            # Store the config for each layer
            lora_configs[layer_name] = lora_config

    else:
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "conv_in", "conv_out", "conv1", "conv2"],
            # target_modules=[],
        )
    # freeze parameters of models to save more memory

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # Move unet, vae and text_encoder to device and cast to weight_dtype
    model.to(accelerator.device, dtype=weight_dtype)

    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=mod,
            model_config=model.config,
            lora_config=lora_config,
        )
    if args.use_ema:
        ema_model.to(accelerator.device)

    # Add adapter and make sure the trainable params are in float32.
    model.add_adapter(lora_config)


    # Prepare everything with our `accelerator`.
    model = accelerator.prepare(
        model
    )


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


# Checks if the accelerator has performed an optimization step behind the scenes
    for mask in ['thick', 'box','extreme']:
        save_path = f"{args.img_path}/{mask}"
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path + "/inpainted", exist_ok=True)
        os.makedirs(save_path + "/mask", exist_ok=True)
        os.makedirs(save_path + "/original", exist_ok=True)
        # Prepare everything with our `accelerator`.
        model = accelerator.prepare(model)


        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.

        noise_scheduler.blur_sigma = args.blur_sigma
        random_steps, m_steps, om_steps = set_random_steps(args)

        noise_scheduler.random_steps = random_steps
        noise_scheduler.m_steps = m_steps
        noise_scheduler.om_steps = om_steps


        unet = accelerator.unwrap_model(model)
        pipeline = InPaintLocalDDPMPipeline(
            unet=unet,
            scheduler=noise_scheduler,
        )


        val_datasets = GivenDataset(args.val_data_path, mask, transform=transforms.ToTensor())
        eval_loader = torch.utils.data.DataLoader(
            val_datasets, batch_size=args.train_batch_size, shuffle=False, num_workers=args.dataloader_num_workers,
            drop_last=True
        )

        count = 0
        for m, (img, mask) in enumerate(eval_loader):
            if count > args.num_samples:
                break
            if os.path.exists(f"{save_path}/inpainted/{(args.num_samples // accelerator.num_processes) * accelerator.process_index + count:04d}.png"):
                count += 1
                continue
            # Generate sample images for visual inspection

            if args.multiple:
                img = img[accelerator.process_index].unsqueeze(0).repeat(10, 1, 1, 1)
                mask = mask[accelerator.process_index].unsqueeze(0).repeat(10, 1, 1, 1)

            # run pipeline in inference (sample random noise and denoise)
            images = pipeline(
                # generator=generator,
                clean_image=img,
                mask=mask.cuda(),
                num_inference_steps=args.ddpm_num_inference_steps,
                output_type="np",
                args=args,
            ).images

            # denormalize the images and save to tensorboard
            images_processed = (images * 255).round().astype("uint8")
            imgs = (img / 2 + 0.5).clamp(0, 1)
            img_processed = (imgs.permute(0, 2, 3, 1).cpu().detach().numpy() * 255).round().astype("uint8")

            for i, (image, im) in enumerate(zip(images_processed, img_processed)):
                plt.imsave(
                    f"{save_path}/inpainted/{(args.num_samples // accelerator.num_processes) * accelerator.process_index + count:04d}.png",
                    image)
                plt.imsave(
                    f"{save_path}/mask/{(args.num_samples // accelerator.num_processes) * accelerator.process_index + count:04d}.png",
                    pipeline.mask[i, 0].cpu().detach().numpy())
                plt.imsave(
                    f"{save_path}/original/{(args.num_samples // accelerator.num_processes) * accelerator.process_index + count:04d}.png",
                    im)
                count += 1


if __name__ == "__main__":
    main()
