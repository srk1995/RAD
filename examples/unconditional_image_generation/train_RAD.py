#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import warnings

warnings.filterwarnings(action="ignore")
import accelerate
import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from opencv_transforms import transforms
from tqdm.auto import tqdm
from diffusers.loaders.dataloaders import InpaintDataset
from diffusers.training_utils import EMAModel

import diffusers
from diffusers import UNet2DModel
from diffusers.models.unets.unet_2d_local import LocalUNet2DModel
from diffusers.schedulers.scheduling_localddpm import LocalDDPMScheduler
from diffusers.pipelines.localddpm.pipeline_localddpm import LocalDDPMPipeline
from diffusers.pipelines.localddpm.inpainting_pipeline_localddpm import InPaintLocalDDPMPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from diffusers.utils.noise_gen import gen_mask_parallel, generate_perlin_noise_2d
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
        "--random_steps", action="store_true", default=False
    )

    parser.add_argument(
        "--time_enc", action="store_true"
    )
    parser.add_argument(
        "--train_mask", type=str, default=None,
    )
    parser.add_argument(
        "--use_ema",
        default=True,
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
    # Load scheduler, tokenizer and modaels.
    # noise_scheduler = LocalDDPMScheduler.from_pretrained(args.pretrained_model_name_or_path)
    noise_scheduler = LocalDDPMScheduler(num_train_timesteps=args.ddpm_num_steps,
                                         num_mask_timesteps=args.ddpm_mask_num_steps,
                                         variance_type="learned_range"
                                         )

    if args.time_enc:
        mod = UNet2DModel
    else:
        mod = LocalUNet2DModel
    model = mod.from_pretrained(
        args.pretrained_model_name_or_path, revision=args.revision, variant=args.variant, low_cpu_mem_usage=False,
    )

    def adaptive_rank_by_feature_dimension(layer, name):
        """
        This function assigns a rank based on the feature dimension (input or output channels).
        For convolution layers, use output channels.
        For attention layers (key, query, value), use the hidden size (input size).
        """
        base_rank = 16  # Minimum rank

        # Conv2d: Feature dimension = out_channels
        if isinstance(layer, torch.nn.Conv2d):
            feature_dim = layer.out_channels
            adaptive_rank_value = max(base_rank, feature_dim // 2)  # Adjust the divisor as needed
            return adaptive_rank_value

        # Linear: Handling attention's key, query, value
        elif isinstance(layer, torch.nn.Linear):
            # Check if it's an attention layer by name matching (assuming transformers-like structure)
            if any(k in name for k in ["to_k", "to_q", "to_v", "to_out.0"]):
                feature_dim = layer.out_features  # Hidden size for attention layers
                adaptive_rank_value = max(base_rank, feature_dim // 2)  # Adjust the divisor as needed
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

    # Freeze the unet parameters before adding adapters
    for param in model.parameters():
        param.requires_grad_(False)



    # Move unet, vae and text_encoder to device and cast to weight_dtype
    model.to(accelerator.device, dtype=weight_dtype)

    # Add adapter and make sure the trainable params are in float32.
    model.add_adapter(lora_config)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(model, dtype=torch.float32)

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

    lora_layers = filter(lambda p: p.requires_grad, model.parameters())

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # Get parameters contain string "time" in their name
    lora_layers = list(lora_layers)

    time_params = [param for name, param in model.named_parameters() if 'time' in name]
    # Freeze the unet parameters before adding adapters
    for param in time_params:
        param.requires_grad_(True)

    params = [
            {'params': time_params,},
            {'params': lora_layers, }
        ]
    optimizer = optimizer_cls(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    params = lora_layers + time_params

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



    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.

    # Preprocessing the datasets.
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

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [augmentations(image) for image in images]
        return examples

    def transform_images(examples):
        images = [augmentations(np.array(image.convert("RGB"))) for image in examples["image"]]
        return {"input": images}

    dataset.set_transform(transform_images)
    val_dataset.set_transform(transform_images)
    if args.train_mask is not None:
        f = args.train_mask
        eval_dataset = InpaintDataset(dataset, f, transform=transforms.ToTensor())
        train_dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers,
            drop_last=True
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = "Local diffusion LoRA"
        accelerator.init_trackers(project_name=run, config=vars(args), init_kwargs={"rum_name": args.exp_name,})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

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
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            if args.train_mask is not None:
                latents = batch[0]
                mask = batch[1]
            else:
                latents = batch["input"].to(dtype=weight_dtype)
                mask = generate_perlin_noise_2d((latents.shape[0], args.resolution, args.resolution), rand=True,
                                                sig=args.blur_sigma).to(latents.device)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            if args.noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += args.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                )

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timestep = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timestep = timestep.long()

            random_steps, m_steps, om_steps = set_random_steps(args)

            noise_scheduler.random_steps = random_steps
            noise_scheduler.m_steps = m_steps
            noise_scheduler.om_steps = om_steps


            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)

            noise_scheduler.set_variables(mask, timestep)
            noise_scheduler.mask_idx = noise_scheduler.b_t.nonzero(as_tuple=True)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timestep)

            # Predict the noise residual
            with (accelerator.accumulate(model)):
                B, _, H, W = noise_scheduler.b_t_bar.shape
                beta_bar = 1 - noise_scheduler.alphas_cumprod
                beta_bar = torch.hstack((torch.tensor([0.0]), beta_bar[:noise_scheduler.mask_steps])).to(
                    latents.device)
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


                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_output = model(noisy_latents, timesteps, return_dict=False)[0]
                if 'adm' in args.pretrained_model_name_or_path:
                    model_output, predicted_variance = torch.split(model_output, 3, dim=1)
                    mse_loss = F.mse_loss(model_output.float(), noise.float())
                    frozen_out = torch.cat([model_output.detach(), predicted_variance], dim=1)
                    vlb_loss = noise_scheduler.get_vlb_loss(latents, noisy_latents, frozen_out, timestep)
                    loss = mse_loss + 1e-3 * vlb_loss.mean()

                elif args.snr_gamma is None:
                    loss = F.mse_loss(model_output.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_output.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
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
                        if args.use_ema:
                            ema_model.store(model.parameters())
                            ema_model.copy_to(model.parameters())

                        accelerator.save_state(save_path)

                        unwrapped_unet = unwrap_model(model)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )


                        InPaintLocalDDPMPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                        if args.use_ema:
                            ema_model.restore(model.parameters())

                        logger.info(f"Saved state to {save_path}")


            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = model.to(torch.float32)

        unwrapped_unet = unwrap_model(model)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        LocalDDPMPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )


        # Final inference
        # Load previous pipeline

    accelerator.end_training()


if __name__ == "__main__":
    main()
