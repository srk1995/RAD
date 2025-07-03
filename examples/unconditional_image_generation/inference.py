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
# from build.lib.diffusers import DDPMPipeline
from diffusers import DDPMPipeline
from diffusers import DDIMPipeline
# from diffusers.pipelines.localddpm.pipeline_localddpm import LocalDDPMPipeline as Pipeline
# from diffusers.pipelines.localddpm.pipeline_localddim import LocalDDIMPipeline as Pipeline
from diffusers.pipelines.ddim import DDIMPipeline as Pipeline
# from diffusers.pipelines.localddpm.inpainting_pipeline_localddim import InPaintLocalDDIMPipeline as Pipeline
# from diffusers.models.unets.unet_2d_local import LocalUNet2DModel
from diffusers.models.unets.unet_2d import UNet2DModel
# from diffusers.schedulers.scheduling_localddpm import LocalDDPMScheduler as Scheduler
# from diffusers.schedulers.scheduling_localddim import LocalDDIMScheduler as Scheduler
# from diffusers.schedulers.scheduling_ddim import DDIMScheduler as Scheduler
from diffusers.schedulers.scheduling_sde_ve import ScoreSdeVeScheduler as Scheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

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
        default="mattymchen/celeba-hq",               ## "Andron00e/Places365-custom-train", "mattymchen/celeba-hq", "tglcourse/lsun_church_train" , "merkol/ffhq-256"
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
        default="google/ddpm-celebahq-256",  # xutongda/adm_lsun_bedroom_256x256, xutongda/adm_imagenet_256x256_unconditional, xutongda/adm_ffhq_256x256
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
        "--train_batch_size", type=int, default=64, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--num_samples", type=int, default=2000, help="The number of images to generate for evaluation."
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
    parser.add_argument("--num_epochs", type=int, default=2000)
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
        default=5e-5,
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
        default="",
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
        "--noise", type=str, default="perlin"
    )
    parser.add_argument(
        "--exp_name", type=str, default="",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args


def main(args):
    args.exp_name += f"Local-{args.dataset_name.split('/')[-1]}-{args.ddpm_num_steps}-{args.ddpm_mask_num_steps}"
    args.output_dir = f"ddpm-model-{args.dataset_name}-{args.resolution}/{args.exp_name}"
    args.img_path = os.path.join(args.output_dir,
                                 f"FID images")
    os.makedirs(args.img_path, exist_ok=True)

    pipeline = DDIMPipeline.from_pretrained(args.model_config_name_or_path).to("cuda")

    # Generate samples
    for iter in range(args.num_samples // 16 + 1):
        # Generate sample images for visual inspection

        # run pipeline in inference (sample random noise and denoise)
        images = pipeline(
            # generator=generator,
            batch_size=1,
            num_inference_steps=args.ddpm_num_inference_steps,
            output_type="np",
            # args=args,
        ).images

        # denormalize the images and save to tensorboard
        images_processed = (images * 255).round().astype("uint8")

        for i, image in enumerate(images_processed):
            plt.imsave(
                f"{args.img_path}/sample_{(args.num_samples ) + iter * args.eval_batch_size + i:04d}.png",
                image)
            # plt.imsave(
            #     f"{args.img_path}/mask/mask_{(args.num_samples // accelerator.num_processes) * accelerator.process_index + iter * args.eval_batch_size + i:04d}.png",
            #     pipeline.mask[i, 0].cpu().detach().numpy())



if __name__ == "__main__":
    args = parse_args()
    main(args)
