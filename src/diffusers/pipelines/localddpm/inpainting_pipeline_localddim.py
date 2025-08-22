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


from typing import List, Optional, Tuple, Union

import torch

from ...utils.torch_utils import randn_tensor
from ...loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    PeftAdapterMixin,
)
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.set_random_steps import set_random_steps

import time


class InPaintLocalDDIMPipeline(
    DiffusionPipeline,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
    PeftAdapterMixin):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        clean_image: torch.Tensor = None,
        mask: torch.Tensor = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        args: Optional = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = InPaintLocalDDIMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)


        self.clean_image = clean_image.to(self.device)
        self.mask = mask.to(self.device)
        # self.mask = self.mask[:, 0, :, :].unsqueeze(1)
        # if True:
        #     self.mask = 1 - self.mask[0].repeat(batch_size, 1, 1, 1)
        timesteps = torch.tensor([len(self.scheduler.m_steps)-1]).to(self.device).repeat(len(self.mask))
        self.scheduler.set_variables(self.mask, timesteps)
        self.scheduler.mask_idx = self.scheduler.b_t.nonzero(as_tuple=True)
        image = self.scheduler.add_noise(self.clean_image, torch.randn_like(clean_image).to(self.device), timesteps)


        # set step values
        self.scheduler.set_timesteps(num_inference_steps)


        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)

        random_steps, m_steps, om_steps = set_random_steps(args)
        self.scheduler.random_steps = random_steps
        self.scheduler.m_steps = m_steps
        self.scheduler.om_steps = om_steps

        timestep = (
            torch.linspace(self.scheduler.num_mask_timesteps - 10, 0, num_inference_steps)
            .round()
            .to(torch.int64)
        )

        self.scheduler.timesteps = timestep
        self.scheduler.config.num_train_timesteps = self.scheduler.num_mask_timesteps
        # start_time = time.time()
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            t = t.repeat(len(self.mask)).to(self.mask.device)
            self.scheduler.set_variables(self.mask, t)
            self.scheduler.mask_idx = self.scheduler.b_t.nonzero(as_tuple=True)

            if not args.time_enc:
                if args.use_b_bar:
                    timesteps = self.scheduler.b_t_bar
                    if 'mult' in args.exp_name:
                        timesteps *= args.ddpm_num_steps
                else:
                    B, _, H, W = self.scheduler.b_t_bar.shape
                    beta_bar = 1 - self.scheduler.alphas_cumprod
                    beta_bar = torch.hstack((torch.tensor([0.0]), beta_bar[:self.scheduler.mask_steps])).to(
                        mask.device)
                    dist = torch.abs(self.scheduler.b_t_bar.view(-1, 1) - beta_bar[None, :])
                    idx_low = torch.argmin(dist, axis=1)

                    idx_high = torch.where(idx_low == 0, 1, idx_low)
                    idx_low = torch.where(idx_low == len(beta_bar) - 1, idx_low - 1, idx_low)

                    idx_high = torch.where((idx_low > 0) & (idx_low < len(beta_bar) - 1) & (
                            beta_bar[idx_low] > self.scheduler.b_t_bar.view(-1)), idx_low, idx_high)
                    idx_low = torch.where((idx_low > 0) & (idx_low < len(beta_bar) - 1) & (
                            beta_bar[idx_low] > self.scheduler.b_t_bar.view(-1)), idx_high - 1, idx_low)

                    idx_high = torch.where((idx_low > 0) & (idx_low < len(beta_bar) - 1) & (
                            beta_bar[idx_low] < self.scheduler.b_t_bar.view(-1)), idx_low + 1, idx_high)
                    low_val = torch.take(beta_bar, idx_low)
                    high_val = torch.take(beta_bar, idx_high)

                    interpolation_factor = (self.scheduler.b_t_bar.view(-1) - low_val) / (high_val - low_val)
                    interpolation_factor[high_val == low_val] = 0
                    timesteps = low_val + interpolation_factor
                    timesteps = timesteps.view(B, -1, H, W)
            else:
                timesteps = t
            model_output = self.unet(image, timesteps).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        # print(f"Inference time: {time.time() - start_time}")
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
