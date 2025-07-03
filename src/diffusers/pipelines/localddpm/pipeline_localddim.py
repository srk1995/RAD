# Copyright 2024 The HuggingFace Team. All rights reserved.
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


from typing import List, Optional, Tuple, Union

import torch

from ...utils.torch_utils import randn_tensor
from ...loaders import FromSingleFileMixin, IPAdapterMixin, PeftAdapterMixin
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.noise_gen import gen_mask_parallel, generate_perlin_noise_2d
from diffusers.utils.set_random_steps import set_random_steps
from ...schedulers.scheduling_localddim import LocalDDIMScheduler


class LocalDDIMPipeline(DiffusionPipeline, PeftAdapterMixin):
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
        scheduler = LocalDDIMScheduler.from_config(scheduler.config)
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        args:   Optional = None,
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
        >>> pipe = LocalDDIMPipeline.from_pretrained("google/ddpm-cat-256")

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

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = torch.randn(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = torch.randn(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        if args.noise == "perlin":
            self.mask = generate_perlin_noise_2d((batch_size, args.resolution, args.resolution), rand=True, sig=0).to(image.device)
        else:
            self.mask = gen_mask_parallel(image.shape, k=args.blur_sigma * 4 + 1 if args.blur_sigma != None else 0,
                                     sig=0, expectation=0.2 * (256 // args.resolution),
                                     device=image.device)

        # self.mask = torch.ones((batch_size, 1, args.resolution, args.resolution), device=image.device)

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)

        random_steps, m_steps, om_steps = set_random_steps(args)
        self.scheduler.random_steps = random_steps
        self.scheduler.m_steps = m_steps
        self.scheduler.om_steps = om_steps


        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            t = t.repeat(len(self.mask)).to(self.mask.device)
            self.scheduler.set_variables(self.mask, t)
            self.scheduler.mask_idx = self.scheduler.b_t.nonzero(as_tuple=True)

            if args.use_b_bar:
                timesteps = self.scheduler.b_t_bar
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

            model_output = self.unet(image, timesteps).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
