# Copyright 2024 UC Berkeley Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.torch_utils import randn_tensor
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


@dataclass
class LocalDDPMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


def betas_for_alpha_bar(
        num_diffusion_timesteps,
        max_beta=0.999,
        alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


# Copied from diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr
def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)


    Args:
        betas (`torch.FloatTensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.FloatTensor`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


class LocalDDPMScheduler(SchedulerMixin, ConfigMixin):
    """
    `LocalDDPMScheduler` explores the connections between denoising score matching and Langevin dynamics sampling.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            An array of betas to pass directly to the constructor without using `beta_start` and `beta_end`.
        variance_type (`str`, defaults to `"fixed_small"`):
            Clip the variance when adding noise to the denoised sample. Choose from `fixed_small`, `fixed_small_log`,
            `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample for numerical stability.
        clip_sample_range (`float`, defaults to 1.0):
            The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        timestep_spacing (`str`, defaults to `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int = 2000,
            num_mask_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            beta_schedule: str = "linear",
            trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
            variance_type: str = "fixed_small",
            clip_sample: bool = True,
            prediction_type: str = "epsilon",
            thresholding: bool = False,
            dynamic_thresholding_ratio: float = 0.995,
            clip_sample_range: float = 1.0,
            sample_max_value: float = 1.0,
            timestep_spacing: str = "leading",
            steps_offset: int = 0,
            rescale_betas_zero_snr: int = False,
    ):
        self.mask_steps = num_mask_timesteps

        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas_out_mask = torch.linspace(beta_start, beta_end, num_train_timesteps - num_mask_timesteps,
                                                 dtype=torch.float32)
            self.betas_mask = torch.linspace(beta_start, beta_end, num_mask_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas_out_mask = torch.linspace(beta_start ** 0.5, beta_end ** 0.5,
                                                 num_train_timesteps - num_mask_timesteps, dtype=torch.float32) ** 2
            self.betas_mask = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_mask_timesteps,
                                             dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas_out_mask = betas_for_alpha_bar(num_train_timesteps - num_mask_timesteps)
            self.betas_mask = betas_for_alpha_bar(num_mask_timesteps)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas_out_mask = torch.linspace(-6, 6, num_train_timesteps - num_mask_timesteps)
            betas_mask = torch.linspace(-6, 6, num_mask_timesteps)

            self.betas_out_mask = torch.sigmoid(betas_out_mask) * (beta_end - beta_start) + beta_start
            self.betas_mask = torch.sigmoid(betas_mask) * (beta_end - beta_start) + beta_start

        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        # Rescale for zero SNR
        if rescale_betas_zero_snr:
            self.betas_out_mask = rescale_zero_terminal_snr(self.betas_out_mask)
            self.betas_mask = rescale_zero_terminal_snr(self.betas_mask)

        self.betas = torch.cat([self.betas_mask, self.betas_out_mask])

        self.alphas_mask = 1.0 - self.betas_mask
        self.alphas_out_mask = 1.0 - self.betas_out_mask
        self.alphas = 1.0 - self.betas

        self.alphas_cumprod_mask = torch.cumprod(self.alphas_mask, dim=0)
        self.alphas_cumprod_out_mask = torch.cumprod(self.alphas_out_mask, dim=0)

        self.alphas_cumprod = torch.cat([self.alphas_cumprod_mask, self.alphas_cumprod_out_mask])
        # self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.custom_timesteps = False
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

        self.variance_type = variance_type
        self.nu = 1 - self.alphas_cumprod[-1]
        self.gamma = torch.log(1 - self.betas) / torch.log(1 - self.nu)

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        return sample

    def set_variables(self, mask, t):
        self.mask = mask
        self.a_t_prepare, self.a_t, self.a_t_bar, self.a_t_bar_prev = torch.zeros_like(
            self.mask), torch.zeros_like(
            self.mask), torch.zeros_like(
            self.mask), torch.zeros_like(
            self.mask)

        t = torch.where(self.random_steps[None, :].to(t.device) == t[:, None])[1]
        mask_idx, out_mask_idx = torch.where(t < self.mask_steps)[0], torch.where(t >= self.mask_steps)[0]

        self.a_t_prepare[mask_idx] = (1 - self.nu * mask[mask_idx])
        self.a_t_prepare[out_mask_idx] = ((1 - self.nu) / (1 - self.nu * mask[out_mask_idx]))
        # self.a_t_prepare = th.clip(self.a_t_prepare, 0, 1)
        self.a_t = self.a_t_prepare ** _extract_into_tensor(self.gamma.to(t.device), t, mask.shape)

        self.b_t = 1 - self.a_t

        self.a_t_bar[mask_idx] = (1 - self.nu * mask[mask_idx]) ** (
                torch.log(_extract_into_tensor(self.alphas_cumprod.to(t.device), t[mask_idx],
                                               mask[mask_idx].shape)) / torch.log(
            1 - self.nu))
        self.a_t_bar[out_mask_idx] = ((1 - self.nu) / (1 - self.nu * mask[out_mask_idx])) ** (
                torch.log(_extract_into_tensor(self.alphas_cumprod.to(t.device), t[out_mask_idx],
                                               mask[out_mask_idx].shape)) / torch.log(
            1 - self.nu)) * (1 - self.nu * mask[out_mask_idx])

        self.b_t_bar = 1 - self.a_t_bar

    def set_timesteps(
            self,
            num_inference_steps: Optional[int] = None,
            device: Union[str, torch.device] = None,
            timesteps: Optional[List[int]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps is used. If `timesteps` is passed,
                `num_inference_steps` must be `None`.

        """
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")

        if timesteps is not None:
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("`custom_timesteps` must be in descending order.")

            if timesteps[0] >= self.config.num_train_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps}."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.custom_timesteps = True
        else:
            if num_inference_steps > self.config.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.config.num_train_timesteps} timesteps."
                )

            self.num_inference_steps = num_inference_steps
            self.custom_timesteps = False

            # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
            if self.config.timestep_spacing == "linspace":
                timesteps = (
                    np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                    .round()[::-1]
                    .copy()
                    .astype(np.int64)
                )
            elif self.config.timestep_spacing == "leading":
                step_ratio = self.config.num_train_timesteps // self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
                timesteps += self.config.steps_offset
            elif self.config.timestep_spacing == "trailing":
                step_ratio = self.config.num_train_timesteps / self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
                timesteps -= 1
            else:
                raise ValueError(
                    f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                )

        self.timesteps = torch.from_numpy(timesteps).to(device)

    def _get_variance(self, variance, current_beta_t, predicted_variance=None, variance_type=None):

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        if variance_type is None:
            variance_type = self.config.variance_type

        # hacks - were probably added for training stability
        if variance_type == "fixed_small":
            variance = variance
        # for rl-diffuser https://arxiv.org/abs/2205.09991
        elif variance_type == "fixed_small_log":
            variance = torch.log(variance)
            variance = torch.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            variance = current_beta_t
        elif variance_type == "fixed_large_log":
            # Glide max_log
            variance = torch.log(current_beta_t)
        elif variance_type == "learned":
            return predicted_variance
        elif variance_type == "learned_range":
            min_log = torch.log(variance)
            max_log = torch.log(current_beta_t.clamp(min=1e-20))
            frac = (predicted_variance + 1) / 2
            log_variance = frac * max_log + (1 - frac) * min_log
            # variance = frac * max_var + (1 - frac) * min_var
            variance = torch.exp(log_variance)
        return variance

    def _threshold_sample(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[LocalDDPMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas

        alpha_prod_t = self.a_t_bar
        if prev_t[0] < 0:
            alpha_prod_t_prev = torch.ones_like(self.a_t_bar)
        elif (t[0] >= self.mask_steps) and (prev_t[0] < self.mask_steps):
            alpha_prod_t_prev = 1 - self.nu * self.mask
        else:
            self.set_variables(self.mask, prev_t)
            alpha_prod_t_prev = self.a_t_bar
        self.set_variables(self.mask, t)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev.clamp(min=1e-20)
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / (alpha_prod_t ** (0.5)).clamp(min=1e-20)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t.clamp(min=1e-20)
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t.clamp(min=1e-20)

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        ## Fixed µ_t when beta_prod_t is zero
        pred_prev_sample = torch.where(beta_prod_t == 0, sample, pred_prev_sample)

        # 6. Add noise
        variance = current_beta_t * beta_prod_t_prev / beta_prod_t.clamp(min=1e-20)
        if t[0] > 0:
            device = model_output.device
            variance_noise = torch.randn(
                model_output.shape, device=device, dtype=model_output.dtype
            )
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(variance, current_beta_t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(variance, current_beta_t, predicted_variance=predicted_variance)
                # variance = torch.exp(0.5 * variance) * variance_noise
                variance = (variance ** 0.5) * variance_noise
            else:
                variance = (self._get_variance(variance, current_beta_t, predicted_variance=predicted_variance) ** 0.5) * variance_noise
        else:
            variance = torch.zeros_like(pred_prev_sample)

        # zero variance when beta_prod_t is zero
        variance = torch.where(current_beta_t == 0, torch.zeros_like(variance), variance)
        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample,)

        return LocalDDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        noisy_samples = self.a_t_bar.to(dtype=original_samples.dtype) ** 0.5 * original_samples + self.b_t_bar.to(
            dtype=original_samples.dtype) ** 0.5 * noise
        return noisy_samples

    def get_velocity(
            self, sample: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as sample
        velocity = self.a_t_bar.to(dtype=sample.dtype) ** 0.5 * noise - self.b_t_bar.to(
            dtype=sample.dtype) ** 0.5 * sample
        return velocity

    def __len__(self):
        return self.config.num_train_timesteps

    def previous_timestep(self, timestep):
        if self.custom_timesteps:
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            if index == self.timesteps.shape[0] - 1:
                prev_t = torch.tensor(-1)
            else:
                prev_t = self.timesteps[index + 1]
        else:
            num_inference_steps = (
                self.num_inference_steps if self.num_inference_steps else self.config.num_train_timesteps
            )
            prev_t = timestep - self.config.num_train_timesteps // num_inference_steps

        return prev_t

    def get_vlb_terms(self, x_start, x_t, model_output, timestep):
        """
        Compute the variational lower bound (VLB) for the current timestep.
        """
        t = timestep
        ## 0-th timestep to 1st timestep
        t = torch.where((t==0) | (t==self.mask_steps), t+1, t)
        prev_t = self.previous_timestep(t)

        model_output, predicted_variance = torch.split(model_output, x_t.shape[1], dim=1)


        # 1. compute alphas, betas
        alpha_prod_t = self.a_t_bar
        if prev_t[0] < 0:
            alpha_prod_t_prev = torch.ones_like(self.a_t_bar)
        elif (t[0] >= self.mask_steps) and (prev_t[0] < self.mask_steps):
            alpha_prod_t_prev = 1 - self.nu * self.mask
        else:
            self.set_variables(self.mask, prev_t)
            alpha_prod_t_prev = self.a_t_bar
        self.set_variables(self.mask, t)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev.clamp(min=1e-20)
        current_beta_t = 1 - current_alpha_t
        self.current_beta_t = current_beta_t
        self.beta_prod_t_prev = beta_prod_t_prev

        x_0_t = (x_t - beta_prod_t ** (0.5) * model_output) / (alpha_prod_t ** (0.5)).clamp(min=1e-20)

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t.clamp(min=1e-20)
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t.clamp(min=1e-20)

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        true_mean = pred_original_sample_coeff * x_start + current_sample_coeff * x_t
        true_variance = current_beta_t * beta_prod_t_prev / beta_prod_t.clamp(min=1e-20)
        true_variance = true_variance.repeat(1, 3, 1, 1)

        beta_t = self.current_beta_t.repeat(1, 3, 1, 1)
        true_mean[beta_t == 0] = x_t[beta_t == 0]
        # true_variance[beta_t == 0] = 0.0
        true_log_variance = torch.log(true_variance.clamp(min=1e-20))


        model_mean = pred_original_sample_coeff * x_0_t + current_sample_coeff * x_t
        model_variance = self._get_variance(true_variance, current_beta_t, predicted_variance=predicted_variance)
        # model_variance = model_variance.repeat(1, 3, 1, 1)

        model_mean[beta_t == 0] = x_t[beta_t == 0]
        # model_log_variance[beta_t == 0] = 0.0
        model_log_variance = torch.log(model_variance.clamp(min=1e-20))

        return true_mean, true_log_variance, model_mean, model_log_variance

    def get_vlb_loss(self, x_start, x_t, model_output, timestep):
        """
        Compute the variational lower bound (VLB) for the current timestep.
        """
        true_mean, true_log_var, model_mean, model_log_var = self.get_vlb_terms(x_start, x_t, model_output, timestep)

        kl = 0.5 * (-1 + model_log_var - true_log_var + torch.exp(true_log_var - model_log_var) + (
                    (true_mean - model_mean) ** 2 * torch.exp(-model_log_var)))
        kl = torch.where(self.current_beta_t !=0, kl, 0)
        # kl = torch.where(self.beta_prod_t_prev != 0, kl, 0)
        kl = kl.mean(dim=(1, 2, 3)) / np.log(2)


        ## discretized gaussian log likelihood
        decoder_nll = -self.discretized_gaussian_log_likelihood(x_start, model_mean, 0.5 * model_log_var)
        decoder_nll = torch.where(self.current_beta_t !=0, decoder_nll, 0)
        decoder_nll = decoder_nll.mean(dim=(1, 2, 3)) / np.log(2.0)
        vlb = torch.where((timestep == 0) | (timestep == self.mask_steps), decoder_nll, kl)

        vlb = torch.where(vlb.isinf(), torch.zeros_like(vlb), vlb)
        return vlb.mean()

    def discretized_gaussian_log_likelihood(self, x, means, log_scales):
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.

        :param x: the target images. It is assumed that this was uint8 values,
                  rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
        assert x.shape == means.shape == log_scales.shape
        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
        )
        assert log_probs.shape == x.shape
        return log_probs

    def approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal.
        """
        return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

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