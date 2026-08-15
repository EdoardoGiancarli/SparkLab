"""
Script for sources shadowgram/parameters sampling at model inference.

References:
    * The Principles of Diffusion Models, Lai et al. (pre-release book, 2026): https://arxiv.org/abs/2510.21890
    * Phil Wang's `denoising-diffusion-pytorch`: https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main
    * Annotated Diffusion, Niels Rogge and Kashif Rasul (with refs inside, nicely detailed)
      https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=51d9a24c
    * APXML website (general info gathering): https://apxml.com/courses/advanced-diffusion-architectures
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Tensor

from .modules import exists


__all__ = []


@dataclass
class DiffusionRegistry:
    """Container with pre-computed quantities for the diffusion process."""
    # forward diffusion
    sqrt_alphas_cumprod: Tensor
    sqrt_one_minus_alphas_cumprod: Tensor
    # model sampling
    alphas_cumprod_prev: Tensor
    posterior_sigma: Tensor


def get_diff_registry(betas: Tensor) -> DiffusionRegistry:
    """Computes the quantities needed for the diffusion process."""
    # double precision to avoid alpha-drift
    betas = betas.to(torch.float64)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # forward diffusion q(x_t | x_{t-1})
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    # sampling (sigmas from DDIM https://arxiv.org/pdf/2010.02502)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    posterior_sigma = torch.sqrt(
        (1.0 - alphas_cumprod_prev) * (1.0 - alphas_cumprod / alphas_cumprod_prev),
    ) / sqrt_one_minus_alphas_cumprod
    # define table with pre-computed vals
    registry = DiffusionRegistry(
        sqrt_alphas_cumprod.float(),
        sqrt_one_minus_alphas_cumprod.float(),
        alphas_cumprod_prev.float(),
        posterior_sigma.float(),
    )
    return registry


def convert_model_output(
    x: Tensor,
    t: Tensor,
    pred_type: str,
    to: str,
    diff_registry: DiffusionRegistry,
) -> Tensor:
    """
    Converts model prediction data to selected domain.

    Args:
        ...
    
    Returns:
        out (Tensor): Model prediction in selected domain.
    """
    pass


def extract(vals: Tensor, t: Tensor, x_dims: int) -> Tensor:
    """
    Extracts the appropriate `vals_t` value for a batch of indices, for
    a generic signal (e.g., images, latent vectors, video, etc.).
    """
    b, *_ = t.shape
    out = vals.gather(-1, t)
    return out.reshape(b, *((1,) * (x_dims - 1)))


class Sampler(ABC, nn.Module):
    """
    Parent class for sampling at model inference.

    Args:
        betas (Tensor): Noise schedule with defined `betas`.
    """
    def __init__(self, betas: Tensor) -> None:
        super().__init__()
        registry = get_diff_registry(betas)
        # register buffers for pre-comp quantities
        for key, value in registry.__dict__.items():
            self.register_buffer(key, value)

    def q_sample(self, x: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """Defines forward diffusion for input tensor by adding noise."""
        if noise is None:
            noise = torch.randn_like(x)
            
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, len(x.shape))
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, len(x.shape))
        
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise

    @abstractmethod
    @torch.no_grad()
    def p_sample(self, *args, **kwargs) -> Tensor:
        """Defines inverse diffusion by subtracting noise from data."""
        pass


class DPMSolverPP2MSampler(Sampler):
    """
    Defines the sampling process at model inference using [DPMSolver++](https://arxiv.org/abs/2211.01095)(2M)
    for joint image-parameter diffusion.

    Args:
        betas (Tensor): Noise schedule with defined `betas`.
        pred_type (str): Model output prediction type ('eps' for noise domain,
                         'x0' for signal, 'v' for velocity). Default to `'eps'`.
    """
    def __init__(self, betas: Tensor, pred_type: Literal['eps', 'x0', 'v'] = 'eps') -> None:
        super().__init__(betas)

        if pred_type not in ['eps', 'x0', 'v']:
            raise ValueError(f"Unsupported prediction type '{pred_type}'.")
        
        self.pred_type = pred_type
        self.lambda_t = torch.log(self.sqrt_alphas_cumprod / self.sqrt_one_minus_alphas_cumprod)
        self.register_buffer('lambda_t', self.lambda_t)

        # multistep history tracking for 2nd order solver
        self.old_x: Optional[tuple[Tensor, Tensor]] = None
        self.old_step: Optional[tuple[Tensor, Tensor]] = None

    def reset_state(self) -> None:
        """Resets sampler multi-step history state to start new sampling loop."""
        self.old_x = None
        self.old_step = None

    def extract_x0(self, model_out: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """Converts model prediction to signal domain."""
        if self.pred_type == 'x0':
            return model_out

        x_dims = len(x_t.shape)
        alpha_t = extract(self.sqrt_alphas_cumprod, t, x_dims)
        sigma_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_dims)

        if self.pred_type == 'eps':
            x0 = (x_t - sigma_t * model_out) / alpha_t
        else:              # 'v'
            x0 = alpha_t * x_t - sigma_t * model_out

        return x0

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x: tuple[Tensor, Tensor],
        t: Tensor,
        condition: tuple[Tensor, Tensor],
        t_prev: Optional[Tensor] = None,
        cfg_scale: float = 10.0,
        **model_kwargs,
    ) -> Tensor:
        """
        Samples joint signal (`img`, `params`) from the model at step `t - 1` in signal domain ('x0').

        Args:
            model (nn.Module): Diffusion U-Net model supporting joint forward pass.
            x (tuple[Tensor, Tensor]): Input noisy tuple (x_img, x_param).
            t (Tensor): Current timestep tensor t (shape: [B]).
            condition (tuple[Tensor, Tensor]): Condition tuple (c_img, c_param).
            t_prev (Optional[Tensor]): Target previous timestep tensor (defaults to `t - 1`).
            cfg_scale (float, optional): Classifier-Free Guidance scale. Defaults to `10.0`.
            **model_kwargs (Any): Kws passed to the model forwards method (e.g., `forward_with_cfg`).

        Returns:
            out (tuple[Tensor, Tensor]): Noisy tuple (x_img_prev, x_param_prev) at `prev_time`.
        """
        # in the following, `s` is the current timestep (input `t`) and `t_prev` is the target timestep `t-1`
        if not exists(t_prev):
            t_prev = torch.clamp(t - 1, min=0)

        x_img, x_pars = x
        c_img, c_pars = condition

        if hasattr(model, 'forward_with_cfg') and cfg_scale != 1.0:
            pred_img, pred_pars = model.forward_with_cfg(
                x_img, x_pars, t, c_img, c_pars, cfg_scale=cfg_scale, **model_kwargs,
            )
        else:
            pred_img, pred_pars = model(
                x_img, x_pars, t, c_img, c_pars, **model_kwargs,
            )

        # convert to x0 domain
        x0_img = self.extract_x0(pred_img, x_img, t)
        x0_pars = self.extract_x0(pred_pars, x_pars, t)

        # extract noise schedule vals for current time `s` and prev time `t`
        img_ndims, pars_ndims = x_img.ndim, x_pars.ndim

        sigma_s_img = extract(self.sqrt_one_minus_alphas_cumprod, t, img_ndims)
        alpha_t_img = extract(self.sqrt_alphas_cumprod, t_prev, img_ndims)
        sigma_t_img = extract(self.sqrt_one_minus_alphas_cumprod, t_prev, img_ndims)

        sigma_s_pars = extract(self.sqrt_one_minus_alphas_cumprod, t, pars_ndims)
        alpha_t_pars = extract(self.sqrt_alphas_cumprod, t_prev, pars_ndims)
        sigma_t_pars = extract(self.sqrt_one_minus_alphas_cumprod, t_prev, pars_ndims)

        # compute log-SNR step `s -> t` + broadcasting
        broadcast_step = lambda m, ndim: m.reshape(-1, *((1,) * (ndim - 1)))
        step = extract(self.lambda_t, t_prev, 1) - extract(self.lambda_t, t, 1)
        step_img, step_pars = map(broadcast_step, (step, step), (img_ndims, pars_ndims))

        # - multi-step DPM-Solver++ update step
        if not exists(self.old_x):
            # first-step: 1st-order Euler step equivalent
            d_img, d_pars = x0_img, x0_pars
        else:
            # subsequent steps: 2nd-order multistep Adams-Bashfort step
            old_x0_img, old_x0_pars = self.old_x
            old_step_img, old_step_pars = map(broadcast_step, self.old_step, (img_ndims, pars_ndims))

            r_img = old_step_img / step_img
            r_pars = old_step_pars / step_pars

            # D = D0 + (0.5 * r) * (D0 - D0_prev)
            d_img = x0_img + (0.5 * r_img) * (x0_img - old_x0_img)
            d_pars = x0_pars + (0.5 * r_pars) * (x0_pars - old_x0_pars)

        # advance sample `x_s -> x_t`
        x_prev_img = (sigma_t_img / sigma_s_img) * x_img + alpha_t_img * (1.0 - torch.exp(-step_img)) * d_img
        x_prev_pars = (sigma_t_pars / sigma_s_pars) * x_pars + alpha_t_pars * (1.0 - torch.exp(-step_pars)) * d_pars

        # 6. Store current predictions in history buffer
        self.old_denoised = (x0_img, x0_pars)
        self.old_step = step

        return x_prev_img, x_prev_pars


@torch.no_grad()
def sample(
    model: nn.Module,
    sampler: DPMSolverPP2MSampler,
    x: tuple[Tensor, Tensor],
    condition: tuple[Tensor, Tensor],
    cfg_scale: float = 10.0,
    num_inference_steps: int = 20,
    full_process: bool = False,
    **kwargs,
) -> tuple[Tensor, Tensor] | list[tuple[Tensor, Tensor]]:
    """
    Samples source shadowgrams and relative parameters from the model,
    with conditioning from input SPSF and initial parameters estimate.
    """
    device = next(model.parameters()).device

    diff_process: list[tuple[Tensor, Tensor]] = []
    to_device: Callable = lambda m: m.to(device)
    to_cpu: Callable = lambda m: m.cpu()
    factory_kws = {'dtype': torch.long, 'device': device}

    pred = tuple(map(to_device, x))
    cond = tuple(map(to_device, condition))
    sampler.to(device)

    nsteps = len(sampler.sqrt_alphas_cumprod)
    timesteps = torch.linspace(nsteps - 1, 0, num_inference_steps + 1, **factory_kws)

    batch = cond[0].shape[0]
    sampler.reset_state()
    for idx in tqdm(range(num_inference_steps), desc=f'Sampling'):
        t = torch.full((batch,), timesteps[idx], **factory_kws)
        t_prev = torch.full((batch,), timesteps[idx + 1], **factory_kws)

        pred = sampler.p_sample(
            model=model,
            x=pred,
            t=t,
            condition=cond,
            t_prev=t_prev,
            cfg_scale=cfg_scale,
            **kwargs,
        )
        if full_process:
            diff_process.append(tuple(map(to_cpu, pred)))
    
    out = tuple(map(to_cpu, pred)) if not full_process else diff_process
    return out


# end