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

        alpha_s_img = extract(self.sqrt_alphas_cumprod, t, img_ndims)
        sigma_s_img = extract(self.sqrt_one_minus_alphas_cumprod, t, img_ndims)
        alpha_t_img = extract(self.sqrt_alphas_cumprod, t_prev, img_ndims)
        sigma_t_img = extract(self.sqrt_one_minus_alphas_cumprod, t_prev, img_ndims)

        alpha_s_pars = extract(self.sqrt_alphas_cumprod, t, pars_ndims)
        sigma_s_pars = extract(self.sqrt_one_minus_alphas_cumprod, t, pars_ndims)
        alpha_t_pars = extract(self.sqrt_alphas_cumprod, t_prev, pars_ndims)
        sigma_t_pars = extract(self.sqrt_one_minus_alphas_cumprod, t_prev, pars_ndims)

        # compute log-SNR step `s -> t` + broadcasting
        broadcast_h = lambda m, ndim: m.reshape(-1, *((1,) * (ndim - 1)))
        h = extract(self.lambda_t, t_prev, 1) - extract(self.lambda_t, t, 1)
        h_img, h_pars = map(broadcast_h, (h, h), (img_ndims, pars_ndims))

        # - multi-step DPM-Solver++ update step
        if not exists(self.old_x):
            # first-step: 1st-order Euler step equivalent
            d_img, d_pars = x0_img, x0_pars
        else:
            # subsequent steps: 2nd-order multistep Adams-Bashfort step
            old_x0_img, old_x0_pars = self.old_x
            ...

        return ...


@torch.no_grad()
def _sample(
    sample_fn: Callable[[Tensor, Tensor], Tensor], 
    x_start: Tensor,
    timesteps: list[int],
    batch_size: int,
    full_process: bool,
) -> Tensor | list[Tensor]:
    """Sampling algorithm for signal denoising through time steps."""
    img = x_start
    diff_process: list[Tensor] = []
    for idx in tqdm(timesteps, desc=f'Sampling', total=len(timesteps)):
        t = torch.full((batch_size,), idx, device=x_start.device, dtype=torch.long)
        img = sample_fn(img, t)
        if full_process:
            diff_process.append(img.cpu())
    
    out = img if not full_process else diff_process
    return out


@torch.no_grad()
def sample(
    model: nn.Module,
    sampler: Sampler,
    timesteps: int | list[int],
    x_shape: torch.Size,
    eta: float = 0.0,
    x_t: Tensor | None = None,
    full_process: bool = False,
) -> Tensor | list[Tensor]:
    """Samples images from the model through denoising diffusion process."""
    device = next(model.parameters()).device
    batch_size = x_shape[0]
    x_start = (
        x_t if x_t is not None
        else torch.randn(x_shape, device=device)
    )
    timesteps_ = (
        list(range(0, timesteps))[::-1] if isinstance(timesteps, int) else timesteps[::-1]
    )
    sampler = sampler.to(device)
    sample_fn = lambda x, t: sampler.p_sample(model, x, t, eta)
    return _sample(sample_fn, x_start, timesteps_, batch_size, full_process)


# end