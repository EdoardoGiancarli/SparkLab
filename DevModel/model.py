"""
Script with U-Net model for LEM-X source shadowgrams generation.
The model is trained with Classifier-Free Guidance and conditioned with the cropped source point-spread function (SPSF) heat-map from
the sky image. At inference, the model provides the source shadowgram matching the given SPSF and an array with the source parameters
--i.e., the source coordinates in the camera local-frame system and the estimate of the collected photon counts.

References:
    * Phil Wang's `denoising-diffusion-pytorch`: https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main,
      CFG script (https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py)
    * Annotated Diffusion, Niels Rogge and Kashif Rasul (with refs inside, nicely detailed)
      https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=51d9a24c
    * The Principles of Diffusion Models, Lai et al. (pre-release book, 2026): https://arxiv.org/abs/2510.21890
    * APXML website (general info gathering): https://apxml.com/courses/advanced-diffusion-architectures
"""

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Tensor

from .modules import (
    exists,
    Downsample,
    Upsample,
    PreNorm,
    ResidualConnection,
    ResnetBlock,
    ConvNextBlock,
    Attention,
    LinearAttention,
    TimeEmbedding,
)


__all__ = [
    'prob_mask_like',
    'decompose',
    'Unet',
    'LOSS__TBD',
]


def prob_mask_like(
    prob: float,
    shape: tuple[int, ...],
    device: Optional[torch.device] = None,
) -> Tensor:
    """Defines mask for events with probability `< prob`."""
    if not (0 <= prob <= 1):
        raise ValueError(f'Invalid prob value {prob}, must be in [0, 1].')

    if prob == 0:
        return torch.zeros(shape, dtype=torch.bool, device=device)
    elif prob == 1:
        return torch.ones(shape, dtype=torch.bool, device=device)

    return torch.rand(shape, device=device) < prob


def decompose(x: Tensor, direction: Tensor) -> tuple[Tensor, Tensor]:
    """Extracts `x` parallel and orthogonal components wrt given direction."""
    # original: `denoising-diffusion-pytorch/cfg`, line 87
    # little refactoring to have all in plain torch
    xshape, dtype = x.shape, x.dtype
    # flatten to (b, n) so to compute dot product per batch,
    # and convert to double for numerical stability
    fx, fdir = map(lambda t: t.flatten(start_dim=1).double(), (x, direction))
    fdir_hat = F.normalize(fdir, dim=-1)

    parallel = (fx * fdir_hat).sum(dim=-1, keepdim=True) * fdir_hat
    orthog = fx - parallel
    parallel, orthog = map(lambda t: t.view(xshape).to(dtype), (parallel, orthog))
    
    return parallel, orthog


class Unet(nn.Module):
    """
    U-Net architecture for LEM-X source shadowgrams generation and parameters estimation (i.e., the
    source coordinates in the camera local reference-frame and the total collected photon counts)
    through Gaussian diffusion.

    The model supports Classifier-Free Guidance (CFG), conditioned on the extracted source point-
    spread function (SPSF) and initial source parameter estimates (coords and SPSF peak counts). 
    Conditioning features are projected into hidden context dimensions via and injected into the
    network through cross-attention blocks across multiple spatial scales.

    Args:
        dim (int):
            Base feature map channel dimension.
        in_channels (int, optional):
            Input image channels (e.g., noisy shadowgram). Defaults to `3`.
        srcparams_num (int, optional):
            Input source params number (e.g., coords + photon counts). Defaults to `3`.
        init_dim (int, optional):
            Output channel dimension for initial convolution. Defaults to `dim`.
        out_channels (int, optional):
            Output channels produced by final convolution. Defaults to `in_channels`.
        dim_mults (tuple[int, ...], optional):
            Channel multipliers per U-Net depth level. Defaults to `(1, 2, 4, 8)`.
        bottleneck_blocks (int, optional):
            Number of repeated bottleneck blocks. Defaults to 1.
        drop_cond_prob (float, optional):
            Condition dropout probability for CFG during training. Defaults to 0.5.
        cond_img_channels (int, optional):
            Channel count of raw input condition image. Defaults to `3`.
        cond_srcparams_num (int, optional):
            Source params number of raw input condition array. Defaults to `3`.
        cond_dim (int, optional):
            Inner hidden dimension for cross-attention conditioning features. Defaults to `256`.
        attn_dim_head (int, optional):
            Channel dimension per attention head. Defaults to `32`.
        attn_heads (int, optional):
            Number of attention heads. Defaults to `4`.
        with_time_emb (bool, optional):
            Whether to compute sinusoidal time embeddings. Defaults to `True`.
        use_convnext (bool, optional):
            If True, uses ConvNeXt blocks; otherwise uses ResNet blocks. Defaults to `True`.
        convnext_mult (int, optional):
            Expansion factor for ConvNeXt hidden dimensions. Defaults to `2`.
        resnet_block_groups (int, optional):
            Group count for GroupNorm in ResNet blocks. Defaults to `8`.
        rmsnorm (bool, optional):
            If `True`, uses RMSNorm instead of GroupNorm. Defaults to `True`.

    ## Architecture:
    ```
        Input (B, in_chs, H, W) ──► init_conv ──► Encoder [Downsample + Self-Attn + Cross-Attn]
                                                    │
                                                    ▼
                                                Bottleneck [Attn + Cross-Attn]
                                                    │
                                                    ▼
        Output (B, out_chs, H, W) ◄── final_conv ◄── Decoder [Upsample + Skip-Cat + Self-Attn + Cross-Attn]
    ```
    """
    def __init__(
        self,
        dim: int,
        in_channels: int = 3,
        srcparams_num: int = 3,
        init_dim: Optional[int] = None,
        out_channels: Optional[int] = None,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        bottleneck_blocks: int = 1,
        drop_cond_prob: float = 0.5,
        cond_img_channels: int = 3,
        cond_srcparams_num: int = 3,
        cond_dim: int = 256,
        attn_dim_head: int = 32,
        attn_heads: int = 4,
        with_time_emb: bool = True,
        use_convnext: bool = True,
        convnext_mult: int | None = 2,
        resnet_block_groups: int | None = 8,
        rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        # determine dimensions
        init_dim = init_dim if exists(init_dim) else dim
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # input proj + arch blocks
        self.init_conv = nn.Conv2d(in_channels, init_dim, 7, padding = 3)

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult, rmsnorm=rmsnorm)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups, rmsnorm=rmsnorm)

        LineAttn = partial(LinearAttention, )

        # time embedding + source params embedding
        time_dim = 4 * dim
        self.time_mlp = (
            TimeEmbedding(dim, time_dim) if with_time_emb else None
        )
        self.x_params_mlp = nn.Sequential(
            nn.Linear(srcparams_num, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # classifier-free guidance
        self.drop_cond_prob = drop_cond_prob
        self.proj_cond = nn.Conv2d(cond_img_channels + cond_srcparams_num, cond_dim, 1)

        # layers
        self.downs = nn.ModuleList([])
        self.bnecks = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # - encoder
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        ResidualConnection(PreNorm(dim_in, LinearAttention(dim_in, rmsnorm=rmsnorm))),
                        ResidualConnection(PreNorm(dim_in, LinearAttention(dim_in, context_dim=cond_dim, rmsnorm=rmsnorm))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        # - bottleneck
        mid_dim = dims[-1]
        for _ in range(bottleneck_blocks):
            self.bnecks.append(
                nn.ModuleList(
                    [
                        block_klass(mid_dim, mid_dim, time_emb_dim=time_dim),
                        ResidualConnection(PreNorm(mid_dim, Attention(mid_dim, heads=attn_heads, dim_head=attn_dim_head))),
                        ResidualConnection(PreNorm(mid_dim, Attention(mid_dim, heads=attn_heads, dim_head=attn_dim_head, context_dim=cond_dim))),
                        block_klass(mid_dim, mid_dim, time_emb_dim=time_dim),
                    ]
                )
            )

        # - decoder
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        ResidualConnection(PreNorm(dim_out, LinearAttention(dim_out, rmsnorm=rmsnorm))),
                        ResidualConnection(PreNorm(dim_out, LinearAttention(dim_out, context_dim=cond_dim, rmsnorm=rmsnorm))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        # output proj: img + params
        self.final_res_block = block_klass(2 * init_dim, init_dim, time_emb_dim=time_dim)

        out_chs = out_channels if exists(out_channels) else in_channels
        self.img_out_conv = nn.Conv2d(init_dim, out_chs, 1)

        self.params_out_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(init_dim, dim),
            nn.GELU(),
            nn.Linear(dim, srcparams_num),
        )

    def forward_with_cfg(
        self,
        x_img: Tensor,
        x_params: Tensor,
        time: Tensor,
        cond_img: Tensor,
        cond_params: Tensor,
        cfg_scale: float = 10.0,
        rmv_parallel_comp: bool = False,
        parallel_comp_scale: float = 0.0,
        cfg_var_rescale: float = 0.0,
        eps: float = 1e-6,
    ) -> tuple[Tensor, Tensor]:
        """
        Applies model to noisy image at given timestep with dual-pass classifier-free guidance
        sampling (CFG), following input condition and with optional parallel component removal
        (APG) and CFG variance rescaling to mitigate artifacts at high guidance scales.

        Args:
            x_img (Tensor):
                Noisy image tensor of shape `(B, in_channels, H, W)`.
            x_params (Tensor):
                Noisy params tensor of shape `(B, srcparams_num)`.
            time (Tensor):
                Diffusion timesteps tensor of shape `(B,)`.
            cond_img (Tensor):
                Conditioning image of shape `(B, cond_img_channels, H_c, W_c)`.
            cond_params (Tensor):
                Conditioning params of shape `(B, cond_srcparams_num)`.
            cfg_scale (float, optional):
                Classifier-Free Guidance scale. Defaults to `10.0`.
            rmv_parallel_comp (bool, optional):
                If `True`, decomposes the guidance update vector and removes components parallel
                to the conditional prediction to prevent clipping artifacts. Defaults to `False`.
            parallel_comp_scale (float, optional):
                Retained fraction of the parallel guidance component when `rmv_parallel_comp=True`.
                Defaults to `0.0`.
            cfg_var_rescale (float, optional):
                Interpolation factor in [0, 1] for matching the std of guided output back
                to conditional output std. Defaults to `0.0` (disabled).
            eps (float, optional):
                Tolerance for CFG variance rescaling. Defaults to `1e-6`.

        Returns:
            out (tuple[Tensor, Tensor]):
                * Model predicted source shadowgram, with shape `(B, out_channels, H, W)`
                * Model predicted source parameters, with shape `(B, srcparams_num)`
        """
        # CFG
        logits_img, logits_params = self.forward(
            x_img, x_params, time, cond_img, cond_params, drop_cond_prob=0.0
        )
        if cfg_scale == 1.0:
            return logits_img, logits_params

        null_c_img = torch.zeros_like(cond_img)
        null_c_params = torch.zeros_like(cond_params)
        uncond_img, uncond_params = self.forward(
            x_img, x_params, time, null_c_img, null_c_params, drop_cond_prob=1.0
        )

        update_img = logits_img - uncond_img
        update_params = logits_params - uncond_params

        # modulate `condition` parallel component in `update` for high guidance
        # https://arxiv.org/abs/2410.02416
        if rmv_parallel_comp:
            parallel_img, orthog_img = decompose(update_img, logits_img)
            update_img = orthog_img + parallel_comp_scale * parallel_img

            parallel_p, orthog_p = decompose(update_params, logits_params)
            update_params = orthog_p + parallel_comp_scale * parallel_p

        scaled_img = uncond_img + cfg_scale * update_img
        scaled_params = uncond_params + cfg_scale * update_params

        # `scaled_logits` variance rescaling for high guidance (NOTE: img only)
        #   * https://arxiv.org/abs/2205.11487
        #   * https://arxiv.org/pdf/2305.08891
        if cfg_var_rescale > 0:
            std_fn = partial(torch.std, dim=tuple(range(1, scaled_img.ndim)), keepdim=True) # std per batch + broadcasting
            rescaled_img = std_fn(logits_img) * scaled_img / (std_fn(scaled_img) + eps)
            scaled_img = (
                rescaled_img * cfg_var_rescale + scaled_img * (1 - cfg_var_rescale)
            )

        return scaled_img, scaled_params

    def forward(
        self,
        x_img: Tensor,
        x_params: Tensor,
        time: Tensor,
        cond_img: Tensor,
        cond_params: Tensor,
        drop_cond_prob: Optional[float] = None,
    ) -> tuple[Tensor, Tensor]:
        b, device = x_img.shape[0], x_img.device
        drop_cond_prob = drop_cond_prob if exists(drop_cond_prob) else self.drop_cond_prob

        # joint CFG condition dropout during training 
        if exists(drop_cond_prob) and drop_cond_prob > 0 and self.training:
            keep_mask = prob_mask_like(1 - drop_cond_prob, (b,), device=device)
            # drop 2D img condition and replace with null cond
            img_mask = keep_mask[..., None, None, None]
            cond_img = torch.where(img_mask, cond_img, torch.zeros_like(cond_img))
            # drop 1D arr condition and replace with null cond
            par_mask = keep_mask[..., None]
            cond_params = torch.where(par_mask, cond_params, torch.zeros_like(cond_params))

        # combine img + params conditions (broadcast `cond_params` to match `cond_img`) and project
        _, _, hc, wc = cond_img.shape
        cond_params_spatial = cond_params[..., None, None].expand(-1, -1, hc, wc)
        cond = torch.cat([cond_img, cond_params_spatial], dim=1)
        cond = self.proj_cond(cond)

        # apply arch
        # - input img proj
        x = self.init_conv(x_img)

        # - combine input params and timestep embedding
        t_emb = self.time_mlp(time) if exists(self.time_mlp) else 0
        p_emb = self.x_params_mlp(x_params)
        t = t_emb + p_emb

        # - skip conns
        r = x.clone()
        h: list[Tensor] = []

        # - encoder
        for block1, block2, attn, cross_attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            x = cross_attn(x, context=cond)
            h.append(x)

            x = downsample(x)

        # - bottleneck
        for block1, attn, cross_attn, block2 in self.bnecks:
            x = block1(x, t)
            x = attn(x)
            x = cross_attn(x, context=cond)
            x = block2(x, t)

        # - decoder
        for block1, block2, attn, cross_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = cross_attn(x, context=cond) 

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)

        pred_img = self.img_out_conv(x)
        pred_params = self.params_out_proj(x)

        return pred_img, pred_params


# end