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
    'Unet',
    'LOSS__TBD',
]


class Unet(nn.Module):
    """
    U-Net model architecture for LEM-X source shadowgrams generation and parameters estimation
    --source coordinates in the camera local-frame system and the estimate of the collected
    photon counts-- through diffusion.
    
    The model accounts for Classifier-Free Guidance, provided by the extracted source point-
    spread function (SPSF) and the initial values of the source parameters (coords estimate
    and source peak counts). Conditioning is performed by injecting the 2D SPSF in the model
    and the source params, which are attended by cross-attention modules.

    Args:
        ...

    ## Architecture:
        ...
    """
    def __init__(
        self,
        dim: int,
        in_channels: int = 3,
        init_dim: Optional[int] = None,
        out_channels: Optional[int] = None,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        bottleneck_blocks: int = 1,
        drop_cond_prob: float = 0.5,
        cond_img_channels: int = 3,   # NOTE: to handle; condition will be SPSF + src params (tuple[Tensor, Tensor])
        cond_dim: int = 256,          # NOTE: to handle (out proj-channels in cross-attn)
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
        self.init_conv = nn.Conv2d(in_channels + cond_img_channels, init_dim, 7, padding = 3)  # NOTE: signal + cond proj (for now cond is only SPSF) 

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # arch blocks
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult, rmsnorm=rmsnorm)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups, rmsnorm=rmsnorm)

        # time embeddings
        if with_time_emb:
            time_dim = 4 * dim
            self.time_mlp = TimeEmbedding(dim, time_dim)
        else:
            time_dim = None
            self.time_mlp = lambda x: None

        # classifier-free guidance
        self.drop_cond_prob = drop_cond_prob
        self.proj_cond = nn.Conv2d(cond_img_channels, cond_dim, 1)

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

        out_chs = out_channels if exists(out_channels) else in_channels
        self.final_res_block = block_klass(2 * init_dim, init_dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(init_dim, out_chs, 1)

    def forward_with_cfg(
        self,
    ) -> tuple[Tensor, Tensor]:
        """
        Applies model to noisy image at given timestep with CFG following input condition.

        Args:
            ...
        
        Returns:
            ...
        """
        raise NotImplementedError('to be inserted')

    def forward(
        self,
        x: Tensor,
        time: Tensor,
        condition: Tensor,
        drop_cond_prob: Optional[float] = None,
    ) -> Tensor:
        # CFG logic - to insert
        cond_tokens = self.proj_cond(condition)
        # NOTE: flatten spatial dimensions to sequence: (B, cond_dim, 30, 50) -> (B, 1500, cond_dim)
        # NOTE: try do `cond_tokens.shape == (B, 1, H * W, cond_dim) or (B, 1, cond_dim, H * W)`,
        #       but check with Attention mechanisms

        # apply arch
        x = torch.cat([x, condition], dim=1)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h: list[Tensor] = []

        # - encoder
        for block1, block2, attn, cross_attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            x = cross_attn(x, context=cond_tokens)
            h.append(x)

            x = downsample(x)

        # - bottleneck
        for block1, attn, cross_attn, block2 in self.bnecks:
            x = block1(x, t)
            x = attn(x)
            x = cross_attn(x, context=cond_tokens)
            x = block2(x, t)

        # - decoder
        for block1, block2, attn, cross_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = cross_attn(x, context=cond_tokens) 

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


# end