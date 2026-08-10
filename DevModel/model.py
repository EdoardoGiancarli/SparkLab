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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Tensor

from .modules import (
    exists,
    Downsample,
    Upsample,
    PreGroupNorm,
    ResidualConnection,
    ResnetBlock,
    ConvNextBlock,
    Attention,
    LinearAttention,
    TimeEmbedding,
)


__all__ = ['Unet', 'LOSS__TBD']


class Unet(nn.Module):
    """
    U-Net model architecture for LEM-X source shadowgrams generation and parameters estimation
    --source coordinates in the camera local-frame system and the estimate of the collected
    photon counts-- through diffusion.
    
    The model accounts for Classifier-Free Guidance, provided by the extracted source point-spread
    function and the initial values of the source parameters (coords estimate and source peak counts).

    Args:
        ...

    ## Architecture:
        ...
    """
    ...




def project() -> None:
    """NOTE: placeholder"""
    ...

def prob_mask_like() -> None:
    """NOTE: placeholder"""
    ...

def repeat() -> None:
    """NOTE: placeholder"""
    ...

def rearrange() -> None:
    """NOTE: placeholder"""
    ...


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        num_classes,
        cond_drop_prob = 0.5,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        attn_dim_head = 32,
        attn_heads = 4
    ):
        super().__init__()

        # classifier free guidance stuff
        self.cond_drop_prob = cond_drop_prob

        # determine dimensions
        self.channels = channels
        input_channels = channels

        init_dim = init_dim if exists(init_dim) else dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings
        time_dim = dim * 4
        self.time_mlp = TimeEmbedding(dim, time_dim)

        # class embeddings
        self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                ResidualConnection(PreGroupNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.mid_attn = ResidualConnection(PreGroupNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                ResidualConnection(PreGroupNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        self.out_dim = out_dim if exists(out_dim) else channels

        self.final_res_block = ResnetBlock(init_dim * 2, init_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        rescaled_phi = 0.,
        remove_parallel_component = True,
        keep_parallel_frac = 0.,
        **kwargs
    ):       
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        update = logits - null_logits

        if remove_parallel_component:
            parallel, orthog = project(update, logits)
            update = orthog + parallel * keep_parallel_frac

        scaled_logits = logits + update * (cond_scale - 1.)

        if rescaled_phi == 0.:
            return scaled_logits, null_logits

        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))
        interpolated_rescaled_logits = rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

        return interpolated_rescaled_logits, null_logits

    def forward(
        self,
        x,
        time,
        classes,
        cond_drop_prob = None
    ):
        batch, device = x.shape[0], x.device

        cond_drop_prob = cond_drop_prob if exists(cond_drop_prob) else self.cond_drop_prob

        # derive condition, with condition dropout for classifier free guidance

        classes_emb = self.classes_emb(classes)

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        c = self.classes_mlp(classes_emb)

        # unet

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)





# end