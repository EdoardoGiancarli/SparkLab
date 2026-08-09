"""
Script with U-Net model for LEM-X source shadowgrams generation.
The model is trained with Classifier-Free Guidance and conditioned with the cropped source point-spread function (SPSF) heat-map from
the sky image. At inference, the model provides the source shadowgram matching the given SPSF and an array with the source parameters
--i.e., the source coordinates in the camera local-frame system and the estimate of the collected photon counts.

References:
    [1] Phil Wang's `denoising-diffusion-pytorch`: https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main
    [2] Annotated Diffusion, Niels Rogge and Kashif Rasul (with refs inside, nicely detailed)
        https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=51d9a24c
    [3] APXML website (general info gathering): https://apxml.com/courses/advanced-diffusion-architectures
"""

from functools import partial

import torch
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


__all__ = ['UnetArch']


# end