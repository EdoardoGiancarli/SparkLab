"""
Script for shadowgram sampling at model inference.

References:
    * The Principles of Diffusion Models, Lai et al. (pre-release book, 2026): https://arxiv.org/abs/2510.21890
    * Phil Wang's `denoising-diffusion-pytorch`: https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main
    * Annotated Diffusion, Niels Rogge and Kashif Rasul (with refs inside, nicely detailed)
      https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=51d9a24c
    * APXML website (general info gathering): https://apxml.com/courses/advanced-diffusion-architectures
"""

import torch
from torch.types import Tensor


__all__ = []



# end