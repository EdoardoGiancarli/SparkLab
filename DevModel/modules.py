"""
Script with U-Net arch modules.

References:
    * Phil Wang's `denoising-diffusion-pytorch`: https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main
    * Annotated Diffusion, Niels Rogge and Kashif Rasul (with refs inside, nicely detailed)
        https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=51d9a24c
    * The Principles of Diffusion Models, Lai et al. (pre-release book, 2026): https://arxiv.org/abs/2510.21890
    * Sasha Rush et al., blog (Transformer): https://nlp.seas.harvard.edu/annotated-transformer/
    * Jay Alammar's blog (Transformer): https://jalammar.github.io/illustrated-transformer/
    * Lilian Weng's blog (Diffusion Models): https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
    * APXML website (general info gathering): https://apxml.com/courses/advanced-diffusion-architectures
    * Original papers
"""

import math
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.types import Tensor


__all__ = [
    'exists',
    'Downsample',
    'Upsample',
    'modulate',

    'PreGroupNorm',
    'ResidualConnection',
    
    'Block',
    'ResnetBlock',
    'ConvNextBlock',

    'Attention',
    'LinearAttention',

    'SinPosEmbedding',
    'TimeEmbedding',
]


def exists(value: Any) -> bool:
    """Checks input `value` existence."""
    return value is not None

def Downsample(chs: int, ksize: int = 4, stride: int = 2, pad: int = 1) -> Callable[[Tensor], Tensor]:
    """Defines a downsampling operation through a `nn.Conv2d` obj."""
    return nn.Conv2d(chs, chs, ksize, stride, pad)

def Upsample(chs: int, ksize: int = 4, stride: int = 2, pad: int = 1) -> Callable[[Tensor], Tensor]:
    """Defines an upsampling operation through a `nn.ConvTranspose2d` obj."""
    return nn.ConvTranspose2d(chs, chs, ksize, stride, pad)

def modulate(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """Modulates input features with given scale and shift."""
    return (1 + scale) * x + shift




class PreGroupNorm(nn.Module):
    """
    Group normalisation applied before a given module [1, 2].
    The DDPM authors interleave the Conv/Attention layers
    of the U-Net with group normalization (Wu et al., 2018).
    Note that there's been a debate about whether to apply
    normalisation before or after Attention in Transformers.
    """
    def __init__(self, dim: int, fn: Callable) -> None:
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return self.fn(x)


class ResidualConnection(nn.Module):
    """
    Defines a residual connection for a network by adding the
    given input obj to the output of a particular function.
    """
    def __init__(self, fn: Callable[[Tensor, Any], Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x + self.fn(x, *args, **kwargs)

    


class Block(nn.Module):
    """Model block architecture with optional AdaGN parameters."""
    def __init__(self, dim: int, dim_out: int, groups: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, scale_shift: Optional[tuple[Tensor, Tensor]] = None) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)

        if exists(scale_shift):
            x = modulate(x, *scale_shift)

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """
    Residual block from: https://arxiv.org/abs/1512.03385,
    with AdaGN-zero implementation for time embeddings.
    """
    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        time_emb_dim: Optional[int] = None,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        # AdaGN with zero init for training stability
        self.proj = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, 2 * dim_out))
            if exists(time_emb_dim) else None
        )
        if exists(self.proj):
            nn.init.zeros_(self.proj[-1].weight)
            nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: Tensor, time_emb: Optional[Tensor] = None) -> Tensor:
        scale_shift = None
        if all(map(exists, (self.proj, time_emb))):
            condition = self.proj(time_emb)
            condition = condition[..., None, None]
            scale_shift = condition.chunk(2, dim=1)

        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class ConvNextBlock(nn.Module):
    """
    Residual block from: https://arxiv.org/abs/2201.03545,
    with AdaGN-zero implementation for time embeddings.
    """
    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        time_emb_dim: Optional[int] = None,
        mult: int = 2,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim) if norm else nn.Identity()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        # AdaGN with zero init for training stability
        self.proj = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, 2 * dim))
            if exists(time_emb_dim) else None
        )
        if exists(self.proj):
            nn.init.zeros_(self.proj[-1].weight)
            nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: Tensor, time_emb: Optional[Tensor] = None) -> Tensor:
        h = self.ds_conv(x)
        h = self.norm(h)
        if all(map(exists, (self.proj, time_emb))):
            condition = self.proj(time_emb)
            condition = condition[..., None, None]
            scale, shift = condition.chunk(2, dim=1)
            h = modulate(h, scale, shift)
        h = self.net(h)
        return h + self.res_conv(x)




class Attention(nn.Module):
    """
    Spatial multi-head attention module.
    Can be used for both self- and cross-attention.
    """
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.scale = pow(dim_head, -0.5)
        self.heads = heads
        hidden_dim = dim_head * heads
        context_dim = context_dim if exists(context_dim) else dim

        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(context_dim, 2 * hidden_dim, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
    
    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if not exists(context):
            context = x
        
        b, _, h, w = x.shape
        _, _, h_c, w_c = context.shape
        q = self.to_q(x).view(b, self.heads, -1, h * w)
        kv = self.to_kv(context).chunk(2, dim=1)
        k, v = map(lambda t: t.view(b, self.heads, -1, h_c * w_c), kv)

        q = q * self.scale
        sim = torch.einsum('bhdi, bhdj -> bhij', q, k)
        # safe softmax norm to prevent numerical overflow / NaN values
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum('bhij, bhdj -> bhdi', attn, v)
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """
    Spatial multi-head attention module, linear variant.
    Can be used for both self- and cross-attention.
    """
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.scale = pow(dim_head, -0.5)
        self.heads = heads
        hidden_dim = dim_head * heads
        context_dim = context_dim if exists(context_dim) else dim

        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(context_dim, 2 * hidden_dim, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1), 
            nn.GroupNorm(1, dim)
        )
    
    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if not exists(context):
            context = x
        
        b, _, h, w = x.shape
        _, _, h_c, w_c = context.shape
        q = self.to_q(x).view(b, self.heads, -1, h * w)
        kv = self.to_kv(context).chunk(2, dim=1)
        k, v = map(lambda t: t.view(b, self.heads, -1, h_c * w_c), kv)

        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        context_mat = torch.einsum('bhdn, bhen -> bhde', k, v)

        out = torch.einsum('bhde, bhdn -> bhen', context_mat, q)
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)




class SinPosEmbedding(nn.Module):
    """Defines sinusoidal position embedding to encode the batch noise time step."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, time: Tensor) -> Tensor:
        device = time.device
        half_dim = self.dim // 2

        freqs = math.log(1e4) / (half_dim - 1)
        freqs = torch.exp(-freqs * torch.arange(half_dim, device=device))
        freqs = time[:, None] * freqs[None, :]

        embeddings = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    """
    Time-step embedding based on sinusoidal encoding for the diffusion process.
    The instance accounts for positional encoding and mid-MLP processing (the
    final projection to scale and shift is moved within the residual blocks).
    """
    def __init__(self, dim: int, emb_chs: int) -> None:
        super().__init__()
        self.emb = nn.Sequential(
            SinPosEmbedding(dim),
            nn.Linear(dim, emb_chs),
            nn.GELU(),
            nn.Linear(emb_chs, emb_chs),
        )
        
    def forward(self, time: Tensor) -> Tensor:
        t = self.emb(time)
        return t


# end


# class SelfAttention(nn.Module):
#     """Spatial multi-head self-attention module."""
#     def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
#         super().__init__()
#         self.scale = pow(dim_head, -0.5)
#         self.heads = heads
#         hidden_dim = dim_head * heads
#         self.to_qkv = nn.Conv2d(dim, 3 * hidden_dim, 1, bias=False)
#         self.to_out = nn.Conv2d(hidden_dim, dim, 1)

#     def forward(self, x: Tensor) -> Tensor:
#         b, _, h, w = x.shape
#         qkv = self.to_qkv(x).chunk(3, dim=1)
#         q, k, v = map(lambda t: t.view(b, self.heads, -1, h * w), qkv)
#         q = q * self.scale
#         sim = torch.einsum('bhdi, bhdj -> bhij', q, k)
#         # safe softmax norm to prevent numerical overflow / NaN values
#         sim = sim - sim.amax(dim=-1, keepdim=True).detach()
#         attn = sim.softmax(dim=-1)
#         out = torch.einsum('bhij, bhdj -> bhdi', attn, v)
#         out = out.reshape(b, -1, h, w)
#         return self.to_out(out)


# class LinearSelfAttention(nn.Module):
#     """Spatial multi-head self-attention module, linear variant."""
#     def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
#         super().__init__()
#         self.scale = pow(dim_head, -0.5)
#         self.heads = heads
#         hidden_dim = dim_head * heads
#         self.to_qkv = nn.Conv2d(dim, 3 * hidden_dim, 1, bias=False)
#         self.to_out = nn.Sequential(
#             nn.Conv2d(hidden_dim, dim, 1), 
#             nn.GroupNorm(1, dim)
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         b, _, h, w = x.shape
#         qkv = self.to_qkv(x).chunk(3, dim=1)
#         q, k, v = map(lambda t: t.view(b, self.heads, -1, h * w), qkv)
#         q = q.softmax(dim=-2)
#         k = k.softmax(dim=-1)
#         q = q * self.scale
#         context = torch.einsum('bhdn, bhen -> bhde', k, v)
#         out = torch.einsum('bhde, bhdn -> bhen', context, q)
#         out = out.reshape(b, -1, h, w)
#         return self.to_out(out)