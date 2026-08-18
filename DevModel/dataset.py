"""
Script for joint diffusion dataset handling.
"""

import math

import torch
from torch.types import Tensor


__all__ = []


def normalise_sgs() -> Tensor:
    """
    Normalises source shadowgrams in dataset.

    Args:
        ...
    
    Returns:
        out (Tensor): Normalised source shadowgram images.
    """
    pass


def symm_norm(x: Tensor, edge: float, amax: float = 1.0) -> Tensor:
    """Normalises tensor from physical bounds `[-edge, edge]` to `[-amax, amax]`."""
    return (abs(amax) / abs(edge)) * x


def inverse_symm_norm(x: Tensor, edge: float, amax: float = 1.0) -> Tensor:
    """Inverse tensor normalisation from `[-amax, amax]` back to `[-edge, edge]`."""
    return (abs(edge) / abs(amax)) * x


def log_snr_norm(x: Tensor, var: Tensor, max_snr: float = 2e3) -> Tensor:
    """Normalises photon counts `x` through Log-SNR transform using variance `var`."""
    return torch.log1p(x / torch.sqrt(var)) / math.log1p(max_snr)


def inverse_log_snr_norm(x: Tensor, var: Tensor, max_snr: float = 2e3) -> Tensor:
    """Inverse normalized Log-SNR back to physical photon counts."""
    cts = torch.sqrt(var) * torch.expm1(math.log1p(max_snr) * x)
    return cts.clamp(min=0.0)


def normalise_params() -> Tensor:
    """
    Normalises source params in dataset.
    The source coordinates are normalised in [-1, 1] mirroring the LEM-X cameras
    FoV, expressed in the instruments local-frame system.
    The collected photons are normalised by applying a log-transform to the SNR,
    using the detector instrumental variance and normalising in [0, 1] through
    a max SNR-value (default to `SNRmax = 2e3`).

    Args:
        ...
    
    Returns:
        out (Tensor): Normalised source parameters array.
    """
    pass


# end