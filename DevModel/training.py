"""
Script for shadowgram sampling at model inference.
"""

import torch
from torch.types import Tensor


__all__ = []


def normalise_sgs() -> Tensor:
    """
    Normalises source shadowgrams in dataset.

    Args:
        ...
    
    Returns:
        out (Tensor): Normalised source shadowgrams images.
    """
    pass

def normalise_params() -> Tensor:
    """
    Normalises source params in dataset.
    The source coordinates are normalised in [-1, 1] mirroring the LEM-X cameras
    FoV, expressed in the instruments local-frame system.
    The collected photons are normalised by applying first a log-transform, and
    then a z-score norm to prevent large values dominance.

    Args:
        ...
    
    Returns:
        out (Tensor): Normalised source parameter arrays.
    """
    pass


# end