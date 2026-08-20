"""
Container for the `bloodmoon` and `darksun` packages, to link LEM-X instrumental coded-aperture and joint-diffusion frameworks.
Here, the used funcs are converted from `Numpy` to `PyTorch` for efficient, uniform interface development.

Reference:
    * Giancarli, E. et al., "Enhancing LEM-X Imaging with the IROS Sky Reconstruction Pipeline", Astronomy & Computing, 2026, in prep.
"""

import torch
from torch.types import Tensor


__all__ = []


# end