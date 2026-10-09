"""
Container for the `bloodmoon` and `darksun` packages, to link LEM-X instrumental coded-aperture and joint-diffusion frameworks.
Here, the used funcs are converted from `Numpy` to `PyTorch` for efficient, uniform interface development.

Reference:
    * Giancarli, E. et al., "Enhancing LEM-X Imaging with the IROS Sky Reconstruction Pipeline", Astronomy & Computing, 2026, in prep.
    * `bloodmoon` package @ https://github.com/peppedilillo/bloodmoon
    * `darksun` package @ https://github.com/EdoardoGiancarli/darksun
"""

from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Callable, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from astropy.io import fits
from astropy.io.fits.fitsrec import FITS_rec
from scipy.stats import binned_statistic_2d

import torch
import torch.nn.functional as F
from torch.types import Tensor


__all__ = [
    # custom types
    'UpscaleFactor',
    'BinsRectangular',
    # I/O funcs for coded-mask/camera data loading
    'validate_fits',
    '_fold',
    'load_from_fits',
    # camera management objects: geometry info/mask pattern/instr properties
    '_bisect_left',
    '_bisect_right',
    '_bisect_interval',
    '_upscale',
    '_shift',
    '_correlate',
    'CodedMaskSpecs',
    'CodedMaskCamera',
    'codedmask',
    # data CAI operations
    'decode',
    'decode_batch',
    'solid_angle',          # TBD
    'solid_angle_profile',  # TBD
    'variance',             # TBD
    'snratio',              # TBD
    # tensor operations
    'argmax',
    'find_boxmax',
    'crop',
    'crop_batch',
    # coords conversion
    'shift2pos',
]


class UpscaleFactor(NamedTuple):
    """
    Upscaling factors for x and y dimensions.

    Args:
        x (Tensor): Upscaling factor for x dimension.
        y (Tensor): Upscaling factor for y dimension.
    """
    x: int
    y: int


class BinsRectangular(NamedTuple):
    """
    Two-dimensional binning structure for rectangular coordinates.

    Args:
        x (Tensor): Tensor of x-coordinate bin edges.
        y (Tensor): Tensor of y-coordinate bin edges.
    """
    x: Tensor
    y: Tensor

"""
        ⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⢻⣿⡗⢶⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⣄
        ⠀⢻⣇⠀⠈⠙⠳⣦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⠶⠛⠋⣹⣿⡿
        ⠀⠀⠹⣆⠀⠀⠀⠀⠙⢷⣄⣀⣀⣀⣤⣤⣤⣄⣀⣴⠞⠋⠉⠀⠀⠀⢀⣿⡟⠁
        ⠀⠀⠀⠙⢷⡀⠀⠀⠀⠀⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⠋⠀⠀
        ⠀⠀⠀⠀⠈⠻⡶⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣠⡾⠋⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⣼⠃⠀⢠⠒⣆⠀⠀⠀⠀⠀⠀⢠⢲⣄⠀⠀⠀⢻⣆⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⢰⡏⠀⠀⠈⠛⠋⠀⢀⣀⡀⠀⠀⠘⠛⠃⠀⠀⠀⠈⣿⡀⠀⠀⠀⠀
        ⠀⠀⠀⠀⣾⡟⠛⢳⠀⠀⠀⠀⠀⣉⣀⠀⠀⠀⠀⣰⢛⠙⣶⠀⢹⣇⠀⠀⠀⠀
        ⠀⠀⠀⠀⢿⡗⠛⠋⠀⠀⠀⠀⣾⠋⠀⢱⠀⠀⠀⠘⠲⠗⠋⠀⠈⣿⠀⠀⠀⠀
        ⠀⠀⠀⠀⠘⢷⡀⠀⠀⠀⠀⠀⠈⠓⠒⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡇⠀⠀⠀
        ⠀⠀⠀⠀⠀⠈⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣧⠀⠀⠀
        ⠀⠀⠀⠀⠀⠈⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠁⠀⠀⠀
"""

def validate_fits(filepath: Path | str) -> bool:
    """
    Validate presence and format of FITS file.

    Checks if the specified file exists and has a valid FITS format signature.
    Supports both string paths and Path objects.

    Args:
        filepath: Path to the FITS file to validate

    Returns:
        True if file exists and has valid FITS format

    Raises:
        FileNotFoundError: If FITS file does not exist
        ValueError: If file is not in valid FITS format
    """

    def validate_signature(filepath: Path) -> bool:
        """
        Following astropy's approach, reads the first FITS card (80 bytes)
        and checks for the SIMPLE keyword signature.

        Args:
            filepath: Path object pointing to the file to validate

        Returns:
            bool: True if file has a valid FITS signature, False otherwise
        """
        try:
            with open(filepath, "rb") as file:
                # FITS signature is supposed to be in the first 30 bytes, but to
                # allow reading various invalid files we will check in the first
                # card (80 bytes).
                simple = file.read(80)
        except OSError:
            return False

        fits_signature = b"SIMPLE  =                    T"

        match_sig = simple[:29] == fits_signature[:-1] and simple[29:30] in (b"T", b"F")
        return match_sig

    if not Path(filepath).is_file():
        raise FileNotFoundError(f"FITS file '{filepath}' does not exist.")
    elif not validate_signature(Path(filepath)):
        raise ValueError("File not in valid FITS format.")
    return True


def _fold(
    ml: FITS_rec,
    mask_bins: BinsRectangular,
) -> NDArray:
    """
    Convert mask data from FITS record to 2D binned array.

    Args:
        ml: FITS record containing mask data
        mask_bins: Binning structure for the mask

    Returns:
        2D array containing binned mask data
    """
    arr = binned_statistic_2d(ml["X"], ml["Y"], ml["VAL"], statistic="max", bins=[mask_bins.x, mask_bins.y])
    return arr[0].T


def load_from_fits(filepath: str | Path) -> tuple:
    """
    Load mask data and specifications from FITS file.

    Extracts mask patterns, decoder patterns, bulk patterns, and geometric
    specifications from a coded mask FITS file. Returns callable thunks for
    lazy loading of array data.

    Args:
        filepath: Path to the mask FITS file

    Returns:
        Tuple containing:
            - get_mask: Callable that returns mask pattern as 2D array
            - get_decoder: Callable that returns decoder pattern as 2D array
            - get_bulk: Callable that returns bulk pattern as 2D array
            - specs: Dictionary of mask specifications and geometric parameters
    """
    h0 = dict(fits.getheader(filepath, ext=0))
    specs = {
        "detector_minx": h0["PLNXMIN"],
        "detector_maxx": h0["PLNXMAX"],
        "detector_miny": h0["PLNYMIN"],
        "detector_maxy": h0["PLNYMAX"],
        "mask_thickness": h0["MASKTHK"],
        "mask_detector_distance": h0["MDDIST"] + h0["MASKTHK"],
    }
    h2 = dict(fits.getheader(filepath, ext=2))
    specs |= {
        "mask_minx": h2["MINX"],
        "mask_miny": h2["MINY"],
        "mask_maxx": h2["MAXX"],
        "mask_maxy": h2["MAXY"],
        "slit_deltax": h2["DXSLIT"],
        "slit_deltay": h2["DYSLIT"],
    }
    h3 = dict(fits.getheader(filepath, ext=3))
    specs |= {
        "mask_deltax": h3["ELXDIM"],
        "mask_deltay": h3["ELYDIM"],
    }

    l, r = specs["mask_minx"], specs["mask_maxx"]
    b, t = specs["mask_miny"], specs["mask_maxy"]
    mask_bins = BinsRectangular(
        np.linspace(l, r, int((r - l) / specs["mask_deltax"]) + 1),
        np.linspace(b, t, int((t - b) / specs["mask_deltay"]) + 1),
    )

    get_mask = lambda: _fold(fits.getdata(filepath, ext=2), mask_bins).astype(int)
    get_decoder = lambda: _fold(fits.getdata(filepath, ext=3), mask_bins)
    get_bulk = lambda: _fold(fits.getdata(filepath, ext=4), mask_bins)

    return get_mask, get_decoder, get_bulk, specs

"""
        ⠀⠀⠀⠀⠀⢸⠓⢄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⢸⠀⠀⠑⢤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⢸⡆⠀⠀⠀⠙⢤⡷⣤⣦⣀⠤⠖⠚⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⣠⡿⠢⢄⡀⠀⡇⠀⠀⠀⠀⠀⠉⠀⠀⠀⠀⠀⠸⠷⣶⠂⠀⠀⠀⣀⣀⠀⠀⠀
        ⢸⣃⠀⠀⠉⠳⣷⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠉⠉⠉⠉⠉⢉⡭⠋
        ⠀⠘⣆⠀⠀⠀⠁⠀⢀⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠋⠀⠀
        ⠀⠀⠘⣦⠆⠀⠀⢀⡎⢹⡀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⡀⣠⠔⠋⠀⠀⠀⠀
        ⠀⠀⠀⡏⠀⠀⣆⠘⣄⠸⢧⠀⠀⠀⠀⢀⣠⠖⢻⠀⠀⠀⣿⢥⣄⣀⣀⣀⠀⠀
        ⠀⠀⢸⠁⠀⠀⡏⢣⣌⠙⠚⠀⠀⠠⣖⡛⠀⣠⠏⠀⠀⠀⠇⠀⠀⠀⠀⢙⣣⠄
        ⠀⠀⢸⡀⠀⠀⠳⡞⠈⢻⠶⠤⣄⣀⣈⣉⣉⣡⡔⠀⠀⢀⠀⠀⣀⡤⠖⠚⠀⠀
        ⠀⠀⡼⣇⠀⠀⠀⠙⠦⣞⡀⠀⢀⡏⠀⢸⣣⠞⠀⠀⠀⡼⠚⠋⠁⠀⠀⠀⠀⠀
        ⠀⢰⡇⠙⠀⠀⠀⠀⠀⠀⠉⠙⠚⠒⠚⠉⠀⠀⠀⠀⡼⠁⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⢧⡀⠀⢠⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠙⣶⣶⣿⠢⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠉⠀⠀⠀⠙⢿⣳⠞⠳⡄⠀⠀⠀⢀⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠀⠀⠹⣄⣀⡤⠋⠀⠀⠀⠀⠀⠀⠀⠀
"""

def _bisect_right(x: Tensor, value: int | float | Tensor) -> int | Tensor:
    """Implementation of `bisect.bisect_right` for torch tensors."""
    out = torch.searchsorted(x, value, side='right')
    return out.item() if isinstance(value, (int, float)) else out


def _bisect_left(x: Tensor, value: int | float | Tensor) -> int | Tensor:
    """Implementation of `bisect.bisect_left` for torch tensors."""
    out = torch.searchsorted(x, value, side='left')
    return out.item() if isinstance(value, (int, float)) else out


def _bisect_interval(x: Tensor, start: float, stop: float) -> tuple[int, int]:
    """
    Given a monotonically increasing tensor of floats and a float interval (start, stop)
    in it, returns the indices of the smallest sub-tensor containing the interval.

    Args:
        a (Tensor): A monotonically increasing tensor of floats.
        start (float): The lower bound of the interval. Must be greater than or equal to
                       the first element of the tensor.
        stop (float): The upper bound of the interval. Must be less than or equal to
                       the last element of the tensor.

    Returns:
        out (tuple[int, int]): A pair of integers (left_idx, right_idx) where:
            - left_idx is the index of the largest value in 'a' that is less than or equal to 'start'
            - right_idx is the index of the smallest value in 'a' that is greater than or equal to 'stop'

    Raises:
        ValueError: If input tensor not monotonically increasing
        ValueError: If the interval [start, stop] is not contained within the tensor bounds
    """
    if torch.any(x.diff() < 0):
        raise ValueError('Input tensor must be monotonically increasing.')
    if not (start >= x[0] and stop <= x[-1]):
        raise ValueError(f"Interval ({start:+.2f}, {stop:+.2f}) out bounds input tensor ({x[0]:+.2f}, {x[-1]:+.2f}).")
    return _bisect_right(x, start) - 1, _bisect_left(x, stop)


def _upscale(x: Tensor, upscale_x: int, upscale_y: int) -> Tensor:
    """
    Oversamples a 2D tensor by repeating elements along the axes.

    Args:
        x (Tensor): Input 2D tensor.
        upscale_x (int): Upscaling factor along the x-axis.
        upscale_y (int): Upscaling factor along the y-axis.

    Returns:
        output (Tensor): Oversampled tensor with the same device and dtype.

    Notes:
        - the total sum is NOT conserved. Hence the function name is somewhat
          off, since there is no "scaling". A better name would be `enlarge` or
          similar. However, we used it for naming variables and parameters in
          many places so we are keeping it, for now.
    """
    for dim, factor in enumerate((upscale_y, upscale_x)):
        x = torch.repeat_interleave(x, factor, dim=dim)
    return x


def _shift(x: Tensor, rows: int, cols: int) -> Tensor:
    """
    Performs a 2D integer shift of a tensor using slicing.
    Areas shifted in from outside the frame are filled with zeros.

    Args:
        arr (Tensor): Input 2D tensor to be shifted.
        rows (int): Shift value along vertical axis.
        cols (int): Shift value along horizontal axis.

    Returns:
        output (Tensor): Shifted tensor with same shape, device, and dtype.
    
    Raises:
        ValueError: If shift values are not integer.

    Examples:
        >>> arr = torch.tensor([[1, 2], [3, 4]])
        >>> _shift(arr, 1, 0)   # Shift down by 1
        tensor([[0, 0],
                [1, 2]])
        >>> _shift(arr, 0, -1)  # Shift left by 1
        tensor([[2, 0],
                [4, 0]])
    """
    if not (isinstance(rows, int) and isinstance(cols, int)):
        raise ValueError('Shift values must be integers.')
    
    # zero-shift
    if rows == 0 and cols == 0:
        return x.clone()
    
    n, m = x.shape
    # avoid memory overload
    if abs(rows) >= n or abs(cols) >= m:
        return torch.zeros_like(x)
    
    arr_ = torch.zeros_like(x)
    arr_ystart, arr_yend = max(0, -rows), n - max(0, rows)
    arr_xstart, arr_xend = max(0, -cols), m - max(0, cols)

    y_start, y_end = max(0, rows), n + min(0, rows)
    x_start, x_end = max(0, cols), m + min(0, cols)

    arr_[y_start : y_end, x_start : x_end] = (
        x[arr_ystart : arr_yend, arr_xstart : arr_xend]
    )
    return arr_


def _correlate(
    a: Tensor,
    b: Tensor,
    mode: Literal['full', 'same'] = 'full',
    method: Literal['auto', 'direct', 'fft'] = 'auto',
) -> Tensor:
    """
    Performs cross-correlation operation between input tensors, following
    `scipy.signal.correlate` approach. Assumes input tensors to be 2D or 4D shaped.
    This exists to perform CAI basic operations between images.

    This version allows for gradient back-propagation when used in a training loop,
    separating the gradient computation of the two input tensors (e.g., if
    `a.requires_grad == False` and `b.requires_grad == True`, `out` will be linked
    to the gradient graph while `a` won't be part of it.)

    NOTE: if 'auto', the operation method is chosen based on the input tensors size,
          avoiding memory overflow and/or high computational costs.
    NOTE: the computation based on 'fft' may be less precise than the 'direct' one.  
    """
    _supp_ndim = [2, 4]
    _supp_mode = ['full', 'same']
    _supp_method = ['auto', 'direct', 'fft']
    
    if (a.ndim not in _supp_ndim) or (b.ndim not in _supp_ndim):
        raise ValueError(f'Input tensors must be 2D or 4D, got {a.ndim}D and {b.ndim}D.')
    if (a.ndim == 4 and a.shape[1] != 1) or (b.ndim == 4 and b.shape[1] != 1):
        raise ValueError(f'Only mono-channel tensors are supported.')
    if mode not in _supp_mode:
        raise ValueError(f"Invalid mode '{mode}', choose between {_supp_mode}.")
    if method not in _supp_method:
        raise ValueError(f"Invalid method '{method}', choose between {_supp_method}.")

    # as of now, method choice is linked to tensors size
    if method == 'auto':
        method = 'direct' if a.numel() * b.numel() < 500 * 500 * 10 * 10 else 'fft'

    # cast to float64 to avoid precision loss
    a_cc, b_cc = map(lambda x: x.to(torch.float64), (a, b))

    if method == 'direct':
        adapt_to_4d = lambda x: x[None, None, ...] if x.ndim == 2 else x
        a_cc, b_cc = map(adapt_to_4d, (a_cc, b_cc))
        h_a, w_a = a_cc.shape[-2:]
        h_b, w_b = b_cc.shape[-2:]
        # use `conv2d` because it actually applies cross-correlation
        # https://stackoverflow.com/questions/42970009/performing-convolution-not-cross-correlation-in-pytorch
        out = F.conv2d(a_cc, b_cc, padding=(h_b - 1, w_b - 1))
        out = out[0, 0] if a.ndim == 2 else out
    else:
        h_a, w_a = a_cc.shape[-2:]
        h_b, w_b = b_cc.shape[-2:]
        out_h = h_a + h_b - 1
        out_w = w_a + w_b - 1
    
        b_cc_flip = torch.flip(b_cc, dims=(-2, -1))
        A, B = map(lambda x: torch.fft.rfft2(x, s=(out_h, out_w)), (a_cc, b_cc_flip))
        out = torch.fft.irfft2(A * B, s=(out_h, out_w))
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == 'same':
        hstart, wstart = (h_b - 1) // 2, (w_b - 1) // 2
        out = out[..., hstart : hstart + h_a, wstart : wstart + w_a]

    return out


@dataclass(frozen=True)
class CodedMaskSpecs:
    """Camera geometry specifics container."""
    detector_minx: float
    detector_maxx: float
    detector_miny: float
    detector_maxy: float
    mask_deltax: float
    mask_deltay: float
    mask_thickness: float
    mask_minx: float
    mask_miny: float
    mask_maxx: float
    mask_maxy: float
    slit_deltax: float
    slit_deltay: float
    # The mask-detector distance can be defined in several ways:
    # - Distance between detector top and mask bottom
    # - Distance between detector top and mask top
    # - Distance between detector top and mask midpoint
    # The key requirement is consistency: whichever definition is used here
    # must match the correction applied in vignetting (see comment in
    # `apply_vignetting`).
    # We define the distance as the separation between detector top and mask top.
    # This choice is empirically motivated: testing showed this definition yields
    # the best results, though we don't fully understand why. Note that this
    # differs from the data convention, where mask-detector distance refers to
    # the separation between detector top and mask bottom.
    mask_detector_distance: float


@dataclass(frozen=True)
class CodedMaskCamera:
    """
    Dataclass containing a coded mask camera system.

    Handles mask pattern, detector geometry, and related calculations for coded mask imaging.
    Uses callable thunks for lazy loading of mask data to maintain hashability and performance.

    Args:
        get_mask: Callable that returns mask pattern as 2D array
        get_decoder: Callable that returns decoder pattern as 2D array
        get_bulk: Callable that returns bulk pattern as 2D array
        specs: CodedMaskSpecs containing geometric parameters and dimensions
        upscale_f: Tuple of upscaling factors for x and y dimensions
        hide_bulk_els_x: Detector physical elements to hide along fine axis, default=`0.0` [mm].
        hide_bulk_els_y: Detector physical elements to hide along coarse axis, default=`0.0` [mm].
        device: Device to allocate bulk, mask, decoder, balancing tensors, default=`'cpu'`.

    Raises:
        ValueError: If detector plane is larger than mask or if upscale factors are not positive
    """
    # NOTE: we are taking thunks here, rather than the actual array. Two reasons for this:
    #   1. python functions are hashable and using thunks keeps camera objects hashable themselves
    #   2. reading data may take time, thunk delays this expensive operation
    # NOTE: the conversion of the bulk/mask/decoder patterns to `torch` is done directly on the
    #       respective arrays, so to leave intact the synergy between astropy and numpy during
    #       loading from FITS files; tensor binning is built directly in `torch`
    # NOTE: now we have to select a device to initialise bulk, mask, decoder, balancing tensors
    get_mask: Callable[[], NDArray]
    get_decoder: Callable[[], NDArray]
    get_bulk: Callable[[], NDArray]
    specs: CodedMaskSpecs
    upscale_x: int = 1
    upscale_y: int = 1
    hide_bulk_els_x: int | float = 0.0
    hide_bulk_els_y: int | float = 0.0
    device: str | torch.device = 'cpu'

    def to(self, device: str | torch.device) -> "CodedMaskCamera":
        """
        Replaces instance updating the device. Reassignes bulk, mask,
        decoder or balancing tensors if already computed once.
        """
        if str(device) == str(self.device):
            return self
        
        # creates new instance with updated device
        new_cam = replace(self, device=device)

        # reassign any tensor already computed
        for attr in ('bulk', 'mask', 'decoder', 'balancing'):
            if attr in self.__dict__:
                new_cam.__dict__[attr]  = self.__dict__[attr].to(device)

        return new_cam

    @cached_property
    def upscale_f(self):
        return UpscaleFactor(x=self.upscale_x, y=self.upscale_y)

    @cached_property
    def shape_detector(self) -> tuple[int, int]:
        """Shape of the detector array (rows, columns)."""
        return len(self.bins_detector.y) - 1, len(self.bins_detector.x) - 1

    @cached_property
    def shape_mask(self) -> tuple[int, int]:
        """Shape of the mask array (rows, columns)."""
        return len(self.bins_mask.y) - 1, len(self.bins_mask.x) - 1

    @cached_property
    def shape_sky(self) -> tuple[int, int]:
        """Shape of the reconstructed sky image (rows, columns)."""
        n, m = self.shape_detector
        o, p = self.shape_mask
        return n + o - 1, m + p - 1
    
    def _mask_detector_artefacts(self, bulk: Tensor) -> Tensor:
        """
        Builds a mask for the detector plane edges, accounting for possible
        artefacts emerging from the reconstruction algorithm used to fit a
        detected photon position.\\
        Input `hide_bulk_els_x` and `hide_bulk_els_y` are expressed in [mm] and
        represent the physical size of the mask on top of the detector plane
        surface along the fine and coarse directions, starting from the boundaries.
        """
        if not any((self.hide_bulk_els_x, self.hide_bulk_els_y)):
            return torch.ones_like(bulk)
        print(f'UserInfo: Using bulk mask of [{self.hide_bulk_els_x} x {self.hide_bulk_els_y}] mm.')
        active_elements = (bulk > 0).to(dtype=bulk.dtype)
        npx_to_hide_x = int(self.hide_bulk_els_x * self.upscale_f.x / self.specs.mask_deltax)
        edges_cover_x = (
            (_shift(active_elements, 0, -npx_to_hide_x) > 0) & (_shift(active_elements, 0, npx_to_hide_x) > 0)
        )
        npx_to_hide_y = int(self.hide_bulk_els_y * self.upscale_f.y / self.specs.mask_deltay)
        edges_cover_y = (
            (_shift(active_elements, -npx_to_hide_y, 0) > 0) & (_shift(active_elements, npx_to_hide_y , 0) > 0)
        )
        return active_elements * edges_cover_x * edges_cover_y

    def _bins_mask(
        self,
        upscale_f: UpscaleFactor,
    ) -> BinsRectangular:
        """Returns bins for mask with given upscale factors."""
        l, r = self.specs.mask_minx, self.specs.mask_maxx
        b, t = self.specs.mask_miny, self.specs.mask_maxy
        xsteps = int((r - l) / (self.specs.mask_deltax / upscale_f.x)) + 1
        ysteps = int((t - b) / (self.specs.mask_deltay / upscale_f.y)) + 1
        return BinsRectangular(torch.linspace(l, r, xsteps), torch.linspace(b, t, ysteps))

    @cached_property
    def bins_mask(self) -> BinsRectangular:
        """Binning structure for the mask pattern."""
        return self._bins_mask(self.upscale_f)

    def _bins_detector(self, upscale_f: UpscaleFactor) -> BinsRectangular:
        """
        Returns bins for detector with given upscale factors.
        The detector bins are aligned to the mask bins.
        To guarantee this, we may need to extend the detector bin a bit over the mask.

         ◀────────────mask────────────▶
         │    │    │    │    │    │    │
         └────┴────┴────┴────┴────┴────┘
        -3   -2   -1    0    +1   +2   +3
              ┌─┬──┬────┬────┬──┬─┐
              │    │    │    │    │
                │               │
                ◀───detector────▶
                │               │
           detector_min   detector_max
        """
        bins = self._bins_mask(upscale_f)
        jmin, jmax = _bisect_interval(bins.x, self.specs.detector_minx, self.specs.detector_maxx)
        imin, imax = _bisect_interval(bins.y, self.specs.detector_miny, self.specs.detector_maxy)
        return BinsRectangular(self.bins_mask.x[jmin : jmax + 1], self.bins_mask.y[imin : imax + 1])

    @cached_property
    def bins_detector(self) -> BinsRectangular:
        """Binning structure for the detector."""
        return self._bins_detector(self.upscale_f)

    def _bins_sky(self, upscale_f: UpscaleFactor) -> BinsRectangular:
        """
        Returns bins for the reconstructed sky image.
        While the mask and detector bins are aligned, the sky-bins are not.

            │    │    │    │    │    │    │
            ◀────┴────┴──mask───┴────┴───▶┘
            0    1    2    3    4    5    6

                      │    │    │
                      ◀───det───▶
                      0    1    2

         │    │    │    │     │    │    │    │
         ◀────┴────┴────┴─sky─┴────┴────┴────▶
         0    1    2    3     4    5    6    7
        """
        binsd, binsm = self._bins_detector(upscale_f), self._bins_mask(upscale_f)
        xstep, ystep = binsm.x[1] - binsm.x[0], binsm.y[1] - binsm.y[0]
        return BinsRectangular(
            torch.linspace(
                binsd.x[0] + binsm.x[0] + xstep / 2,
                binsd.x[-1] + binsm.x[-1] - xstep / 2,
                self.shape_sky[1] + 1,
            ),
            torch.linspace(
                binsd.y[0] + binsm.y[0] + ystep / 2,
                binsd.y[-1] + binsm.y[-1] - ystep / 2,
                self.shape_sky[0] + 1,
            ),
        )

    @cached_property
    def bins_sky(self) -> BinsRectangular:
        """Returns bins for the sky-shift domain."""
        return self._bins_sky(self.upscale_f)

    @cached_property
    def mask(self) -> Tensor:
        """2D tensor representing the coded mask pattern."""
        mask = torch.from_numpy(self.get_mask())
        mask = _upscale(mask, *self.upscale_f)
        return mask.to(self.device)

    @cached_property
    def decoder(self) -> Tensor:
        """2D tensor representing the mask pattern used for decoding."""
        decoder = torch.from_numpy(self.get_decoder())
        decoder = _upscale(decoder, *self.upscale_f)
        return decoder.to(self.device)

    @cached_property
    def bulk(self) -> Tensor:
        """
        2D tensor representing the bulk (sensitivity) array of the mask.
        """
        bulk = torch.from_numpy(self.get_bulk())
        bulk[~torch.isclose(bulk, torch.zeros_like(bulk))] = 1
        bins = self._bins_mask(self.upscale_f)
        xmin, xmax = _bisect_interval(bins.x, self.specs.detector_minx, self.specs.detector_maxx)
        ymin, ymax = _bisect_interval(bins.y, self.specs.detector_miny, self.specs.detector_maxy)
        # why `xmin: xmax` rather than `xmin: xmax + 1`?
        # `bins.x[xmin:xmax + 1]` is the smallest subarray of `bins.x` spanning `det_minx` and `det_maxx`
        # the bin edges number of the subarray is `xmax - xmin + 1`.
        # the number of matrix elements in the subarray is `xmax - xmin + 1 - 1 == xmax - xmin`
        upscaled = _upscale(bulk, *self.upscale_f)[ymin:ymax, xmin:xmax]
        bulk_cover = self._mask_detector_artefacts(upscaled)
        return (upscaled * bulk_cover).to(self.device)

    @cached_property
    def balancing(self) -> Tensor:
        """2D tensor representing the correlation between decoder and bulk patterns."""
        return _correlate(self.decoder, self.bulk, mode='full', method='fft')


def codedmask(
    mask_filepath: str | Path,
    upscale_x: int = 1,
    upscale_y: int = 1,
    **kwargs,
) -> CodedMaskCamera:
    """
    Create a CodedMaskCamera from FITS file data.

    Loads mask patterns and specifications from a FITS file and creates a CodedMaskCamera
    instance with lazy-loaded tensor data.

    Args:
        mask_filepath (str | Path): Path to the mask FITS file.
        upscale_x (int): Upscaling factor for x direction (default: 1).
        upscale_y (int): Upscaling factor for y direction (default: 1).
        **kwargs: Argument keywords for the CodedMaskCamera object.

    Returns:
        CodedMaskCamera object containing mask patterns and specifications.

    Raises:
        ValueError: If detector plane is larger than mask or upscale factors are invalid
        NotImplementedError: If mask_filepath is not a valid FITS file
    """
    if validate_fits(mask_filepath):
        get_mask, get_decoder, get_bulk, specs_dict = load_from_fits(mask_filepath)
        specs = CodedMaskSpecs(**specs_dict)
        if not (
            specs.detector_minx >= specs.mask_minx and
            specs.detector_maxx <= specs.mask_maxx and
            specs.detector_miny >= specs.mask_miny and
            specs.detector_maxy <= specs.mask_maxy
        ):
            raise ValueError("Detector plane is larger than mask.")

        if not (
            (isinstance(upscale_x, int) and upscale_x > 0) and (isinstance(upscale_y, int) and upscale_y > 0)
        ):
            raise ValueError("Upscale factors must be positive integers.")

        return CodedMaskCamera(
            get_mask=get_mask,
            get_decoder=get_decoder,
            get_bulk=get_bulk,
            specs=specs,
            upscale_x=upscale_x,
            upscale_y=upscale_y,
            **kwargs,
        )
    raise NotImplementedError("Only reading masks from fits file is supported.")

"""
            ⠀⠀⣠⡶⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
            ⠀⣰⣿⠃⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀
            ⢸⣿⣯⠀⠀⠀⠀⠀⠀⢠⣴⣿⣿⣿⣿⣦⠀⠀⠀⠀⠀
            ⢼⣿⣿⣆⠀⢀⣀⣀⣴⣿⣿⣿⠋⠀⠀⠀⠀⠀⠀⠀⠀
            ⢸⣿⣿⣿⣿⣿⣿⣿⠿⠿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀
            ⠀⢻⣿⠋⠙⢿⣿⣿⡀⠀⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀
            ⠀⢸⠿⢆⣀⣼⣿⣿⣿⣿⡏⠀⢹⠀⠀⠀⠀⠀⠀⠀⠀
            ⠀⠀⡀⣨⡙⠟⣩⣙⣡⣬⣴⣤⠏⠀⠀⠀⠀⠀⠀⣀⡀
            ⠀⠀⠙⠿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⣀⣤⣾⣿⣿⡇
            ⠀⠀⠀⠀⠀⢀⣿⣿⣿⣿⣿⣿⣇⠀⢸⣿⣿⠿⠿⠛⠃
            ⠀⠀⠀⠀⢠⣿⣿⢹⣿⢹⣿⣿⣿⢰⣿⠿⠃⠀⠀⠀⠀
            ⠀⢀⣀⣤⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡛⠀⠀⠀⠀⠀⠀
            ⠀⠻⠿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠿⠛⠓⠀⠀⠀⠀⠀
            ⠀⠀⠀⠀⠀⠀⠀⠉⠀⠉⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
"""

def decode(camera: CodedMaskCamera, detector: Tensor) -> Tensor:
    """
    Reconstructs balanced sky image from detector counts using cross-correlation.

    Args:
        camera (CodedMaskCamera): Instance containing mask and decoder patterns.
        detector (Tensor): 2D tensor of detector counts.

    Returns:
        out (Tensor): Balanced cross-correlation sky image.
    """
    if detector.ndim != 2:
        raise ValueError(f'Invalid detector shape {detector.shape}, must be 2D (H, W).')
    cc = _correlate(camera.decoder, detector, mode='full', method='fft')
    sum_det, sum_bulk = map(torch.sum, (detector, camera.bulk))
    cc_bal = cc - camera.balancing * sum_det / sum_bulk
    return cc_bal


def decode_batch(camera: CodedMaskCamera, detector: Tensor) -> Tensor:
    """
    Reconstructs balanced sky image from detectors batch using cross-correlation.

    Args:
        camera (CodedMaskCamera): Instance containing mask and decoder patterns.
        detector (Tensor): 4D tensor of detector counts.

    Returns:
        out (Tensor): Balanced cross-correlation sky image batch.
    """
    if detector.ndim != 4:
        raise ValueError(f'Invalid detector shape {detector.shape}, must be 4D (B, C, H, W).')
    dec, bal = map(
        lambda x: x[None, None, ...].expand(detector.shape[0], 1, -1, -1), (camera.decoder, camera.balancing),
    )
    cc = _correlate(dec, detector, mode='full', method='fft')
    sum_det, sum_bulk = detector.sum(dim=(-2, -1), keepdim=True), camera.bulk.sum()
    cc_bal = cc - bal * sum_det / sum_bulk
    return cc_bal


def solid_angle() -> None:
    """"""
    raise NotImplementedError('To be imported from `bloodmoon`.')


def solid_angle_profile() -> None:
    """"""
    raise NotImplementedError('To be imported from `bloodmoon`.')


def variance() -> None:
    """"""
    raise NotImplementedError('To be imported from `bloodmoon`.')


def snratio() -> None:
    """"""
    raise NotImplementedError('To be imported from `bloodmoon`.')

"""
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢀⣴⣾⣿⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀            
⠀⠀⠀⠀⢀⣴⠿⢟⣛⣩⣤⣶⣶⣶⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀             ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⠀⠀⢀⣴⣿⠿⠸⣿⣿⣿⣿⣿⣿⡿⢿⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀            ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⠀⢠⠞⠉⠀⠀⠀⣿⠋⠻⣿⣿⣿⠀⣦⣿⠏⠀⠀⠀⢀⣀⣀⣀⣀⣀⠀⠀          ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⢠⠏⠀⠀⠀⠀⠀⠻⣤⣷⣿⣿⣿⣶⢟⣁⣒⣒⡋⠉⠉⠁⠀⠀⠀⠈⠉⡧        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⢻⡀⠀⠀⠀⠀⠀⣀⡤⠌⢙⣛⣛⣵⣿⣿⡛⠛⠿⠃⠀⠀⠀⠀⠀⢀⡜⠁            ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⠀⠉⠙⠒⠒⠛⠉⠁⠀⠸⠛⠉⠉⣿⣿⣿⣿⣦⣄⠀⠀⠀⢀⣠⠞⠁⠀⠀        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣿⣿⡿⣿⣿⣷⡄⠞⠋⠀⠀⠀⠀⠀        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣷⡻⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀            ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢨⣑⡙⠻⠿⠿⠈⠙⣿⣧⠀⠀⠀⠀⠀⠀          ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣷⡀⠀⠀⠀⠀⢹⣿⣆⠀⠀⠀⠀⠀          ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⡇⠀⠀⠀⠀⠸⣿⣿⡄⠀⠀⠀⠀         ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⠀⡿⣿⣿⠀⠀⠀⠀              ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠀⠀⠀⠀⠀            ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
                                           ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
                                            ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠛⠻⠁⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
                                               ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⠀⡀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
                                          ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠙⣿⣿⣿⠂⡀⠊⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
                                           ⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⠂⠈⠩⢛⠊⠀⠘⠒⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
                                          ⣿⣿⣿⣿⡿⢛⡙⠻⢷⣤⣷⣤⣴⣶⣴⣄⠀⠀⢀⠘⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
                                              ⣿⠁⣾⣿⣧⠈⠻⣿⣿⣿⣿⣿⣿⣦⠀⠀⢻⣶⣄⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
                                               ⡆⢻⣿⣿⣧⠀⠙⢿⣿⣿⣿⠟⠁⠀⡐⠐⠛⢿⣷⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
                                               ⣷⡈⢻⣿⣿⣇⠀⠀⠙⠛⠃⠀⠀⠀⡁⠀⠀⠈⣷⣸⢆⢙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
                                              ⣿⣿⣷⡀⠈⠻⢿⣧⡀⠀⢸⠀⠀⠀⠀⠀⠀⠀⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
                                               ⣿⣿⣿⣆⠀⠀⢻⣿⣦⡜⠀⣀⣀⣠⣤⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
                                              ⣿⣿⣿⣿⣿⣿⣶⣿⣿⠟⢀⣼⣿⣿⣿⡟⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
                                               ⣿⣿⣿⣿⣿⣿⣿⡟⠀⢼⣿⣿⣿⣿⣉⣄⠈⠻⠿⡿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
                                             ⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⢀⠀⣿⣿⣿⣿⣿⣿⣷⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
                                            ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
"""

def argmax(x: Tensor) -> tuple[int, int]:
    """
    Finds indices of maximum value along spatial dimension.

    Args:
        x (Tensor): Input 2D tensor to search.

    Returns:
        out (tuple[int, int]): Indices of maximum value.
    """
    i, j = map(int, torch.unravel_index(x.argmax(), x.shape))
    return i, j


def find_boxmax(
    x: Tensor,
    centre: tuple[int, int],
    boxsize: tuple[int, int] = (5, 5),
) -> tuple[tuple[int, int], int | float]:
    """
    Finds the argmax within given box of the input tensor.
    Box is resized if exceeds tensor spatial boundaries.

    Args:
        x (Tensor):
            Input tensor with any shape.
        centre (tuple[int, int]):
            Centre position to place the box.
        boxsize (tuple[int, int], optional (default=`(5, 5)`)):
            Size of the box to search for max value and its position.
    
    Returns:
        argmax (tuple[int, int]): Max indexes along spatial dimension.
        max (int | float): Max value within given box.
    """
    a, b = x.shape
    p, q = centre
    u, v = boxsize
    rcut = slice(max(0, p - u), min(p + u + 1, a))
    ccut = slice(max(0, q - v), min(q + v + 1, b))
    n, m = argmax(x[..., rcut, ccut])
    pos = (max(0, p - u) + n, max(0, q - v) + m)
    val = x[..., *pos]
    return pos, val


def crop(
    image: Tensor,
    pos: tuple[int, int],
    crp: tuple[int, int],
    strict: bool = True,
) -> Tensor:
    """
    Crops tensor spatial dimensions at given position and with given cropping.

    Args:
        image (Tensor):
            Input tensor to crop, 2D or higher.
        pos (tuple[int, int]):
            Centre position (y, x) for cropping.
        crp (tuple[int, int]):
            Size of the cropping along (y, x).
        strict (bool, optional (default=`True`)):
            If `False` allows for the cropping to be adapted wrt the tensor edges
            when they are exceeded.
    
    Returns:
        output (Tensor):
            Cropped tensor. The cut is performed by centering the cropped tensor,
            so that the final shape is `2 * crp + 1` along the two axes.
    
    Raises:
        ValueError: If `crp` is not a positive int tuple
        IndexError: If `crp` wrt indexes exceeds edges (only if `strict` is `True`)
    
    ## Notes:
        - Negative indexes for `pos` are allowed.
    """
    *_, h, w = image.shape

    crp_y, crp_x = crp
    if (crp_y < 0) or (crp_x < 0):
        raise ValueError("Crop size must be a tuple of positive integers.")

    y = pos[0] + h if pos[0] < 0 else pos[0]
    x = pos[1] + w if pos[1] < 0 else pos[1]

    boundary_y = ((0 <= y - crp_y) and (y + crp_y < h))
    boundary_x = ((0 <= x - crp_x) and (x + crp_x < w))
    if not (boundary_y and boundary_x):
        msg = f'Crop size {crp} at pos {pos} exceeds tensor edges'
        if not strict:
            crp_y, crp_x = min(y, h - y - 1), min(x, w - x - 1)
            print(f'UserInfo: {msg}, new cropping: {crp_y, crp_x}.')
        else:
            raise IndexError(msg + '.')

    return image[..., y - crp_y : y + crp_y + 1, x - crp_x : x + crp_x + 1]


def crop_batch(
    image: Tensor,
    pos: Tensor | tuple[Tensor, Tensor],
    crp: tuple[int, int],
) -> Tensor:
    """
    Crops tensor spatial dimensions at given position and with given cropping.

    This vectorised version pairs the `i-th` position with `i-th` batch item,
    keeping a uniform crop size (to fit every cropped sub-tensor within a
    tensor with same batch, channel dims than `image`, and to allow for fully
    vectorised GPU indexing and gradient flowing).

    Because this func assumes uniform cropping size, there is no adaptation
    wrt to the spatial dim edges (see `strict` arg in `crop()` func, above.)

    Args:
        image (Tensor):
            Input tensor to crop, 2D or higher.
        pos (Tensor | tuple[Tensor, Tensor]):
            Centre position (y, x) for cropping, can be a [B, 2] tensor or a
            tuple `(yy, xx)` with tensor idxs of shape [B,].
        crp (tuple[int, int]):
            Size of the cropping along (y, x).
    
    Returns:
        output (Tensor):
            Cropped tensor. The cut is performed by centering the cropped tensor,
            so that the final shape is `2 * crp + 1` along the spatial dims.
    
    Raises:
        ValueError: If `crp` is not a positive int tuple
        ValueError: If centre positions idxs are not positive
        IndexError: If `crp` wrt indexes exceeds spatial dim edges
    
    ## Notes:
        - Differently from `crop()`, negative idxs for `pos` are NOT allowed.
    """
    device = image.device
    b, c, h, w = image.shape

    crp_y, crp_x = crp
    if crp_y < 0 or crp_x < 0:
        raise ValueError("Crop size must be a tuple of positive integers.")

    yy, xx = pos if isinstance(pos, tuple) else (pos[:, 0], pos[:, 1])
    if (yy < 0).any() or (xx < 0).any():
        raise ValueError("Centre positions idxs must be positive.")
    
    boundary_y = (yy - crp_y >= 0) & (yy + crp_y < h)
    boundary_x = (xx - crp_x >= 0) & (xx + crp_x < w)
    if not (boundary_y.all() and boundary_x.all()):
        raise IndexError("Crop size exceeds tensor edges for one or more items in batch.")

    # create crop grid
    rows = torch.arange(-crp_y, crp_y + 1, device=device)
    cols = torch.arange(-crp_x, crp_x + 1, device=device)
    rows, cols = torch.meshgrid(rows, cols, indexing='ij')
    rows = yy.view(b, 1, 1, 1) + rows
    cols = xx.view(b, 1, 1, 1) + cols
    # define explicitly batch/channel indxs to satisfy torch broadcasting
    # rules, which is more efficient wrt creating a mask
    b_idxs = torch.arange(b, device=device).view(b, 1, 1, 1)
    c_idxs = torch.arange(c, device=device).view(1, c, 1, 1)

    return image[b_idxs, c_idxs, rows, cols]

"""
            ⠀⠀⠀⠀⠀⣠⣴⣾⣶⣿⣿⣶⣶⣶⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
            ⠀⠀⠀⣠⣼⣿⣿⣿⣋⣠⣿⣿⣿⣿⣿⡖⢶⣶⣤⣤⣀⣤⣶⣿⣷⡤⠶⢦⡀⠀
            ⠀⠀⣤⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀⠘⢿⣿⣿⣿⡿⢁⣾⡀⠀⠀⣷⠀
            ⠀⠀⠻⣿⠟⠛⠛⠻⠟⠛⠋⢹⣿⣿⣿⢣⡆⠀⠈⠛⠛⠋⠀⢻⣿⣿⣶⠟⣻⣦
            ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⢇⣾⠃⠀⣠⣤⣴⣤⣤⣀⠉⠙⢁⣴⡿⠁
            ⠀⠀⠀⠀⠀⢠⠀⠀⠀⢰⣤⣿⣿⢋⣬⡄⢀⣾⣿⣿⣿⣿⣿⣿⣧⠀⢿⣿⠀⠀
            ⠀⠀⠀⠀⠀⣠⠻⠿⠿⠿⠿⣛⣵⣿⣿⣧⢸⣿⣿⣿⣿⣿⣿⣿⣿⣄⣼⣿⡇⠀
            ⠀⠀⠀⠀⣠⣿⡀⢸⣿⣿⣿⣿⣿⣿⣿⠿⠆⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀
            ⢀⣀⣴⣾⣿⣿⡇⣬⣭⣭⣭⣭⣭⣶⣶⣿⣷⡄⢈⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀
            ⠰⢾⣿⣿⣿⣿⡇⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⢐⣛⡻⣿⣿⣿⣿⣿⣿⠻⣿⣿⠀
            ⠀⠁⠀⣠⣶⣿⣷⢸⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⠋⡵⠿⢿⣿⣿⣿⢟⣄⢹⡏⠀
            ⠀⠀⣰⣿⣿⣿⣿⣆⢲⣶⣶⣶⣶⣶⣶⣶⣿⢇⣷⣾⣿⡏⣟⣯⣶⣿⣿⡾⠀⠀
            ⠀⠀⣿⣿⣿⣿⣿⣿⣦⠹⣿⣿⣿⣿⣿⣿⣿⡜⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡶⠋
            ⠀⠀⠘⣿⣿⣿⣿⣿⣿⣷⣬⠉⠿⣛⣻⣿⣯⣥⣹⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀
            ⠀⣠⣶⣿⣿⣿⣿⣿⣿⠿⠿⠦⠄⠀⠀⠉⠉⠉⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀
            ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⠿⠛⠿⣿⠟⠛⠛⠀⠀
"""

def shift2pos(
    camera: CodedMaskCamera,
    shift_x: float | Tensor,
    shift_y: float | Tensor,
) -> tuple[int, int] | tuple[Tensor, Tensor]:
    """
    Convert continuous sky-shift coordinates to nearest discrete pixel indices.

    Args:
        camera (CodedMaskCamera): Instance containing binning information.
        shift_x (float | Tensor): x-coordinate in sky-shift space (mm).
        shift_y (float | Tensor): y-coordinate in sky-shift space (mm).

    Returns:
        out (tuple[int, int] | tuple[Tensor, Tensor]): Sky image grid (row, column) idxs.
    """
    return (
        _bisect_right(camera.bins_sky.y, shift_y) - 1,
        _bisect_right(camera.bins_sky.x, shift_x) - 1,
    )


# end