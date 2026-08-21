"""
Script for joint diffusion dataset handling.
"""

import math
from pathlib import Path
import pickle
import random
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
)
import warnings

from tqdm import tqdm

import torch
from torch.types import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

from .training import set_default


__all__ = []


class SrcDiffusionDataset(Dataset):
    """
    Defines dataset for LEM-X sources shadowgrams and parameters joint-diffusion.

    The dataset is composed of:
        * shadowgrams: tensor `[N, C, H_sg, W_sg]` containing the sources
          ground-truth detector images;
        * src_params: tensor `[N, 3]` containing the sources ground-truth
          parameters (camera local-frame coords, total collected photons);
        * psfs: tensor `[N, C, H_c, W_c]` containing the sources point-spread
          function (SPSF) heat-maps, acting as img conditioning;
        * ext_params: tensor `[N, 3]` containing the sources extracted parameters
          (SPSF peak coords + peak counts), acting as img conditioning.
    
    The data is distributed as tuples containing (shadowgrams, src_params) for the
    signal, (psfs, ext_params) representing diffusion conditioning.
    """
    def __init__(
        self,
        shadowgrams: Tensor,
        src_params: Tensor,
        psfs: Tensor,
        ext_params: Tensor,
    ) -> None:
        super().__init__()
        self.n = shadowgrams.shape[0]
        self.shadowgrams = shadowgrams
        self.src_params = src_params
        self.psfs = psfs
        self.ext_params = ext_params

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        x = (self.shadowgrams[idx], self.src_params[idx])
        condition = (self.psfs[idx], self.ext_params[idx])
        return (x, condition)


def gather_data_filepaths(dirpath: str | Path, pattern: str, shuffle: bool = False) -> list[str]:
    """Groups all the data file-paths (of the given format) inside the specified directory."""
    dirpath_ = Path(dirpath)
    paths_list: list[str] = [str(path) for path in dirpath_.glob(pattern)]
    if shuffle: random.shuffle(paths_list)
    return paths_list


def load_pickle(filepath: str | Path, **kwargs) -> Any:
    """Loads data from `pickle` file."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Input file {filepath} does not exists.")
    print('Loading...')
    with open(filepath, "rb") as handle:
        data = pickle.load(handle, **kwargs)
    print('Data loaded!')
    return data


def safe_load(filepath: str | Path, msg: str = '', lvl: Literal['warn', 'err'] = 'warn', **kwargs) -> Any:
    """Loads data from `.pt` file, handling eventual loading failures."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Input file {filepath} does not exists.")

    print('Loading...')
    try:
        data = torch.load(filepath, weights_only=False, **kwargs)
        print('Data loaded!')
        return data
    except Exception as e:
        err_msg = msg if msg else str(e)
        if lvl == 'warn':
            warnings.warn(f'Failed to load {filepath}: {err_msg}')
            return None

        raise type(e) from e
    

def normalise_sgs(img: Tensor, footprint: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Normalises source shadowgrams in dataset through z-score.

    The shadowgrams contain the collected photon distribution from a
    given source, so the normalisation is performed by subtracting
    the `mean` and dividing by the `std` of the data.

    Mean and std are computed only wrt the detected photons, NOT over
    the whole array (i.e., where `footprint > 0`), with `footprint`
    being the projected mask pattern layout onto the detector plane.

    Args:
        img (Tensor): Input shadowgrams with shape `[N, C, H_sg, W_sg]`.
        footprint (Tensor): Shadowgrams projection footprint `[N, C, H_sg, W_sg]`.
        eps (float, optional (default=`1e-6`)): Tolerance for counts std.
    
    Returns:
        out (Tensor): Normalised source shadowgram images.
    """ 
    mask = (footprint > 0).float()
    nels = mask.sum(dim=(-2, -1), keepdim=True)

    mu = img.sum(dim=(-2, -1), keepdim=True) / nels
    std = torch.square(mask * (img - mu))
    std = std.sum(dim=(-2, -1), keepdim=True) / nels
    std = std.sqrt().clamp(min=eps)

    return mask * (img - mu) / std


def normalise_psfs(img: Tensor, var: Tensor, max_snr: float = 2e3, eps: float = 1e-6) -> Tensor:
    """
    Normalises source point-spread function (SPSF) profiles in dataset
    through log-SNR transform using input SPSF variance profiles.

    Args:
        img (Tensor):
            Input SPSFs with shape `[N, C, H_c, W_c]`.
        var (Tensor):
            SPSF variance profiles for SNR computation, `[N, C, H_c, W_c]`.
        max_snr (float, optional (default=`2e3`)):
            Max value for log-SNR normalisation.
        eps (float, optional (default=`1e-6`)):
            Tolerance for SPSF counts and variance values.
    
    Returns:
        out (Tensor): Normalised SPSF images.
    """
    img, var = map(lambda x: x.clamp(min=eps), (img, var))
    return torch.log1p(img / torch.sqrt(var)) / math.log1p(max_snr)


def symm_norm(x: Tensor, edge: float, amax: float = 1.0) -> Tensor:
    """Normalises tensor from physical bounds `[-edge, edge]` to `[-amax, amax]`."""
    return abs(amax / edge) * x


def inverse_symm_norm(x: Tensor, edge: float, amax: float = 1.0) -> Tensor:
    """Inverse tensor normalisation from `[-amax, amax]` back to `[-edge, edge]`."""
    return abs(edge / amax) * x


def log_snr_norm(x: Tensor, var: Tensor, max_snr: float = 2e3) -> Tensor:
    """Normalises photon counts `x` through log-SNR transform using variance `var`."""
    return torch.log1p(x / torch.sqrt(var)) / math.log1p(max_snr)


def inverse_log_snr_norm(x: Tensor, var: Tensor, max_snr: float = 2e3) -> Tensor:
    """Inverse normalized log-SNR back to physical photon counts."""
    cts = torch.sqrt(var) * torch.expm1(math.log1p(max_snr) * x)
    return cts.clamp(min=0.0)


def normalise_params(params: Tensor, camdata: dict[str, Any], variances: Tensor) -> Tensor:
    """
    Normalises source params in dataset.

    The source coordinates are normalised in [-1, 1] mirroring the LEM-X cameras
    FoV, expressed in the instruments local-frame system.
    The collected photons are normalised by applying a log-transform to the SNR,
    using the detector instrumental variance and normalising in [0, 1] through
    a max SNR-value (default to `SNRmax = 2e3`).

    Args:
        params (Tensor): Parameter container (coords + counts), `[N, 3]`.
        camdata (dict[str, Any]): Container for LEM-X cameras specifics.
        variances (Tensor): Intrumental variances at source PSF peaks.
            
    Returns:
        out (Tensor): Normalised source parameters array.
    """
    out = torch.empty_like(params)
    # extract camera FoV specs from digital binning (centered wrt pixel idxs), used for camera local-frame
    # coordinates normalisation in [-1, 1]
    # camera specs @ https://github.com/peppedilillo/bloodmoon/blob/b9301449f252550343cb9815dc03e2ad19901c59/bloodmoon/mask.py#L66
    # sky arr binning @ https://github.com/peppedilillo/bloodmoon/blob/b9301449f252550343cb9815dc03e2ad19901c59/bloodmoon/mask.py#L110)
    pxdim_x, pxdim_y = camdata['specs']['mask_deltax'], camdata['specs']['mask_deltay']
    upx, upy = camdata['upsampling'].values()
    skyarr_bins_x, skyarr_bins_y = camdata['bins']['sky']
    edge_x, edge_y = (
        abs(skyarr_bins_x[0] - 0.5 * pxdim_x / upx),
        abs(skyarr_bins_y[0] - 0.5 * pxdim_y / upy),
    )
    out[:, 0] = symm_norm(params[:, 0], edge_x)
    out[:, 1] = symm_norm(params[:, 1], edge_y)
    out[:, 2] = log_snr_norm(params[:, 2], variances)
    return out


def get_dataset(
    dirpath: str | Path,
    handle_chs: Optional[Callable[[Tensor], Tensor]] = None,
) -> SrcDiffusionDataset:
    """
    Generates Dataset object containing data for LEM-X sources joint-diffusion.

    Args:
        dirpath (str | Path):
            Path to directory with dataset files.
        handle_chs (Callable, optional (default=`None`)):
            Callable for tensors channel dim handling. If None, dataset
            tensors will be broadcasted to have one channel.
    
    Returns:
        out (SrcDiffusionDataset): Dataset for LEM-X sources joint-diffusion.
    """
    # sources shadowgrams `[N, H_sg, W_sg]` and ground-truth params `[N, 3]`
    sgs_list: list[Tensor] = []
    gtpars_list: list[Tensor] = []
    # sources PSFs `[N, H_c, W_c]` and extracted params `[N, 3]`
    psfs_list: list[Tensor] = []
    extpars_list: list[Tensor] = []
    # shadowgram footprints, instrumental variance profiles `[N, H_c, W_c]`
    # and relative value at sources PSF peaks `[N,]` and camera specs
    sg_fps_list: list[Tensor] = []
    psfvars_list: list[Tensor] = []
    extvars_list: list[Tensor] = []

    camdata: Optional[dict[str, Any]] = None

    # glob dataset files and store data from dataset chunks
    # data structure @ /SparkLab/IROSDiffusion/gen_IROSdiffusion_dataset.py
    pathslist = gather_data_filepaths(dirpath, 'irosdiffsn_dataset*.pt')
    if not pathslist:
        raise FileNotFoundError(f'No files matching pattern in {dirpath}.')

    to_tensor32f = lambda x: torch.as_tensor(x, dtype=torch.float32)
    
    for idx, dspath in tqdm(enumerate(pathslist), desc='Loading Data'):
        
        # safe load dataset with warning, if None go to next iter
        data: Optional[dict[str, Any]] = safe_load(dspath)
        if data is None:
            continue

        sgs, gtpars = map(to_tensor32f, (tuple(data['data'].values())))
        sgs_list.append(sgs)
        gtpars_list.append(gtpars)

        psfs, extpars = map(to_tensor32f, (tuple(data['conditioning'].values())))
        psfs_list.append(psfs)
        extpars_list.append(extpars)

        sg_fps, psfvars, extvars = map(to_tensor32f, (tuple(data['info'].values())))
        sg_fps_list.append(sg_fps)
        psfvars_list.append(psfvars)
        extvars_list.append(extvars)

        is_last = idx == len(pathslist) - 1
        if is_last:
            camdata = data['camera']

    # concatenate data from different dataset
    sgs, gtpars, psfs, extpars, sg_fps, psfvars, extvars = map(
        lambda x: torch.cat(x, dim=0),
        (sgs_list, gtpars_list, psfs_list, extpars_list, sg_fps_list, psfvars_list, extvars_list),
    )

    # handle img tensors channel dim (default: 1 channel)
    get_channels = set_default(handle_chs, lambda x: x.unsqueeze(dim=1))
    sgs, sg_fps, psfs, psfvars = map(get_channels, (sgs, sg_fps, psfs, psfvars))

    # normalise data
    sgs = normalise_sgs(sgs, sg_fps)
    psfs = normalise_psfs(psfs, psfvars)
    gtpars = normalise_params(gtpars, camdata, extvars)
    extpars = normalise_params(extpars, camdata, extvars)

    return SrcDiffusionDataset(sgs, gtpars, psfs, extpars)


def get_dataloaders(
    dataset: Dataset,
    batch_size: int,
    valid_size: float = 0.0,
    **kwargs,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Generates DataLoader containers for training and validation subset from given Dataset obj.

    Args:
        dataset (Dataset): Input Dataset obj with data.
        batch_size (int): Data batch lenght.
        valid_size (float, optional (default=0.0)): Validation dataset size wrt total data.
    
    Returns:
        out (tuple[DataLoader, Optional[DataLoader]]):
            * DataLoader obj for training dataset
            * DataLoader obj for validation dataset (if `valid_size > 0`)
    """
    if valid_size and not (0.0 < valid_size < 1.0):
        raise ValueError(f"Invalid 'valid_size' value {valid_size}, must be in [0, 1).")
    
    print('Baking DataLoaders...')    
    if valid_size > 0:
        vlen = int(valid_size * len(dataset))
        tlen = len(dataset) - vlen
        train, valid = random_split(dataset, [tlen, vlen])
        train_dl = DataLoader(train, batch_size, shuffle=True, **kwargs)
        valid_dl = DataLoader(valid, batch_size, shuffle=False, **kwargs)
    else:
        train_dl = DataLoader(dataset, batch_size, shuffle=True, **kwargs)
        valid_dl = None
    print('DataLoaders ready-to-go!')

    return train_dl, valid_dl


# end