"""
Script for joint diffusion dataset handling.
"""

import math
from pathlib import Path
import pickle
import random
from typing import Any, Callable, Optional

from tqdm import tqdm

import torch
from torch.types import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

from .training import set_default


__all__ = []


class IROSDiffusionDataset(Dataset):
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
    return abs(amax / edge) * x


def inverse_symm_norm(x: Tensor, edge: float, amax: float = 1.0) -> Tensor:
    """Inverse tensor normalisation from `[-amax, amax]` back to `[-edge, edge]`."""
    return abs(edge / amax) * x


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


def get_dataset(
    dirpath: str | Path,
    handle_chs: Optional[Callable[[Tensor], Tensor]] = None,
) -> IROSDiffusionDataset:
    """
    Generates Dataset object containing data for LEM-X sources joint-diffusion.

    Args:
        ...
    
    Returns:
        ...
    """
    sgs_list: list[Tensor] = []
    gtpars_list: list[Tensor] = []

    psfs_list: list[Tensor] = []
    extpars_list: list[Tensor] = []

    camdata: Optional[dict[str, Any]] = None

    # glob dataset files and store data
    pathslist = gather_data_filepaths(dirpath, 'irosdiffsn_dataset*.pickle')

    for idx, dspath in tqdm(enumerate(pathslist), desc='Loading Data'):
        is_last = idx == len(pathslist) - 1
        data: dict[str, Any] = load_pickle(dspath)

        sgs, gtpars = map(torch.tensor, (tuple(data['data'].values())))
        sgs_list.append(sgs)
        gtpars_list.append(gtpars)

        psfs, extpars = map(torch.tensor, (tuple(data['condition'].values())))
        psfs_list.append(psfs)
        extpars_list.append(extpars)

        if is_last: camdata = data['camera']

    # concatenate data from different dataset
    sgs, gtpars, psfs, extpars = map(
        lambda x: torch.cat(x, dim=0),
        (sgs_list, gtpars_list, psfs_list, extpars_list),
    )

    # handle img tensors channel dim (default: 1 channel)
    get_channels = set_default(handle_chs, lambda x: x.unsqueeze(dim=1))
    sgs, psfs = map(get_channels, (sgs, psfs))

    # normalise data
    # - images
    sgs = ...
    psfs = ...
    # - params
    gtpars = ...
    extpars = ...

    return IROSDiffusionDataset(sgs, gtpars, psfs, extpars)


def get_dataloaders(
    dataset: Dataset,
    batch_size: int,
    valid_size: float = 0.0,
    **kwargs,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Generates DataLoader containers for training and validation subset from given Dataset obj.

    Args:
        ...
    
    Returns:
        ...
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