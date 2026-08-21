"""
Dataset Generation for IROS Diffusion.
    * [paper]    ...
    * [git repo] https://github.com/EdoardoGiancarli/SparkLab/tree/main


- The whole dataset is composed of:
    * *data* (`dict[str, NDArray]`):
        - array `[N, H_sg, W_sg]` representing the $N$ source shadowgrams acting as ground-truth
          images for the joint-diffusion process;
        - array `[N, 3]` representing the respective $N$ source parameters (local-frame coords,
          collected photons) acting as ground-truth parameters for the joint-diffusion process;

    * *conditioning* (`dict[str, NDArray]`):
        - array `[N, H_spsf, W_spsf]` representing the $N$ source PSF heat-maps acting as images
          conditioning for the joint-diffusion process;
        - array `[N, 3]` representing the respective $N$ source PSF parameters (extracted
          local-frame coords at peak, peak counts) acting as parameters conditioning for the
          joint-diffusion process;
    
    * *info* (`dict[str, Any]`):
        - array `[N, H_sg, W_sg]` with the $N$ source shadowgram footprints, needed for centering;
        - array `[N, H_spsf, W_spsf]` with instrumental SPSF variance profiles, needed for
          SPSFs normalisation during data pre-processing;
        - array `[N,]` with instrumental variance values at each SPSF peak, needed for peak
          counts normalisation during data pre-processing;

    * *camera*  (`dict[str, Any]`):
        - dict container with main objects and info on a LEM-X coded-mask camera, including:
            * `dict[str, int | float]` containing each camera geometry specifics;
            * `dict[str, int` containing imaging sampling factors along the axes;
            * `dict[str, tuple[NDArray, NDArray]]` containing detector, mask and sky arrays binning structure;
            * `dict[str, NDArray]` containing detector sensitivity, mask, decoder and sky balancing arrays;
            * `dict[str, tuple[int, int]]` containing detector, mask and sky array shapes.

- Currently, the imaging is performed at upsampling $(\\text{up}_{fine}, \\text{up}_{coarse}) = 2, 1$ to
  limit the computational cost (eventually consider higher upsamplings for improved discretisation and
  resolution, coupled with external `DownSampler`s and `UpSampler`s for diffusion interface).

- The tests on the IROS Diffusion framework are carried out starting from an ideal instrumental setup, analysing
  how the joint-diffusion process responds to each instrumental effect and sequentially introducing realisms:

    * **\\[current\\]** *1st TEST*: full-ideal camera, with thin mask plate (no vignetting), infinite detector
                                    spatial resolution (no counts dispersion) and infinite detector density
                                    (no inclined penetration from off-axis photons);

    * *2nd TEST*: semi-ideal camera, with thick mask plate (yes vignetting), infinite detector spatial resolution
                  (no counts dispersion) and infinite detector density (no inclined penetration from off-axis photons);

    * *3rd TEST*: semi-ideal camera, with thick mask plate (yes vignetting), finite detector spatial resolution
                  (yes counts dispersion) but infinite detector density (no inclined penetration from off-axis photons);

    * *4th TEST*: real camera, with thick mask plate (yes vignetting), finite detector spatial resolution (yes counts
                  dispersion) and finite detector density (yes inclined penetration from off-axis photons);
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from tqdm import tqdm
import torch

from irosbm.coords import angle2shift, shift2pos, pos2shift
from irosbm.images import argmax
from irosbm.mask import CodedMaskCamera, codedmask
from irosbm.mask import decode, variance
from irosbm.optim import model_shadowgram

# - helpers
def get_dirpaths() -> tuple[str, str]:
    """Returns dirpath with mask file and where to save generated datasets, based on OS."""
    iswin = Path('/mnt/d').is_dir()
    if iswin:
        return (
            '/mnt/d/PhD_AASS/Coding/Images_fits',
            '/mnt/d/PhD_AASS/Coding/Images_fits',
        )
    return (
        '/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data/Simulations',
        '/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Diffusion/IROSdiffusion_datasets'
    )

@dataclass
class Dataset:
    """
    Data container for:
        - actual data passed to model:
            * ground-truth signal (shadowgrams `[N, H_s, W_s]`, params `[N, 3]`)
            * conditioning (SPSFs `[N, H_c, W_c]`, init params `[N, 3]`)
        
        - data needed during dataset normalisation and/or signal modulation:
            * shadowgrams footprint `[N, H_s, W_s]`
            * SPSFs variance profiles `[N, H_c, W_c]`
            * SPSFs peak variance values `[N,]`
    """
    # model data
    sgs: NDArray
    gt_params: NDArray
    psfs: NDArray
    init_params: NDArray
    # norm + modulation data
    sg_footprints: NDArray
    psfvars: NDArray
    peaks_var: NDArray

def find_boxmax(arr: NDArray, centre: tuple[int, int], boxsize: int | tuple = 5) -> tuple[int, int]:
    """Finds the argmax of given arr in a box centered around `pos`."""
    box = (boxsize, boxsize) if isinstance(boxsize, int) else boxsize
    rcut, ccut = map(lambda idx, b: slice(idx - b, idx + b + 1), centre, box)
    by, bx = box
    p, q = centre
    n, m = argmax(arr[rcut, ccut])
    return p - by + n, q - bx + m

# - data
def gen_coords(camera: CodedMaskCamera, n: int, rng: Optional[Generator] = None) -> NDArray:
    """Generates `n` random camera local-frame coords in `84° x 84°` FoV."""
    if rng is None: rng = np.random.default_rng()
    lim = angle2shift(camera, 42)
    return rng.uniform(-lim, lim, (n, 2))

def generate_data(
    camera: CodedMaskCamera,
    coords: NDArray,
    rates: NDArray,
    vignetting: bool | Callable,
    psfy: bool | Callable,
    crp: tuple[int, int] = (38, 6),
    rng: Optional[Generator] = None,
) -> Dataset:
    """Generates dataset."""
    if rng is None:
        rng = np.random.default_rng()

    n = coords.shape[0]
    cy, cx = crp

    sgs = np.empty((n, *camera.shape_detector), dtype=np.int16)
    gt_params = np.empty((n, 3), dtype=np.float32)

    psfs = np.empty((n, 2 * cy + 1, 2 * cx + 1), dtype=np.float32)
    init_params = np.empty((n, 3), dtype=np.float32)

    sg_fps = np.empty((n, *camera.shape_detector), dtype=np.float16)
    psfvars = np.empty((n, 2 * cy + 1, 2 * cx + 1), dtype=np.float32)
    peaks_var = np.empty(n, dtype=np.float32)

    loop = tqdm(enumerate(zip(rates, coords)), total=n, desc='Generating Data')
    for idx, (r, crd) in loop:
        sx, sy = crd
        sg_fp = model_shadowgram(camera, sx, sy, vignetting, psfy, normalise=False)
        sg = sg_fp * rng.poisson(r, size=camera.shape_detector)

        sky = decode(camera, sg)
        varmap = variance(camera, sg)
        src_pos = shift2pos(camera, sx, sy)
        y, x = find_boxmax(sky, src_pos, (10, 5))

        sgs[idx] = sg
        gt_params[idx] = (sx, sy, sg.sum())
        psfs[idx] = sky[y - cy : y + cy + 1, x - cx : x + cx + 1]
        init_params[idx] = (*pos2shift(camera, y, x), sky[y, x])

        sg_fps[idx] = sg_fp
        psfvars[idx] = varmap[y - cy : y + cy + 1, x - cx : x + cx + 1]
        peaks_var[idx] = varmap[y, x]

    ds = Dataset(
        sgs=sgs, gt_params=gt_params, psfs=psfs, init_params=init_params,
        sg_footprints=sg_fps, psfvars=psfvars, peaks_var=peaks_var,
    )
    return ds

# - dataset handling
def gather_dataset(camera: CodedMaskCamera, data: Dataset) -> dict[str, Any]:
    """
    Gather all objs being part of the IROS Diffusion dataset.
    """
    out: dict[str, Any] = {}

    out['data'] = {'imgs': data.sgs, 'params': data.gt_params}
    out['conditioning'] = {'imgs': data.psfs, 'params': data.init_params}
    out['info'] = {
        'data_imgs_footprints': data.sg_footprints,
        'psfvar_img': data.psfvars,
        'peakvar_arr': data.peaks_var,
    }
    
    out['camera'] = {
        'specs': camera.specs.__dict__,
        'upsampling': {
            'ups_fine': camera.upscale_x,
            'ups_coarse': camera.upscale_y,
        },
        'arrays': {
            'mask': camera.mask,
            'decoder': camera.decoder,
            'bulk': camera.bulk,
            'balancing': camera.balancing,
        },
        'data_shape': {
            'detector': camera.shape_detector,
            'mask': camera.shape_mask,
            'sky': camera.shape_sky,
        },
        'bins': {
            'detector': (camera.bins_detector.x, camera.bins_detector.y),
            'mask': (camera.bins_mask.x, camera.bins_mask.y),
            'sky': (camera.bins_sky.x, camera.bins_sky.y),
        }
    }
    return out

def save_dataset(data: dict[str, Any], save_to: str | Path, overwrite: bool = False, **kwargs) -> None:
    """Saves dataset in `.pt` format."""
    if Path(save_to).exists() and not overwrite:
        print('Dataset already saved!')
        return
    print('Saving...')
    torch.save(data, save_to, **kwargs)
    print('Dataset saved!')
    return

def generate_in_step(
    camera: CodedMaskCamera,
    batches: int | tuple[int],
    save_to_dirpath: str | Path,
    start_from_ID: int = 1,
    rmin: float = 0.1,
    rmax: float = 10.0,
    vignetting: bool | Callable = False,
    psfy: bool | Callable = False,
    psf_crp: tuple[int, int] = (38, 6),
    rng: Optional[Generator] = None,
) -> None:
    """
    Generates multiple datasets with `batches` elements in `len(batches)` steps.
    This implementantion is used to save memory.
    """
    if rng is None:
        rng = np.random.default_rng()

    batches = (batches,) if isinstance(batches, int) else batches

    for step, batch in enumerate(batches):
        print(f'\n# -------------- STEP: {step}, ELEMENTS: {batch} --------------')
        # generate ground-truth source coords + poisson noise rates
        coords: NDArray = gen_coords(camera, batch, rng)
        rates: NDArray = rng.uniform(rmin, rmax, batch)
        # generate data
        data: Dataset = generate_data(camera, coords, rates, vignetting, psfy, psf_crp, rng)
        # gather data to make dataset and save
        save_to: str = f"{save_to_dirpath}/irosdiffsn_dataset_ID{start_from_ID + step}_nels{batch}.pt"
        dataset: dict[str, Any] = gather_dataset(camera, data)
        save_dataset(dataset, save_to)

    return





def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--batches",
        nargs='*',
        type=int,
        help="Number of arrays to generate at each batch. Pass as sequence of integers, e.g.: 100 2000 500 ...",
    )
    parser.add_argument(
        "--start_from_ID",
        type=int,
        default=1,
        help="Determines start ID number for output dataset file (default: %(default)s).",
    )
    args = parser.parse_args()


    MASKPATH, DSPATH = get_dirpaths()

    # get coded-mask camera specs
    MASK_PATH: str = f"{MASKPATH}/mask_NTHT_20260129_CORRECTED.fits"
    UPS_X, UPS_Y = 2, 1

    VIGNETTING: bool = False
    PSFY: bool = False

    wfm: CodedMaskCamera = codedmask(MASK_PATH, UPS_X, UPS_Y)

    # - for this test (full-ideal camera), the source shadowgrams will be generated manually by approximating sources emission
    #   with Poisson noise, and then multiplying for the shifted mask pattern to perform the projection onto the detector plane.
    # - this configuration allows to avoid momentarily ad-hoc observations performed with the WISEMAN simulator.
    #   NOTE: this approximation is valid only in the first case as I'm assuming a full-ideal setup; the shadowgram significancy
    #         is tested in `dmIROS_dataset.ipynb`
    rnd_gen: Generator = np.random.default_rng()

    # config dataset generation
    batches: int | tuple[int, int] = tuple(args.batches)
    start_from_ID: int = args.start_from_ID

    # generate data
    generate_in_step(
        camera=wfm,
        batches=batches,
        save_to_dirpath=DSPATH,
        start_from_ID=start_from_ID,
        vignetting=VIGNETTING,
        psfy=PSFY,
        rng=rnd_gen,
    )
    return


if __name__ == '__main__':
    main()


# end