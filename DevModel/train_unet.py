"""
Script for U-Net model training, with wandb logging.
"""

import argparse
from functools import partial
from pathlib import Path
import logging
from typing import Any, Optional
import json

from torch.utils.data import DataLoader
import torch.optim as opt
import wandb

import spark as pk

from pkdev.camera import CodedMaskCamera, codedmask
from pkdev.dataset import get_dataset, get_dataloaders
from pkdev.model import exists, Unet, JointDiffusionLoss
from pkdev.sampling import DPMSolverPP2MSampler
from pkdev.training import (
    TrainParams,
    TrainResults,
    CheckPointManager,
    config_training,
    train_model,
)

# helpers
def parse_args() -> argparse.Namespace:
    descr = """

    U-Net trainer for LEM-X sources joint-diffusion, with WnB logging.

    Required Args:
        * runID (str): WnB run ID.
        * epochs (int): Number of training epochs.
    
    """
    parser = argparse.ArgumentParser(
        description=descr,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    train_grp = parser.add_argument_group("Training Hyperparameters")        # -------------------------
    train_grp.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    train_grp.add_argument("--lr", type=float, default=5e-3, help="Initial learning rate (default: %(default)s).")
    train_grp.add_argument("--lr_patience", type=int, default=10, help="Epochs to wait before reducing learning rate (default: %(default)s).")
    train_grp.add_argument("--lr_factor", type=float, default=0.5, help="Learning rate reducing factor at plateau (default: %(default)s).")

    data_grp = parser.add_argument_group("Dataset Configuration")            # -------------------------
    data_grp.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation (default: %(default)s).")
    data_grp.add_argument("--valid_size", type=float, default=0.2, help="Fraction of data to use for validation (default: %(default)s).")
    data_grp.add_argument("--datasetID", type=str, default='srcIROS_jointdiffusion.pt', help="Dataset ID (default: %(default)s).")
    data_grp.add_argument(
        "--reshape_sgs_to", type=int, nargs=2, default=None, help="Shadowgrams reshaping dims (`[H, W]`) (default: %(default)s)."
    )
    data_grp.add_argument("--reshape_mode", type=str, default='nearest', help="Shadowgrams interpolation mode (default: %(default)s).")

    diff_grp = parser.add_argument_group("Diffusion Noise Scheduler")        # -------------------------
    diff_grp.add_argument("--beta_start", type=float, default=1e-4, help="Noise schedule start value (default: %(default)s).")
    diff_grp.add_argument("--beta_end", type=float, default=0.02, help="Noise schedule end value (default: %(default)s).")
    diff_grp.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps (default: %(default)s).")

    data_grp = parser.add_argument_group("Coded-Mask Camera Configuration")  # -------------------------
    data_grp.add_argument(
        "--mask_pattern", type=str, default='mask_NTHT_20260129_CORRECTED.fits', help="Coded-mask file with camera specifics (default: %(default)s)."
    )
    data_grp.add_argument(
        "--wfm_upfine", type=int, default=2,
        help="Upsampling factor for digital binning structure along the camera FINE axis (default: %(default)s).",
    )
    data_grp.add_argument(
        "--wfm_upcoarse", type=int, default=1,
        help="Upsampling factor for digital binning structure along the camera COARSE axis (default: %(default)s).",
    )

    wnb_grp = parser.add_argument_group("WnB Logging")                       # -------------------------
    wnb_grp.add_argument("--project", type=str, default='Src-Joint-Diffusion', help="WnB project name (default: %(default)s).")
    wnb_grp.add_argument("--runID", type=str, required=True, help="WnB run ID.")
    wnb_grp.add_argument("--wnbkeypath", type=str, default=None, help="Path to JSON file containing WnB API key (default: %(default)s).")

    ckpt_grp = parser.add_argument_group("Checkpoint and Early Stopping")    # -------------------------
    ckpt_grp.add_argument(
        "--log_ckpnt_every", type=int, default=None, help="Epoch interval to save model checkpoint (if None defaults to `epochs // 10`).",
    )
    ckpt_grp.add_argument(
        "--log_bestmodel_every", type=int, default=None, help="Epoch interval to check and save best model (default: %(default)s).",
    )
    ckpt_grp.add_argument(
        "--patience_bestmodel",
        type=int,
        default=None,
        help="Window size for evaluating best model improvements (if None AND best model check enabled, defaults to `log_bestmodel_every // 2`).",
    )
    ckpt_grp.add_argument(
        "--check_training_every", type=int, default=None, help="Epoch interval to check for early stopping conditions (default: %(default)s).",
    )
    ckpt_grp.add_argument(
        "--patience_train",
        type=int,
        default=None,
        help="Patience for overall early stopping (if None AND model train check, defaults to `check_training_every // 2`).",
    )

    model_grp = parser.add_argument_group("Model Architecture Parameters")   # -------------------------
    model_grp.add_argument("--dim", type=int, default=8, help="Base dimension (default: %(default)s).")
    model_grp.add_argument(
        "--dim_mults",
        type=int,
        nargs='+',
        default=[1, 2, 4, 8],
        help="Dimension multipliers, also config arch depth as `len(dim_mults) - 1`. Pass values separated by spaces, default: %(default)s.",
    )
    model_grp.add_argument("--bottleneck_blocks", type=int, default=1, help="Number of bottleneck blocks (default: %(default)s).")
    model_grp.add_argument("--drop_cond_prob", type=float, default=0.5, help="Condition dropout probability (default: %(default)s).")
    model_grp.add_argument("--cond_dim", type=int, default=32, help="Conditioning dimension (default: %(default)s).")
    model_grp.add_argument("--attn_dim_head", type=int, default=16, help="Attention head dimension (default: %(default)s).")
    model_grp.add_argument("--attn_heads", type=int, default=4, help="Number of attention heads (default: %(default)s).")
    model_grp.add_argument("--convnext_mult", type=int, default=2, help="ConvNeXt dimension multiplier (default: %(default)s).")

    args = parser.parse_args()

    args.dim_mults = tuple(args.dim_mults)
    if len(args.dim_mults) == 0:
        args.dim_mults = (1,)

    if args.log_ckpnt_every is None:
        args.log_ckpnt_every = max(1, args.epochs // 10)

    if args.patience_bestmodel is None and exists(args.log_bestmodel_every):
        args.patience_bestmodel = max(1, args.log_bestmodel_every // 2)

    if args.patience_train is None and exists(args.check_training_every):
        args.patience_train = max(1, args.check_training_every // 2)
    
    return args

def _select_os() -> tuple[str, Path]:
    """Selects OS on which task are operated."""
    machine = {
        'win': Path('/mnt/d'),
        'deb': Path('/mnt/dbb8f47e-da06-47bf-8ef5-038092af70f7'),
        'quasar': Path('/home/egiancarli'),
    }
    root = next(((m, p) for m, p in machine.items() if p.is_dir()), None)
    
    if root is None:
        raise ValueError('A0, ma ndo sei finit*?')

    return root

def select_mask_dirpath() -> Path:
    """Returns filepath to coded-mask pattern file with camera specifics, based on OS."""
    paths = {
        'win': Path('PhD_AASS/Coding/Images_fits'),
        'deb': Path('Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data/Simulations'),
        'quasar': Path('lem-x/Coding/CameraFiles'),
    }
    os_, root = _select_os()
    dirpath = paths[os_]
    return root / dirpath

def select_dataset_checkpnt_dirpaths() -> tuple[Path, Path]:
    """Returns dirpaths to dataset(s) and model checkpoints, based on OS."""
    paths = {
        'win': Path('PhD_AASS/Coding/IROS_Diffusion'),
        'deb': Path('Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Diffusion'),
        'quasar': Path('lem-x/IROS_Diffusion'),
    }
    os_, root = _select_os()
    dirpath = paths[os_]
    return root / dirpath / 'SrcDiffusionDataset', root / dirpath / 'ModelChkPoints'


# `wandb` wrappers
def wnb_login(filepath: Optional[str | Path] = None, **kwargs) -> None:
    """Execute `wandb` login, optionally extracting API key from `.json` file."""
    key: Optional[str] = None

    if filepath is not None:
        with open(filepath, mode='r') as f:
            data = json.load(f)
        key = data.get('wnd_key')

    wandb.login(key=key, **kwargs)
    return

def run_w_wnb(
    project: str,
    runID: str,
    camera: CodedMaskCamera,
    params: TrainParams,
    epochs: int,
    learning_rate: float,
    dataloaders: tuple[DataLoader, Optional[DataLoader]],
    ckpnt_manager: Optional[CheckPointManager] = None,
    entity: Optional[str] = None,
    **model_kws: Any,
) -> TrainResults:
    """
    Performs model training with `wandb` logging.
    """
    train_dl, valid_dl = dataloaders
    wnb_factory = dict(
        entity=entity,
        project=project,
        name=runID,
        config={
            'device': params.device,
            'epochs': epochs,
            'dataset': 'JointDiffusionDataset',
            'batch_size': train_dl.batch_size,
            'lr': learning_rate,
            'architecture': 'U-Net',
        },
        save_code=False,
    )
    with wandb.init(**wnb_factory) as logger:
        try:
            results: TrainResults = train_model(
                camera=camera,
                params=params,
                epochs=epochs,
                learning_rate=learning_rate,
                train_dl=train_dl,
                valid_dl=valid_dl,
                ckpnt_manager=ckpnt_manager,
                wandb_logger=logger,
                **model_kws,
            )
        except Exception as e:
            print(f'\n\n[Train Failure] {e.__class__.__name__} hit during training :c\n\n')
            if ckpnt_manager is not None:
                ckpnt_manager.save_checkpoint(
                    state_dict=params.model.state_dict(),
                    name='model_checkpnt-last_updated.pt',
                    info={'runID': runID},
                )
            raise

    return results





def train():
    args = parse_args()
    maskpath = select_mask_dirpath()
    dspath, chkpntpath = select_dataset_checkpnt_dirpaths()

    # define coded-mask camera obj
    wfm = codedmask(
        maskpath / args.mask_pattern, upscale_x=args.wfm_upfine, upscale_y=args.wfm_upcoarse,
    )

    # config dataset + dataloaders
    ds_processed = dspath / f'processed/{args.datasetID}'
    if Path(ds_processed).is_file():
        dataset = pk.load_dataset(ds_processed)
    else:
        dataset = get_dataset(
            dirpath=dspath / 'raw',
            reshape_sgs_to=tuple(args.reshape_sgs_to),
            reshape_mode=args.reshape_mode,
        )
        pk.save_dataset(dataset, ds_processed)

    train_dl, valid_dl = get_dataloaders(dataset, args.batch_size, args.valid_size)

    # noise scheduler + define training params + CheckPointManager
    scheduler = pk.NoiseScheduler(args.timesteps)
    betas = scheduler.cosine()

    model: Unet = Unet(
        dim=args.dim,
        dim_mults=args.dim_mults,
        bottleneck_blocks=args.bottleneck_blocks,
        drop_cond_prob=args.drop_cond_prob,
        cond_dim=args.cond_dim,
        attn_dim_head=args.attn_dim_head,
        attn_heads=args.attn_heads,
        convnext_mult=args.convnext_mult,
    )
    tpars_factory = dict(
        model=model,
        sampler=DPMSolverPP2MSampler(betas, pred_type='v'),
        loss=JointDiffusionLoss(wfm),
        optimiser=opt.Adam,
        lr_scheduler=partial(
            opt.lr_scheduler.ReduceLROnPlateau, patience=args.lr_patience, factor=args.lr_factor,
        ),
        model_info={'project': args.project, 'runID': args.runID},
    )
    tpars = config_training(**tpars_factory)

    mng_factory = dict(
        log_ckpnt_every=args.log_ckpnt_every,
        savepath=chkpntpath,
        log_bestmodel_every=args.log_bestmodel_every,
        patience_bestmodel=args.patience_bestmodel,
        check_training_every=args.check_training_every,
        patience_train=args.patience_train,
    )
    chkpnt_mng = CheckPointManager(**mng_factory)

    # train model
    wnb_login(args.wnbkeypath)
    results: TrainResults = run_w_wnb(
        project=args.project,
        runID=args.runID,
        camera=wfm,
        params=tpars,
        epochs=args.epochs,
        learning_rate=args.lr,
        dataloaders=(train_dl, valid_dl),
        ckpnt_manager=chkpnt_mng,
    )

    # save trained model + results
    pk.save_model(
        state_dict=tpars.model.state_dict(),
        save_to=chkpntpath / f'../unet_jointdiffusion-{args.runID}.pt',
        info={
            'loss': {
                'train_loss': results.train_loss,
                'valid_loss': results.valid_loss,
            },
            'noise_schedule': {
                'stype': 'cosine',
                'beta_start': args.beta_start,
                'beta_end': args.beta_end,
                'timesteps': args.timesteps,
            },
            'setup': {
                'project': args.project,
                'runID': args.runID,

                'datasetID': args.datasetID,
                'batch_size': args.batch_size,
                'valid_size': args.valid_size,
                'epochs': args.epochs,
                'learning_rate': args.lr,
                'lr_patience': args.lr_patience,
                'lr_factor': args.lr_factor,
            
                'log_ckpnt_every': args.log_ckpnt_every,
                'log_bestmodel_every': args.log_bestmodel_every,
                'patience_bestmodel': args.patience_bestmodel,
                'check_training_every': args.check_training_every,
                'patience_train': args.patience_train,

                'camera_binning_upsampling': (wfm.upscale_x, wfm.upscale_y),
                'camera_bulk_artefact_mask': (wfm.hide_bulk_els_x, wfm.hide_bulk_els_y),
            },
        }
    )

    return




if __name__ == '__main__':
    train()


# end