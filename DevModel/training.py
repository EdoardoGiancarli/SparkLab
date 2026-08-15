"""
Script for shadowgram sampling at model inference.
"""

from itertools import islice
from typing import Any, Callable, NamedTuple, Optional
import warnings

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.types import Tensor
from torch.utils.data import DataLoader

from wandb import Run

from .modules import exists


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


class TrainResults(NamedTuple):
    """
    Model trainer results. Contains the average
    train and valid loss values from training.
    """
    train_loss: list[float]
    valid_loss: list[float]


class TrainParams(NamedTuple):
    """Container with training operations."""
    model: nn.Module
    loss: Callable
    optimiser: Callable
    lr_scheduler: Callable
    device: Optional[str | torch.device]


def config_training(
    model: nn.Module,
    loss: Callable,
    optimiser: Callable,
    lr_scheduler: Callable,
    device: Optional[str | torch.device] = None,
    verbose: bool = True,
) -> TrainParams:
    """
    Creates a container with training operations.
    Callables can be also passed as partially initialised.
    """
    if not exists(device):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f'Training INFO:\n'
            f'  - operating on device: {device}\n'
            f'  - model parameters: {n_params}\n'
        )

    return TrainParams(model, loss, optimiser, lr_scheduler, device)


def check_loss_val(loss_val: Tensor, msg: str) -> bool:
    """Checks the current loss value and raises a warning if is NaN."""
    if not loss_val.isnan():
        return True
    warnings.warn(msg)
    return False


def log_with_wandb(logger: Run, data: dict[str, Any], epoch: int) -> None:
    try:
        logger.log(data, step=epoch)
    except Exception as e:
        warnings.warn(f'wandb.log failed @ E = {epoch}: {e}')
    return


def train_model(
    params: TrainParams,
    epochs: int,
    learning_rate: float,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    wandb_logger: Run,
) -> TrainResults:
    """
    Basic train routine.
    """
    # config procedure/loss container
    avg_train_loss, avg_valid_loss = [], []
    tdl_len, vdl_len = map(len, (train_dl, valid_dl))

    # setup model/optimiser/scaler for memory saving
    device = params.device
    device_type = device.type if isinstance(device, torch.device) else device
    model = params.model.to(device)
    loss_fn = params.loss
    tot_pars_to_opt = (
        list(model.parameters()) + list(loss_fn.parameters()) if any(p.requires_grad for p in loss_fn.parameters())
        else model.parameters()
    )
    optimiser = params.optimiser(tot_pars_to_opt, lr=learning_rate)
    scheduler = params.scheduler(optimiser)
    scaler = GradScaler(device)

    # training loop
    loop = tqdm(range(epochs))
    for epoch in loop:
        
        # ---------------------------   TRAINING   ---------------------------
        loop.set_description('Training Model')
        model.train()
        running_batches = 0
        running_train_loss = 0.0


        # --------------------------   VALIDATION   --------------------------
        loop.set_description('Validating Model')
        model.eval()
        running_valid_batches = 0
        running_valid_loss = 0.0

        
        # -------------------------   WANDB LOGGING   ------------------------
        if exists(wandb_logger):
            log_with_wandb(
                logger=wandb_logger,
                data={
                    'train/loss': avg_train_loss[epoch],
                    'train/lr': optimiser.param_groups[0]["lr"],
                    'valid/loss': avg_valid_loss[epoch],
                },
                epoch=epoch,
            )
    
    return TrainResults(avg_train_loss, avg_valid_loss)


# end