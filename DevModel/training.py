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
from .sampling import Sampler


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
    sampler: Sampler
    loss: Callable
    optimiser: Callable
    lr_scheduler: Callable
    device: Optional[str | torch.device]


def config_training(
    model: nn.Module,
    sampler: Sampler,
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
            f'  - model parameters: {n_params:,}\n'
        )
    tp = TrainParams(model, sampler, loss, optimiser, lr_scheduler, device)
    return tp


def log_with_wandb(logger: Run, data: dict[str, Any], epoch: int) -> None:
    """Logs input data using `wandb`."""
    try:
        logger.log(data, step=epoch)
    except Exception as e:
        warnings.warn(f'wandb.log failed @ E = {epoch}: {e}')


def train_model(
    params: TrainParams,
    epochs: int,
    learning_rate: float,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    wandb_logger: Run,
    **model_kws,
) -> TrainResults:
    """
    Basic train routine.
    """
    # setup model/optimiser/scaler for memory saving
    device = params.device
    device_type = device.type if isinstance(device, torch.device) else str(device).split(':')[0]

    model = params.model.to(device)
    sampler = params.sampler.to(device)
    loss_fn = params.loss

    tot_pars_to_opt = (
        list(model.parameters()) + list(loss_fn.parameters())
        if hasattr(loss_fn, 'parameters') and any(p.requires_grad for p in loss_fn.parameters())
        else model.parameters()
    )
    optimiser = params.optimiser(tot_pars_to_opt, lr=learning_rate)
    scheduler = params.lr_scheduler(optimiser)
    scaler = GradScaler(device_type)

    # config procedure/loss container
    tdl_len, vdl_len = map(len, (train_dl, valid_dl))
    ntsteps = len(sampler.sqrt_alphas_cumprod)
    avg_train_loss, avg_valid_loss = [], []

    # training loop
    loop = tqdm(range(epochs))
    for epoch in loop:
        
        # ---------------------------   TRAINING   ---------------------------
        model.train()
        loop.set_description('Training Model')
        running_batches = 0
        running_train_loss = 0.0

        for batch, (x, condition) in enumerate(islice(train_dl, tdl_len)):
            loop.set_postfix({'batch': f'{batch + 1}/{tdl_len}'})

            x_img, x_pars = map(lambda m: m.to(device), x)
            c_img, c_pars = map(lambda m: m.to(device), condition)
            # # NOTE: if DataLoaders yield 5D tensors for imgs/pars
            # x_img, x_pars = map(lambda m: m.squeeze(0).to(device), x.chunk(2, dim=0))
            # c_img, c_pars = map(lambda m: m.squeeze(0).to(device), condition.chunk(2, dim=0))

            # sample `t` uniformally for every entry in the batch and add noise to x
            t = torch.randint(0, ntsteps, (x_img.shape[0],), device=device).long()
            img_noise, pars_noise = torch.randn_like(x_img), torch.randn_like(x_pars)
            
            x_img, x_pars = map(
                lambda data, noise: sampler.q_sample(data, t, noise),
                (x_img, x_pars), (img_noise, pars_noise),
            )

            optimiser.zero_grad()
            with autocast(device_type=device_type):
                pred_img, pred_pars = model(x_img, x_pars, t, c_img, c_pars, **model_kws)
                tot_loss_val, *_ = loss_fn(pred_img, img_noise, pred_pars, pars_noise)

            scaler.scale(tot_loss_val).backward()
            scaler.step(optimiser)
            scaler.update()

            if not tot_loss_val.isnan():
                running_batches += 1
                running_train_loss += tot_loss_val.item()
            else:
                warnings.warn(f'Train loss NaN @ E: {epoch + 1}, B: {batch + 1}')

        avg_train_loss.append(running_train_loss / max(running_batches, 1))


        # --------------------------   VALIDATION   --------------------------
        model.eval()
        loop.set_description('Validating Model')
        running_valid_batches = 0
        running_valid_loss = 0.0

        with torch.no_grad():
            for batch, (x, condition) in enumerate(islice(valid_dl, vdl_len)):
                loop.set_postfix({'batch': f'{batch + 1}/{vdl_len}'})
    
                x_img, x_pars = map(lambda m: m.to(device), x)
                c_img, c_pars = map(lambda m: m.to(device), condition)
    
                # sample `t` uniformally for every entry in the batch and add noise to x
                t = torch.randint(0, ntsteps, (x_img.shape[0],), device=device).long()
                img_noise, pars_noise = torch.randn_like(x_img), torch.randn_like(x_pars)

                x_img, x_pars = map(
                    lambda data, noise: sampler.q_sample(data, t, noise),
                    (x_img, x_pars), (img_noise, pars_noise),
                )
    
                with autocast(device_type=device_type):
                    pred_img, pred_pars = model(x_img, x_pars, t, c_img, c_pars, **model_kws)
                    tot_loss_val, *_ = loss_fn(pred_img, img_noise, pred_pars, pars_noise)
    
                if not tot_loss_val.isnan():
                    running_valid_batches += 1
                    running_valid_loss += tot_loss_val.item()
                else:
                    warnings.warn(f'Train loss NaN @ E: {epoch + 1}, B: {batch + 1}')

        # loss logging and lr update
        avg_v_loss_val = running_valid_loss / max(running_valid_batches, 1)
        avg_valid_loss.append(avg_v_loss_val)
        try:
            scheduler.step(avg_v_loss_val)
        except AttributeError:
            scheduler.step()

        
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