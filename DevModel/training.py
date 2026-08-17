"""
Script for U-Net and sources shadowgrams/parameters joint diffusion.
"""

from itertools import islice
from pathlib import Path
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    OrderedDict,
)
import warnings

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.types import Tensor
from torch.utils.data import DataLoader

from wandb import Run

import spark as pk

from .modules import exists
from .sampling import Sampler


__all__ = []


def set_default(val: Any, default: Any) -> Any:
    """Sets default value for input `val` in case it doesn't exist."""
    return val if exists(val) else default


class TrainParams(NamedTuple):
    """Container with training operations."""
    model: nn.Module
    sampler: Sampler
    loss: Callable
    optimiser: Callable
    lr_scheduler: Callable
    device: Optional[str | torch.device]


class TrainResults(NamedTuple):
    """
    Model trainer results. Contains the average
    train and valid loss values from training.
    """
    train_loss: Tensor
    valid_loss: Tensor


class CheckPointManager:
    """
    Object for model checkpoints managing. `CheckPointManager` accounts for:
        * saving model checkpoints (condition to be defined outside in the training loop);
        * saving best model via validation/training loss comparison, accounting for model
          overfitting and validation loss trend;
        * interrupting training due to not improving loss or possible model overfitting.
    """
    def __init__(
        self,
        log_ckpnt_every: int,
        savepath: str | Path,
        checkpointID: int = 0,
        log_bestmodel_every: Optional[int] = None,
        patience_bestmodel: Optional[int] = 25,
        check_training_every: Optional[int] = None,
        patience_train: Optional[int] = 50,
    ) -> None:
        self.savepath = Path(savepath)
        self.savepath.mkdir(parents=True, exist_ok=True)

        self.log_ckpnt_every = log_ckpnt_every
        self.checkpointID = checkpointID

        self.log_bestmodel_every = log_bestmodel_every
        self.patience_bestmodel = patience_bestmodel
        self.check_training_every = check_training_every
        self.patience_train = patience_train

        self.best_valid_loss_val: float = float('inf')

    def reset_checkpointID(self, value: int = 0) -> None:
        self.checkpointID = value

    def save_checkpoint(
        self,
        state_dict: OrderedDict,
        name: Optional[str] = None,
        info: Optional[dict[str, Any]] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        """Saves model checkpoint with given info in specified directory."""
        name_ = name if exists(name) else f'model_checkpnt_{self.checkpointID}.pt'
        pk.save_model(
            state_dict=state_dict,
            save_to=f'{self.savepath}/{name_}',
            info=info,
            overwrite=overwrite,
            **kwargs,
        )
        self.checkpointID += 1
        return

    def _is_model_overfitting(self, train_loss: Tensor, valid_loss: Tensor, eps: float) -> bool:
        """Checks model overfitting by train/valid losses comparison."""
        # model is overfitting if valid significantly higher than train
        return (valid_loss - train_loss).mean().item() > abs(eps)

    def save_best_model(
        self,
        train_loss: Tensor,
        valid_loss: Tensor,
        tol_overfit: Optional[float] = None,
        sigma_overfit: Optional[float] = 1.0,
        tol_validloss: Optional[float] = None,
        sigma_validloss: Optional[float] = 1.0,
        patience: Optional[int] = None,
        compare_fn: Optional[Callable[[Tensor, Tensor], bool]] = None,
    ) -> bool:
        """Checks model performance by analysing overfitting and valid loss."""
        if exists(compare_fn):
            return compare_fn(train_loss, valid_loss)

        p = set_default(patience, self.patience_bestmodel)
        p = max(1, min(p, len(train_loss) // 2 - 1))
        t, v = train_loss[-p:], valid_loss[-p:]

        # check overfitting
        eps = set_default(tol_overfit, sigma_overfit * t.std().item())
        is_overfit = self._is_model_overfitting(t, v, eps)

        # check valid loss by comparing mean val in last `p` iters wrt prev
        eps = set_default(tol_validloss, sigma_validloss * v.std().item())
        win_vmean = v.mean().item()
        is_better_globally: bool = win_vmean < self.best_valid_loss_val
        is_better_in_win: bool = valid_loss[-2 * p : -p].mean().item() - win_vmean > abs(eps)

        is_new_best = is_better_in_win and is_better_globally and not is_overfit

        if is_new_best:
            self.best_valid_loss_val = win_vmean

        return is_new_best

    def _is_loss_plateauing(self, loss: Tensor, eps: float) -> bool:
        """Checks if train/valid losses have reached a plateau."""
        return (loss.mean() - loss.min()).item() < abs(eps)

    def interrupt_training(
        self,
        train_loss: Tensor,
        valid_loss: Tensor,
        tol_overfit: Optional[float] = None,
        sigma_overfit: Optional[float] = 1.0,
        patience: Optional[int] = None,
        compare_fn: Optional[Callable[[Tensor, Tensor], bool]] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Checks if training can be interrupted, e.g. because loss is not improving
        over number of epochs despite adaptive learning rate and/or overfitting.
        """
        if exists(compare_fn):
            return compare_fn(train_loss, valid_loss)

        p = set_default(abs(patience), self.patience_train)
        t, v = train_loss[-p:], valid_loss[-p:]

        eps = set_default(tol_overfit, sigma_overfit * t.std().item())
        is_overfit = self._is_model_overfitting(t, v, eps)

        is_plateaued = self._is_loss_plateauing(t, eps) and self._is_loss_plateauing(v, v.std().item())

        if is_plateaued:
            return True, f'Train/Valid losses plateaued for {patience} epochs.'

        if is_overfit:
            return True, f'Possible model overfitting detected.'

        return False, None


def execute_every(epoch: int, step: Optional[int]) -> bool:
    """Returns boole flag for `epoch` reaching given `step`."""
    if exists(step):
        return epoch % step == 0
    return False


def log_with_wandb(logger: Run, data: dict[str, Any], epoch: int) -> None:
    """Logs input data using `wandb`."""
    try:
        logger.log(data, step=epoch)
    except Exception as e:
        warnings.warn(f'wandb.log failed @ E = {epoch}: {e}')


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


def train_model(
    params: TrainParams,
    epochs: int,
    learning_rate: float,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    wandb_logger: Run,
    ckpnt_manager: Optional[CheckPointManager] = None,
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
    stop_training: bool = False
    stop_msg: Optional[str] = None

    tdl_len, vdl_len = map(len, (train_dl, valid_dl))
    ntsteps = len(sampler.sqrt_alphas_cumprod)
    avg_train_loss, avg_valid_loss = [], []

    # training loop
    loop = tqdm(range(1, epochs + 1))
    for epoch in loop:

        if stop_training:
            print(
                f'\n[Early Train Stop] Interrupting training @ E = {epoch}/{epochs} | {stop_msg}\n'
            )
            break
        
        # ---------------------------   TRAINING   ---------------------------
        model.train()
        loop.set_description('Training Model')
        running_batches = 0
        running_train_loss = 0.0

        for batch, (x, condition) in enumerate(islice(train_dl, tdl_len), start=1):
            loop.set_postfix({'batch': f'{batch}/{tdl_len}'})

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
                warnings.warn(f'Train loss NaN @ E: {epoch}, B: {batch}')

        avg_train_loss.append(running_train_loss / max(running_batches, 1))


        # --------------------------   VALIDATION   --------------------------
        model.eval()
        loop.set_description('Validating Model')
        running_valid_batches = 0
        running_valid_loss = 0.0

        with torch.no_grad():
            for batch, (x, condition) in enumerate(islice(valid_dl, vdl_len), start=1):
                loop.set_postfix({'batch': f'{batch}/{vdl_len}'})
    
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
                    warnings.warn(f'Train loss NaN @ E: {epoch}, B: {batch}')

        # loss logging and lr update
        avg_v_loss_val = running_valid_loss / max(running_valid_batches, 1)
        avg_valid_loss.append(avg_v_loss_val)

        
        # -------------------------   WANDB LOGGING   ------------------------
        if exists(wandb_logger):
            log_with_wandb(
                logger=wandb_logger,
                data={
                    'train/loss': avg_train_loss[-1],
                    'train/lr': optimiser.param_groups[0]["lr"],
                    'valid/loss': avg_valid_loss[-1],
                },
                epoch=epoch,
            )

        # model checkpoint and lr update
        if exists(ckpnt_manager):
            if execute_every(epoch, ckpnt_manager.log_ckpnt_every):
                name = f'model_{model.__class__.__name__}_ckpnt{ckpnt_manager.checkpointID}_E{epoch}.pt'
                info = {
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'valid_loss': avg_valid_loss,
                    'lr': optimiser.param_groups[0]["lr"],
                }
                ckpnt_manager.save_checkpoint(
                    state_dict=model.state_dict(),
                    name=name,
                    info=info,
                )
                print(f'\n[Model Checkpoint] New model checkpoint saved @ E = {epoch}/{epochs}!\n')

            if execute_every(epoch, ckpnt_manager.log_bestmodel_every):
                save_best = ckpnt_manager.save_best_model(
                    train_loss=torch.tensor(avg_train_loss),
                    valid_loss=torch.tensor(avg_valid_loss),    
                )
                if save_best:
                    name = f'bestmodel_{model.__class__.__name__}.pt'
                    info = {
                        'epoch': epoch,
                        'train_loss': avg_train_loss,
                        'valid_loss': avg_valid_loss,
                        'lr': optimiser.param_groups[0]["lr"],
                    }
                    ckpnt_manager.save_checkpoint(
                        state_dict=model.state_dict(),
                        name=name,
                        info=info,
                        overwrite=True,
                    )
                    print(f'\n[Best Model] New best model saved @ E = {epoch}/{epochs}!\n')

            if execute_every(epoch, ckpnt_manager.check_training_every):
                stop_training, stop_msg = ckpnt_manager.interrupt_training(
                    train_loss=torch.tensor(avg_train_loss),
                    valid_loss=torch.tensor(avg_valid_loss),    
                )

        try:
            scheduler.step(avg_v_loss_val)
        except TypeError:
            scheduler.step()
    
    return TrainResults(*tuple(map(torch.tensor, (avg_train_loss, avg_valid_loss))))




def execute_step(
    model: nn.Module,
    optimiser: Callable,
    scaler: Callable,
    loss_fn: Callable,
    sampler: Sampler,
    x: tuple[Tensor, Tensor],
    cond: tuple[Tensor, Tensor],
    ntsteps: int,
    device: str | torch.device,
    device_type: str,
    **model_kws,
) -> Tensor | tuple[Tensor, ...]:
    """Executes training/validation steps, returning output loss results."""
    # NOTE: think about integrating this code for the train/valid loops over respective datasets
    #       PRO:
    #           * simpler main training func structure (?)
    #           * same operations for train and valid steps
    #           * extracts all losses, to be handled in main training func (e.g., logging/analyses)
    #           * independent from Callables implementation (?) (NOTE: model/loss_fn must accept fixed args/kwargs)
    #       CONS:
    #           * assumes same operations kind for train and valid computations
    #           * may reduce readability of the main training func (?)
    #           * lots of args/kwargs (?)

    x_img, x_pars = map(lambda m: m.to(device), x)
    c_img, c_pars = map(lambda m: m.to(device), cond)
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

    if model.training: optimiser.zero_grad()
    with autocast(device_type=device_type):
        pred_img, pred_pars = model(x_img, x_pars, t, c_img, c_pars, **model_kws)
        loss_vals = loss_fn(pred_img, img_noise, pred_pars, pars_noise)

    if model.training: 
        scaler.scale(loss_vals[0]).backward()
        scaler.step(optimiser)
        scaler.update()

    return loss_vals


# end