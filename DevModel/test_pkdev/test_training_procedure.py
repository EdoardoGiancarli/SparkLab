"""
Test module for training procedure (spark's `training.py` module).
"""

from typing import Optional
import unittest
from unittest import TestCase

import torch
import torch.optim as opt
from torch.types import Tensor

from pkdev.camera import codedmask
from pkdev.dataset import SrcDiffusionDataset, get_dataloaders
from pkdev.model import Unet, JointDiffusionLoss
from pkdev.sampling import DPMSolverPP2MSampler
from pkdev.training import mask_sdd_zero_response_area, config_training, train_model

from spark import NoiseScheduler

from .assets import maskpath


def assert_tensor_allclose(residuals: Tensor, eps: float = 1e-8, msg: Optional[str] = None) -> None:
    torch.testing.assert_close(residuals, torch.zeros_like(residuals), atol=eps, rtol=0.0, msg=msg)
    return




class TestTraining(TestCase):
    """
    Test class for `spark`'s training procedure.
    """
    def setUp(self):
        self.wfm = codedmask(maskpath, 5, 2)
        self.wfm_epoch = codedmask(maskpath, 1, 1)

    def test_sdd_response_area_correction(self):
        """Tests the `mask_sdd_zero_response_area` func."""
        zero_response = ~(self.wfm.bulk > 0).to(torch.bool)

        # test single detector image
        detector = torch.randint(1, 20, self.wfm.shape_detector, dtype=torch.float32)
        detector = mask_sdd_zero_response_area(self.wfm, detector)

        self.assertTrue(detector.ndim == 2)
        assert_tensor_allclose(detector[zero_response] - self.wfm.bulk[zero_response])

        # test detector images batch
        detector = torch.randint(1, 20, (5, 1, *self.wfm.shape_detector), dtype=torch.float32)
        detector = mask_sdd_zero_response_area(self.wfm, detector)

        self.assertTrue(detector.ndim == 4)
        for d in detector:
            assert_tensor_allclose(d[0][zero_response] - self.wfm.bulk[zero_response])
        return

    def test_run_train_epoch_no_pinn(self):
        """Tests model training epoch: model + loss + backward, no PINN term in loss."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if device != 'cuda':
            print('\n[INFO] cuda not available, skipped `test_run_train_epoch_no_pinn()` in TestTraining.\n')
            return

        # config data
        batch_size = 8
        mock_sg_shape = (80, 120)
        mock_psf_shape = (10 * 2 + 1, 4 * 2 + 1)

        sg_batch, sg_pars_batch = (
            torch.randn((2 * batch_size, 1, *mock_sg_shape)),
            (2 * torch.rand((2 * batch_size, 3)) - 1) * self.wfm_epoch.bins_sky.x[-10],
        )
        psf_batch, ext_pars_batch = (
            torch.randn((2 * batch_size, 1, *mock_psf_shape)),
            (2 * torch.rand((2 * batch_size, 3)) - 1) * self.wfm_epoch.bins_sky.x[-10],
        )
        # define datasets
        ds = SrcDiffusionDataset(sg_batch, sg_pars_batch, psf_batch, ext_pars_batch)
        train_dl, valid_dl = get_dataloaders(ds, batch_size, 0.2)
        # noise scheduler + define model & training params
        scheduler = NoiseScheduler(10)
        betas = scheduler.cosine()

        model: Unet = Unet(
            dim=4,
            dim_mults=[1, 2, 4, 8],
            cond_dim=2,
            attn_dim_head=2,
            attn_heads=2,
            convnext_mult=2,
        )
        tpars_factory = dict(
            model=model,
            sampler=DPMSolverPP2MSampler(betas, pred_type='v'),
            loss=JointDiffusionLoss(self.wfm_epoch.to(device), include_pinn_term=False),
            optimiser=opt.Adam,
            device=device,
        )
        tpars = config_training(**tpars_factory)
        # train epoch
        print('[INFO] Starting test train model - single epoch - no PINN')
        results = train_model(self.wfm_epoch, tpars, 1, 1e-3, train_dl, valid_dl)
        return

    def test_run_train_epoch_w_pinn(self):
        """Tests model training epoch: model + loss + backward, with PINN term in loss."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device != 'cuda':
            print('\n[INFO] cuda not available, skipped `test_run_train_epoch_w_pinn()` in TestTraining.\n')
            return

        # config data
        batch_size = 8
        mock_sg_shape = (80, 120)
        mock_psf_shape = (10 * 2 + 1, 4 * 2 + 1)

        l = len(self.wfm_epoch.bins_sky.y)  # NOTE: need narrower FoV to account for SPSF cropping
        sg_batch, sg_pars_batch = (
            torch.randn((2 * batch_size, 1, *mock_sg_shape)),
            (2 * torch.rand((2 * batch_size, 3)) - 1) * self.wfm_epoch.bins_sky.y[l // 2 + 10],
        )
        psf_batch, ext_pars_batch = (
            torch.randn((2 * batch_size, 1, *mock_psf_shape)),
            (2 * torch.rand((2 * batch_size, 3)) - 1) * self.wfm_epoch.bins_sky.y[l // 2 + 10],
        )

        # define datasets
        ds = SrcDiffusionDataset(sg_batch, sg_pars_batch, psf_batch, ext_pars_batch)
        train_dl, valid_dl = get_dataloaders(ds, batch_size, 0.2)
        # noise scheduler + define model & training params
        scheduler = NoiseScheduler(10)
        betas = scheduler.cosine()

        model: Unet = Unet(
            dim=4,
            dim_mults=[1, 2, 4, 8],
            cond_dim=2,
            attn_dim_head=2,
            attn_heads=2,
            convnext_mult=2,
        )
        tpars_factory = dict(
            model=model,
            sampler=DPMSolverPP2MSampler(betas, pred_type='v'),
            loss=JointDiffusionLoss(self.wfm_epoch.to(device), include_pinn_term=True),
            optimiser=opt.Adam,
            device=device,
        )
        tpars = config_training(**tpars_factory)
        # train epoch
        print('[INFO] Starting test train model - single epoch - with PINN')
        results = train_model(self.wfm_epoch, tpars, 1, 1e-3, train_dl, valid_dl)
        return




if __name__ == '__main__':
    unittest.main()


# end