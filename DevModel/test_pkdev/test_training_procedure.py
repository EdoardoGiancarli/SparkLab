"""
Test module for training procedure (spark's `training.py` module).
"""

from typing import Optional
import unittest
from unittest import TestCase

import torch
from torch.types import Tensor

from pkdev.camera import codedmask
from pkdev.training import mask_sdd_zero_response_area

from .assets import maskpath


def assert_tensor_allclose(residuals: Tensor, eps: float = 1e-8, msg: Optional[str] = None) -> None:
    torch.testing.assert_close(residuals, torch.zeros_like(residuals), atol=eps, rtol=0.0, msg=msg)
    return




class TestTraining(TestCase):
    """
    Test class for `spark`'s training procedure.
    """
    def setUp(self):
        upx, upy = 5, 2
        self.wfm = codedmask(maskpath, upx, upy)

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




if __name__ == '__main__':
    unittest.main()


# end