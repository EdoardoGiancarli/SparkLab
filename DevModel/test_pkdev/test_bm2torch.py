"""
Test module for `bloodmoon` package conversion to torch, within spark's `camera.py` module.

The tests are divided in:
    * testing custom, inner funcs
    * testing CodedMaskCamera instance
    * testing CAI operations
    * testing additional/helper funcs
"""

from bisect import bisect_right, bisect_left
from typing import Optional
import unittest
from unittest import TestCase

import numpy as np
from scipy.signal import correlate
import torch
from torch.types import Tensor

from irosbm.types import BinsRectangular as bmBinsRectangular
from irosbm.mask import (
    _bisect_interval as bm_bisect_interval,
    codedmask as bm_codedmask,
    decode as bm_decode,
)
from irosbm.images import (
    _upscale as bm_upscale,
    _shift as bm_shift,
)

from pkdev.camera import (
    BinsRectangular,
    _bisect_right,
    _bisect_left,
    _bisect_interval,
    _upscale,
    _shift,
    _correlate,
    codedmask,
    decode,
    decode_batch,
    argmax,
    find_boxmax,
)

from .assets import maskpath


def assert_tensor_allclose(residuals: Tensor, eps: float = 1e-8, msg: Optional[str] = None) -> None:
    torch.testing.assert_close(residuals, torch.zeros_like(residuals), atol=eps, rtol=0.0, msg=msg)
    return


class TestCustomFuncs(TestCase):
    """
    Test class for the `_bisect_left`, `_bisect_right`, `_bisect_interval`, `_upscale`, `_shift`, `_correlate` funcs.
    """
    def test_bisect_fns(self):
        """Test func for `_bisect_left`, `_bisect_right`."""
        val = 10.0
        a_np = np.linspace(-val, val, int(2 * val) + 1) / val
        start, stop = 0.55, 0.95

        expected = (bisect_right(a_np, start), bisect_left(a_np, stop))

        a_tr = torch.from_numpy(a_np).clone()
        result = (_bisect_right(a_tr, start), _bisect_left(a_tr, stop))

        self.assertEqual(result, expected)
        return

    def test_bisect_interval(self):
        """Test func for `_bisect_interval`."""
        val = 10.0
        a_np = np.linspace(-val, val, int(2 * val) + 1) / val
        start, stop = -0.32, 0.12

        expected = bm_bisect_interval(a_np, start, stop)

        a_tr = torch.from_numpy(a_np).clone()
        result = _bisect_interval(a_tr, start, stop)

        self.assertEqual(result, expected)
        return

    def test_upscaling(self):
        """Test func for `_upscale`."""
        size = (20, 30)
        upx, upy = 10, 3

        m_np = np.random.randint(0, 16, size)
        expected = torch.from_numpy(bm_upscale(m_np, upx, upy)).clone()

        m_tr = torch.from_numpy(m_np).clone()
        result = _upscale(m_tr, upx, upy)

        assert_tensor_allclose(result - expected)
        return

    def test_shift(self):
        """Test func for `_shift`."""
        size = (20, 30)
        shifts = ((3, 3), (3, -3), (-3, 3), (-3, -3))

        m_np = np.random.randint(0, 16, size)
        m_tr = torch.from_numpy(m_np).clone()

        for (sx, sy) in shifts:
            expected = torch.from_numpy(bm_shift(m_np, sx, sy)).clone()
            result = _shift(m_tr, sx, sy)
            assert_tensor_allclose(result - expected)

        return

    def _correlation_test_logic(
        self,
        in_shape: tuple[int, int],
        kernel_shape: tuple[int, int],
        mode: str,
        eps: float = 1e-8,
    ) -> None:
        out_shape = (
            tuple(n + m - 1 for n, m in zip(in_shape, kernel_shape)) if mode == 'full' else in_shape
        )
        a_np, b_np = np.random.randint(0, 16, in_shape), np.random.randint(0, 16, kernel_shape)
        a_tr, b_tr = map(lambda x: torch.from_numpy(x).clone(), (a_np, b_np))

        expected = correlate(a_np, b_np, mode=mode)
        result = _correlate(a_tr, b_tr, mode=mode)

        self.assertEqual(result.shape, out_shape)
        assert_tensor_allclose(result - torch.from_numpy(expected), eps=eps)
        return

    def test_correlate_odd_parity(self):
        """Test func for `_correlate`, focussing on odd-shaped tensors."""
        in_shape = (11, 21)
        kernel_shape = (3, 7)
        self._correlation_test_logic(in_shape, kernel_shape, mode='full')
        self._correlation_test_logic(in_shape, kernel_shape, mode='same')
        return

    def test_correlate_even_parity(self):
        """Test func for `_correlate`, focussing on even-shaped tensors."""
        in_shape = (10, 20)
        kernel_shape = (4, 8)
        self._correlation_test_logic(in_shape, kernel_shape, mode='full')
        self._correlation_test_logic(in_shape, kernel_shape, mode='same')
        return

    def test_correlation_method_int(self):
        """Test func for `_correlate`, focussing on 'direct' vs 'fft' method and int32 inputs."""
        in_shape = (100, 120)
        kernel_shape = (11, 13)

        a, b = map(lambda s: torch.randint(1, 100, s, dtype=torch.int32), (in_shape, kernel_shape))
        cc_direct = _correlate(a, b, method='direct')
        cc_fft = _correlate(a, b, method='fft')

        res = (cc_direct - cc_fft).to(torch.float32)
        assert_tensor_allclose(res, eps=1e-6)
        return

    def test_correlation_method_float(self):
        """Test func for `_correlate`, focussing on 'direct' vs 'fft' method and float32 inputs."""
        in_shape = (100, 120)
        kernel_shape = (11, 13)

        a, b = map(lambda s: torch.randint(1, 100, s, dtype=torch.float32), (in_shape, kernel_shape))
        cc_direct = _correlate(a, b, method='direct')
        cc_fft = _correlate(a, b, method='fft')

        res = cc_direct - cc_fft
        assert_tensor_allclose(res, eps=1e-6, msg=f'mean_res (@ 3sigma) = {res.mean().item()} +/- {3 * res.std().item()}')
        return




class TestCMC(TestCase):
    """
    Test class for the `CodedMaskCamera` instance conversion to torch.
    """
    def setUp(self):
        upx, upy = 5, 2
        self.wfm_bm = bm_codedmask(maskpath, upx, upy)
        self.wfm = codedmask(maskpath, upx, upy)

    def test_binning(self):
        """
        Tests the binning structure of CMC tensors.
        """
        def inspect_bins(bins: BinsRectangular, expected: bmBinsRectangular, grid: str, eps: float = 2e-5) -> None:
            write_outmsg = lambda axis, res: (
                f'[{grid}] Failed test on {axis} axis, mean_res (@ 3 sigma) = {res.mean().item()} +/- {3 * res.std().item()}'
            )
            res_x = bins.x - torch.from_numpy(expected.x)
            assert_tensor_allclose(res_x, eps=eps, msg=write_outmsg('X', res_x))
            res_y = bins.y - torch.from_numpy(expected.y)
            assert_tensor_allclose(res_y, eps=eps, msg=write_outmsg('Y', res_y))
            return

        inspect_bins(self.wfm.bins_detector, self.wfm_bm.bins_detector, grid='DETECTOR')
        inspect_bins(self.wfm.bins_mask, self.wfm_bm.bins_mask, grid='MASK')
        inspect_bins(self.wfm.bins_sky, self.wfm_bm.bins_sky, grid='SKY')
        return

    def test_mask(self):
        """Tests the mask/decoder patterns tensor initilisation (pattern vals and shape)."""
        assert_tensor_allclose(self.wfm.mask - torch.from_numpy(self.wfm_bm.mask))
        return

    def test_bulk(self):
        """Tests the detector bulk tensor initilisation (pattern vals and shape)."""
        assert_tensor_allclose(self.wfm.bulk - torch.from_numpy(self.wfm_bm.bulk))
        return

    def test_balancing(self):
        """Tests the instrumental balancing tensor initilisation (pattern vals and shape)."""
        assert_tensor_allclose(self.wfm.balancing - torch.from_numpy(self.wfm_bm.balancing))
        return

    def test_move_to_device(self):
        """Tests CMC device shift."""
        if torch.cuda.is_available():
            wfm = codedmask(maskpath, 5, 2)
            wfm = wfm.to('cuda')
            self.assertEqual(str(wfm.device).split(':')[0], 'cuda')
            self.assertEqual(str(wfm.bulk.device).split(':')[0], 'cuda')
        else:
            print('\n[INFO] cuda not available, skipped `test_move_to_device()` in TestCMC.\n')
        return



class TestCAIFuncs(TestCase):
    """
    Test class for CAI operations.
    """
    def setUp(self):
        upx, upy = 2, 1
        self.wfm_bm = bm_codedmask(maskpath, upx, upy)
        self.wfm = codedmask(maskpath, upx, upy)

    def test_decode_func(self):
        """Tests if decoding behaves accordingly to base bm version, with `decode`."""
        d_np = np.random.randint(1, 100, self.wfm_bm.shape_detector) * self.wfm_bm.bulk
        sky_np = bm_decode(self.wfm_bm, d_np)

        d_tr = torch.from_numpy(d_np).clone()
        sky_tr = decode(self.wfm, d_tr)

        assert_tensor_allclose(sky_tr - torch.from_numpy(sky_np))
        return

    def test_decode_detector_batch(self):
        """Tests the decoding of a detector images batch, with `decode_batch`."""
        d_batch = (
            torch.randint(1, 100, (5, 1, *self.wfm.shape_detector)) * torch.cat(5 * [self.wfm.bulk[None, None, ...]], dim=0)
        )
        sky_batch = decode_batch(self.wfm, d_batch)
        self.assertEqual(sky_batch.shape, (5, 1, *self.wfm.shape_sky))

        for idx, d in enumerate(d_batch):
            sky = decode(self.wfm, d[0])
            assert_tensor_allclose(sky - sky_batch[idx, 0])

        return



class TestHelperFuncs(TestCase):
    """
    Test class for additional/helper funcs.
    """
    def setUp(self):
        pass

    def test_argmax(self):
        """Tests the `argmax` func."""
        a = torch.tensor(
            [
                [0, 5, 2, 6, 3, 7, 1, 3, 6,],
                [0, 5, 2, 6, 3, 7, 1, 3, 6,],
                [0, 5, 2, 6, 3, 7, 1, 3, 6,],
                [0, 100, 2, 6, 3, 7, 1, 3, 6,],
                [0, 5, 2, 6, 3, 7, 1, 3, 6,],
            ],
            dtype=torch.float32,
        )
        self.assertEqual(argmax(a), (3, 1))
        return

    def test_findboxmax(self):
        """Tests the `find_boxmax` func."""
        a = torch.tensor(
            [
                [0, 5, 2, 6, 3, 7, 1, 3, 6,],
                [0, 5, 2, 6, 3, 7, 1, 3, 6,],
                [0, 5, 2, 6, 3, 100, 1, 3, 6,],
                [0, 5, 2, 6, 3, 7, 1, 3, 6,],
                [0, 105, 2, 6, 3, 7, 1, 3, 6,],
            ],
            dtype=torch.float32,
        )
        centre = (1, 5)
        self.assertEqual(find_boxmax(a, centre=centre, boxsize=(2, 2)), ((2, 5), 100))
        centre = (3, 2)
        self.assertEqual(find_boxmax(a, centre=centre, boxsize=(3, 3)), ((4, 1), 105))
        return




if __name__ == '__main__':
    unittest.main()


# end