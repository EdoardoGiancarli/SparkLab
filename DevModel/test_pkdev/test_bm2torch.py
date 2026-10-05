"""
Test module for `bloodmoon` package conversion to torch, within spark's `camera.py` module.

The tests are divided in:
    * testing custom, inner funcs
    * testing CodedMaskCamera instance
    * testing CAI operations
"""

from bisect import bisect_right, bisect_left
import unittest
from unittest import TestCase

import numpy as np
from scipy.signal import correlate
import torch
from torch.types import Tensor

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
    _bisect_right,
    _bisect_left,
    _bisect_interval,
    _upscale,
    _shift,
    _correlate,
    codedmask,
    decode,
)


def assert_tensor_allclose(residuals: Tensor, eps: float = 1e-8) -> None:
    torch.testing.assert_close(residuals, torch.zeros_like(residuals), atol=eps, rtol=0.0)
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

    def _correlation_test_logic(self, in_shape: tuple[int, int], kernel_shape: tuple[int, int], mode: str) -> None:
        out_shape = (
            tuple(n + m - 1 for n, m in zip(in_shape, kernel_shape)) if mode == 'full' else in_shape
        )
        a_np, b_np = np.random.randint(0, 16, in_shape), np.random.randint(0, 16, kernel_shape)
        a_tr, b_tr = map(lambda x: torch.from_numpy(x).clone(), (a_np, b_np))

        expected = correlate(a_np, b_np, mode=mode)
        result = _correlate(a_tr, b_tr, mode=mode)

        self.assertEqual(result.shape, out_shape)
        assert_tensor_allclose(result - torch.from_numpy(expected))
        return

    def test_correlate_odd_parity(self):
        """Test func for `_correlate`, focussing on odd-shaped tensors."""
        in_shape = (11, 21)
        kernel_shape = (3, 7)
        # full mode
        self._correlation_test_logic(in_shape, kernel_shape, mode='full')
        # same mode
        self._correlation_test_logic(in_shape, kernel_shape, mode='same')
        return

    def test_correlate_even_parity(self):
        """Test func for `_correlate`, focussing on even-shaped tensors."""
        in_shape = (10, 20)
        kernel_shape = (4, 8)
        # full mode
        self._correlation_test_logic(in_shape, kernel_shape, mode='full')
        # same mode
        self._correlation_test_logic(in_shape, kernel_shape, mode='same')
        return



class TestCMC(TestCase):
    """
    Test class for the `CodedMaskCamera` instance conversion to torch.
    """
    def setUp(self) -> None:
        ...



class TestCAIFuncs(TestCase):
    """
    Test class for CAI operations.
    """
    def setUp(self) -> None:
        ...




if __name__ == '__main__':
    unittest.main()


# end