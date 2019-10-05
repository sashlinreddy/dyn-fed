"""Test for quantizations
"""
import logging
import unittest

import numpy as np

from fault_tolerant_ml.utils.maths import (linspace_quantization,
                                           reconstruct_approximation)

logger = logging.getLogger("ftml.utils.tests.test_matrix_ops")

class TestMatrixOps(unittest.TestCase):
    """Tests for matrix ops
    """
    def setUp(self):
        """Set up
        """

    def tearDown(self):
        """Tear down
        """

    def test_quantization(self):
        """Test quantization size
        """
        interval = 100
        W = np.random.randn(784, 10)

        msg = linspace_quantization(W, interval=interval)

        # Check if desired no. of bytes
        assert msg.nbytes == 7852

    def test_reconstruct_approx(self):
        """Test reconstruct approximation
        """

        interval = 100
        W = np.random.randn(784, 10).astype(np.float32)

        # Quantize the parameters
        msg = linspace_quantization(W, interval=interval)

        buf = memoryview(msg.tostring())

        # Reconstruct the parameters
        r_W = reconstruct_approximation(buf, W.shape, r_dtype=np.float32)
        logger.info(f"Reconstructed dtype={r_W.dtype}")
        # Check data type
        assert r_W.dtype == np.float32

    def test_reconstruct_error(self):
        """Test reconstruct error
        """
        # Reconstruct error should be small enough for us to be comfortable
        # with approximating the parameters
        eps = 0.1
        interval = 100
        W = np.random.randn(784, 10).astype(np.float32)

        msg = linspace_quantization(W, interval=interval)

        buf = memoryview(msg.tostring())

        r_W = reconstruct_approximation(buf, W.shape, r_dtype=np.float32)

        delta = np.max(abs(W - r_W))
        logger.info(f"Delta={delta}")

        self.assertTrue(
            delta < eps,
            "Reconstruct error is too large, consider a different interval for np.linspace"
        )
