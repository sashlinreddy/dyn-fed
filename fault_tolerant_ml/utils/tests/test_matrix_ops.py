import unittest
import numpy as np
import logging
from fault_tolerant_ml.utils.maths import linspace_quantization, reconstruct_approximation

logger = logging.getLogger("ftml.utils.tests.test_matrix_ops")

class TestMatrixOps(unittest.TestCase):

    def setUp(self):
        """Set up
        """
        pass

    def tearDown(self):
        """Tear down
        """
        pass

    def test_quantization(self):
        """Test quantization size
        """
        interval = 100
        theta = np.random.randn(784, 10)

        msg = linspace_quantization(theta, interval=interval)

        # Check if desired no. of bytes
        assert msg.nbytes == 7852

    def test_reconstruct_approx(self):
        """Test reconstruct approximation
        """

        interval = 100
        theta = np.random.randn(784, 10).astype(np.float32)

        # Quantize the parameters
        msg = linspace_quantization(theta, interval=interval)

        buf = memoryview(msg.tostring())

        # Reconstruct the parameters
        r_theta = reconstruct_approximation(buf, theta.shape, r_dtype=np.float32)
        logger.info(f"Reconstructed dtype={r_theta.dtype}")
        # Check data type
        assert r_theta.dtype == np.float32

    def test_reconstruct_error(self):
        """Test reconstruct error
        """

        # Reconstruct error should be small enough for us to be comfortable with approximating the parameters
        eps = 0.1
        interval = 100
        theta = np.random.randn(784, 10).astype(np.float32)

        msg = linspace_quantization(theta, interval=interval)

        buf = memoryview(msg.tostring())

        r_theta = reconstruct_approximation(buf, theta.shape, r_dtype=np.float32)

        delta = np.max(abs(theta - r_theta))
        logger.info(f"Delta={delta}")

        assert delta < eps, "Reconstruct error is too large, consider a different interval for np.linspace"

