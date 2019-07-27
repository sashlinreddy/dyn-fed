import unittest
import numpy as np
from fault_tolerant_ml.utils.maths import linspace_quantization, reconstruct_approximation

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
        """Test quantization
        """
        interval = 100
        theta = np.random.randn(784, 10)

        msg = linspace_quantization(theta, interval=interval)

        assert msg.nbytes == 7852

    def test_reconstruct_approx(self):
        """Test reconstruct approximation
        """

        interval = 100
        theta = np.random.randn(784, 10).astype(np.float32)

        msg = linspace_quantization(theta, interval=interval)

        buf = memoryview(msg.tostring())
        # struct_field_names = ["min_val", "max_val", "interval", "bins"]
        # struct_field_types = [np.float32, np.float32, np.int32, 'b']
        # struct_field_shapes = [1, 1, 1, (theta.shape)]
        # dtype=(list(zip(struct_field_names, struct_field_types, struct_field_shapes)))

        # data = np.frombuffer(buf, dtype=dtype)
        # min_theta_val, max_theta_val, interval, theta_bins = data[0]

        # # Generate lineared space vector
        # bins = np.linspace(min_theta_val, max_theta_val, interval, dtype=np.float32)
        # theta = bins[theta_bins].reshape(theta.shape)
        r_theta = reconstruct_approximation(buf, theta.shape, r_dtype=np.float32)
        print(f"Reconstructed dtype={r_theta.dtype}")
        assert r_theta.dtype == np.float32