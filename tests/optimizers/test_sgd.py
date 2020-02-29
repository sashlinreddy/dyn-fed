"""Tests for SGD
"""
import logging
import os
import unittest

import numpy as np

from dyn_fed.data import MNist
from dyn_fed.lib.io import file_io
from dyn_fed.losses import CrossEntropyLoss
from dyn_fed.operators import Tensor
from dyn_fed.optimizers import SGDOptimizer
from dyn_fed.utils import maths

logger = logging.getLogger("dfl.optimizers.tests.test_sgd")

class TestSGDOptimizer(unittest.TestCase):
    """Tests for SGD
    """
    def setUp(self):
        """Set up
        """

        cfg = file_io.load_model_config("config.yml")
        executor_cfg = cfg['executor']
        data_dir = executor_cfg['shared_folder']
        # Get data
        filepaths = {
            "train": {
                "images": os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
                "labels": os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
            },
            "test": {
                "images": os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
                "labels": os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
            }
        }
        self.data = MNist(filepaths)

    def tearDown(self):
        """Tear down
        """

    def test_sgd_compute_gradients(self):
        """Test computing gradients for sgd optimizer
        """

        W_prev = Tensor(np.random.randn(784, 10).astype(np.float32) * 0.01, is_param=True)
        loss = CrossEntropyLoss()
        optimizer = SGDOptimizer(loss, learning_rate=0.99, role="worker")

        # Forward pass
        s = self.data.X_train @ W_prev
        y_pred = maths.sigmoid(s)

        logger.debug(f"Before W_prev.grad={W_prev.grad}")

        np.testing.assert_array_equal(W_prev.grad.data, np.zeros_like(W_prev.data))

        W = optimizer.compute_gradients(self.data.X_train, self.data.y_train, y_pred, W_prev)

        logger.debug(f"After W_prev.grad={W_prev.grad}")

        assert np.all(W.grad.data == W_prev.grad.data)
