"""Tests for SGD
"""
import logging
import unittest

import numpy as np

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# pylint: disable=wrong-import-position
from fault_tolerant_ml.proto.ftml_pb2 import Setup, Tensor
from fault_tolerant_ml.proto.utils import reconstruct_setup

logger = logging.getLogger("ftml.optimizers.tests.protos.test_protobuf")

class TestProtobuf(unittest.TestCase):
    """Tests for SGD
    """
    def setUp(self):
        """Set up
        """

    def tearDown(self):
        """Tear down
        """

    def test_setup_data(self):
        """Test serialization for feature data
        """
        X = np.random.randn(10000, 784).astype(np.float32)
        y = np.random.randint(0, 2, size=(10000, 10))
        n_samples = X.shape[0]
        state = 0

        X_proto = Tensor(
            data=X.tostring(),
            rows=X.shape[0],
            columns=X.shape[1],
            dtype=X.dtype.str
        )

        y_proto = Tensor(
            data=y.tostring(),
            rows=y.shape[0],
            columns=y.shape[1],
            dtype=y.dtype.str
        )

        sent_msg = Setup(
            n_samples=n_samples,
            state=state,
            X=X_proto,
            y=y_proto
        )

        buffer = sent_msg.SerializeToString()

        logger.debug(f"Size of buffer={len(buffer)}")

        X_p, y_p, n_samples_p, state_p = reconstruct_setup(buffer)

        # pylint: disable=no-member
        self.assertEqual(n_samples_p, n_samples)
        self.assertEqual(state_p, state)
        logger.debug(f"X_p.shape={X_p.shape}")
        logger.debug(f"y_p.shape={y_p.shape}")

        np.testing.assert_array_equal(X_p, X, "Deserialization failed for X")
        np.testing.assert_array_equal(y_p, y, "Deserialization failed for y")
