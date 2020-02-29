"""Tests for SGD
"""
import logging
import unittest

import numpy as np

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# pylint: disable=wrong-import-position
from dyn_fed.layers import Layer
from dyn_fed.models import Model
from dyn_fed.proto.utils import (params_to_string,
                                           parse_params_from_string,
                                           parse_setup_from_string,
                                           setup_to_string)


logger = logging.getLogger("dfl.optimizers.tests.protos.test_protobuf")

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

        buffer = setup_to_string(X, y, n_samples, state)

        logger.debug(f"Size of buffer={len(buffer)}")

        X_p, y_p, n_samples_p, state_p = parse_setup_from_string(buffer)

        # pylint: disable=no-member
        self.assertEqual(n_samples_p, n_samples)
        self.assertEqual(state_p, state)
        logger.debug(f"X_p.shape={X_p.shape}")
        logger.debug(f"y_p.shape={y_p.shape}")

        np.testing.assert_array_equal(X_p, X, "Deserialization failed for X")
        np.testing.assert_array_equal(y_p, y, "Deserialization failed for y")

    def test_one_layer_parameter_proto(self):
        """Test parameter protobuf serialization for one layer model
        """
        # Setup model
        model = Model(optimizer=None)
        model.add(Layer(784, 10, activation="linear"))
        
        buffer = params_to_string(model.layers)

        logger.debug(f"Size of buffer={len(buffer)}")

        layers = parse_params_from_string(buffer)

        self.assertTrue(len(layers) == 1)
        np.testing.assert_array_equal(model.layers[0].W.data, layers[0][0])
        np.testing.assert_array_equal(model.layers[0].b.data, layers[0][1])

    def test_multiple_layer_linear_parameter_proto(self):
        """Test parameter protobuf serialization for multiple layer model
        """
        # Setup model
        model = Model(optimizer=None)
        model.add(Layer(784, 128, activation="sigmoid"))
        model.add(Layer(128, 10, activation="sigmoid"))
        
        buffer = params_to_string(model.layers)

        logger.debug(f"Size of buffer={len(buffer)}")

        layers = parse_params_from_string(buffer)

        self.assertTrue(len(layers) == 2)
        np.testing.assert_array_equal(model.layers[0].W.data, layers[0][0])
        np.testing.assert_array_equal(model.layers[0].b.data, layers[0][1])
        np.testing.assert_array_equal(model.layers[1].W.data, layers[1][0])
        np.testing.assert_array_equal(model.layers[1].b.data, layers[1][1])
