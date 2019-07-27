import unittest

from fault_tolerant_ml.models import Model
from fault_tolerant_ml.layers import Layer

class TestFault_tolerant_ml(unittest.TestCase):
    """Tests for `fault_tolerant_ml` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_valid_add_layer(self):
        """Test valid add layer
        """
        model = Model()
        model.add(Layer(784, 32))

        # This should assert an invalid layer exception
        # model.add(tuple((784, 32)))

        assert isinstance(model.layers[0], Layer)
