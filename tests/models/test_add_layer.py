import unittest

import dyn_fed as ft
from dyn_fed.layers import Layer

class TestFault_tolerant_ml(unittest.TestCase):
    """Tests for `dyn_fed` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_valid_add_layer(self):
        """Test valid add layer
        """
        model = ft.Model()
        model.add(Layer(784, 32))

        # This should assert an invalid layer exception
        # model.add(tuple((784, 32)))

        assert isinstance(model.layers[0], Layer)
