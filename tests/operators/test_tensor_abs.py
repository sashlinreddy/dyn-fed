import unittest
from copy import copy
import logging
from dyn_fed.operators import Tensor

logger = logging.getLogger("dfl.operators.tests.test_tensor_abs")

class TestTensorAbs(unittest.TestCase):

    def setUp(self):
        """Set up
        """
        pass

    def tearDown(self):
        """Tear down
        """
        pass

    def test_tensor_abs(self):
        """Test absolute value of tensor
        """

        t1 = Tensor([-2, 3, -4])

        assert abs(t1).data.tolist() == [2, 3, 4]