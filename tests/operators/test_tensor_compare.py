"""Test comparison operators for tensor
"""
import unittest
import logging
from dyn_fed.operators import Tensor

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
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

    def test_tensor_lt(self):
        """Test < operator
        """

        t1 = Tensor([-2, 3, -4])

        assert (t1 < 0).astype(int).data.tolist() == [1, 0, 1]

    def test_tensor_le(self):
        """Test <= operator
        """

        t1 = Tensor([-2, 3, 0, -4])

        assert (t1 <= 0).astype(int).data.tolist() == [1, 0, 1, 1]

    def test_tensor_gt(self):
        """Test > operator
        """

        t1 = Tensor([-2, 3, -4])

        assert (t1 > 0).astype(int).data.tolist() == [0, 1, 0]

    def test_tensor_ge(self):
        """Test >= operator
        """

        t1 = Tensor([-2, 3, 0, -4])

        assert (t1 >= 0).astype(int).data.tolist() == [0, 1, 1, 0]
        