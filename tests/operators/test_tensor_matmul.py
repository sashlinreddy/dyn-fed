import unittest
from copy import copy
import logging
import numpy as np
from dyn_fed.operators import Tensor

logger = logging.getLogger("dfl.operators.tests.test_tensor_matmul")

class TestTensorMatMul(unittest.TestCase):

    def setUp(self):
        """Set up
        """
        pass

    def tearDown(self):
        """Tear down
        """
        pass

    def test_tensor_matmul(self):
        """Test matmul for tensor class
        """

        t1 = Tensor([[1, 2, 3], [4, 5, 6]]) # (2, 3)
        t2 = Tensor([[2], [2], [2]]) # (3, 1)

        logger.info(f"t2.shape={t2.shape}")

        t3 = t1 @ t2

        logger.info(f"t3={t3}")

        assert t3.data.tolist() == [[12], [30]]

    
    def test_tensor_rmatmul(self):
        """Test rmatmul for tensor class
        """

        t1 = [[1, 2, 3], [4, 5, 6]] # (2, 3)
        t2 = Tensor([[2], [2], [2]]) # (3, 1)

        logger.info(f"t2.shape={t2.shape}")

        t3 = t1 @ t2

        logger.info(f"t3={t3}")

        assert t3.data.tolist() == [[12], [30]]