import unittest
from copy import copy
import logging
import numpy as np
from fault_tolerant_ml.operators import Tensor

logger = logging.getLogger("ftml.operators.tests.test_tensor_zeros_like")

class TestTensorZerosLike(unittest.TestCase):

    def setUp(self):
        """Set up
        """
        pass

    def tearDown(self):
        """Tear down
        """
        pass

    def test_tensor_abs(self):
        """Test zeros_like for tensor class
        """

        t1 = Tensor([-2, 3, -4])

        t3 = t1.zeros_like()

        logger.info(f"t1.data.zeros_like={np.zeros_like(t1.data)}")
        logger.info(f"t3={t3}")

        assert t3.data.tolist() == np.zeros_like(t1.data).tolist()