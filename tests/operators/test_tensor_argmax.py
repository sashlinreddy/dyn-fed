import unittest
from copy import copy
import logging
import numpy as np
from fault_tolerant_ml.operators import Tensor

logger = logging.getLogger("ftml.operators.tests.test_tensor_argmax")

class TestTensorArgmax(unittest.TestCase):

    def setUp(self):
        """Set up
        """
        pass

    def tearDown(self):
        """Tear down
        """
        pass

    def test_tensor_argmax_axis0(self):
        """Test argmax axis=0 for tensor class
        """

        t1 = Tensor([[9, 4, 16], [5, 10, 3], [14, 1, 90], [-20, 20, 9]])

        t3 = np.argmax(t1, axis=0)
        logger.info(f"t3={t3}")

        assert t3.data.tolist() == [2, 3, 2]

    def test_tensor_argmax_axis1(self):
        """Test argmax axis=0 for tensor class
        """

        t1 = Tensor([[9, 4, 16], [5, 10, 3], [14, 1, 90], [-20, 20, 9]])

        t3 = np.argmax(t1, axis=1)
        logger.info(f"t3={t3}")

        assert t3.data.tolist() == [2, 1, 2, 1]