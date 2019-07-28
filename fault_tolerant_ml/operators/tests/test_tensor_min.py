import unittest
from copy import copy
import logging
import numpy as np
from fault_tolerant_ml.operators import Tensor

logger = logging.getLogger("ftml.operators.tests.test_tensor_min")

class TestTensorSqrt(unittest.TestCase):

    def setUp(self):
        """Set up
        """
        pass

    def tearDown(self):
        """Tear down
        """
        pass

    def test_tensor_min_no_axis(self):
        """Test min for tensor class
        """

        t1 = Tensor([9, 4, 16])

        t3 = np.min(t1)
        logger.info(f"t1={t1}")
        logger.info(f"t3={t3}")
        logger.info(t3.data.tolist())

        assert t3.data.tolist() == 4

    def test_tensor_min_axis0(self):
        """Test min along axis 0 for tensor class
        """

        t1 = Tensor(np.array([[9, 4, 16], [5, 10, 3], [14, 1, 90], [-20, 20, 9]]))

        t3 = np.min(t1, axis=0)
        logger.info(f"t1={t1}")
        logger.info(f"t3={t3}")
        logger.info(t3.data.tolist())

        assert t3.data.tolist() == [-20, 1, 3]

    def test_tensor_min_axis1(self):
        """Test min along axis 1 for tensor class
        """

        t1 = Tensor(np.array([[9, 4, 16], [5, 10, 3], [14, 1, 90], [-20, 20, 9]]))

        t3 = np.min(t1, axis=1)
        logger.info(f"t1={t1}")
        logger.info(f"t3={t3}")
        logger.info(t3.data.tolist())

        assert t3.data.tolist() == [4, 3, 1, -20]