import unittest
from copy import copy
import logging
import numpy as np
from dyn_fed.operators import Tensor

logger = logging.getLogger("dfl.operators.tests.test_tensor_sqrt")

class TestTensorSqrt(unittest.TestCase):

    def setUp(self):
        """Set up
        """
        pass

    def tearDown(self):
        """Tear down
        """
        pass

    def test_tensor_sqrt(self):
        """Test sqrt for tensor class
        """

        t1 = Tensor([9, 4, 16])

        t3 = np.sqrt(t1)
        logger.info(f"t1={t1}")
        logger.info(f"t3={t3}")

        assert t3.data.tolist() == [3, 2, 4]