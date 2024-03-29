import unittest
from copy import copy
import logging
from dyn_fed.operators import Tensor

logger = logging.getLogger("dfl.operators.tests.test_tensor_copy")

class TestTensorCopy(unittest.TestCase):

    def setUp(self):
        """Set up
        """
        pass

    def tearDown(self):
        """Tear down
        """
        pass

    def test_tensor_deepcopy(self):
        """Test tensor deepcopy
        """
        t1 = Tensor([2, 3, 4])
        t3 = t1.copy()

        assert t1 != t3

    def test_tensor_reference(self):
        """Test tensor reference assignment
        """
        t1 = Tensor([2, 3, 4])
        t3 = t1
        t3 = t3 + 1 # Creates new Tensor object, therefore would not point to same object
        logger.info(f"t3={t3}")
        assert t1 != t3

    def test_tensor_shallowcopy(self):
        """Test tensor shallow copy
        """
        t1 = Tensor([2, 3, 4])
        t3 = copy(t1)
        t3 = t3 + 1 # Creates new Tensor object, therefore would not point to same object
        
        logger.info(f"t3={t3}, t1={t1}")
        assert t1.data.tolist() == [2, 3, 4]

    def test_parameter_copy(self):
        """Test if parameter variable is copied
        """
        t1 = Tensor([2, 3, 4], is_param=True)
        t3 = t1.copy()
        
        logger.info(f"t3={t3}")
        assert t3.is_param