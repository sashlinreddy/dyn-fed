import unittest
from fault_tolerant_ml.ml.ops.tensorpy import Tensor

class TestTensorMul(unittest.TestCase):
    
    def test_simple_mul(self):
        """Test simple add
        """
        t1 = Tensor([1, 2, 3], requires_grad=True) # (2, 3)
        t2 = Tensor([4, 5, 6], requires_grad=True) # (3,)

        t3 = t1 * t2 

        assert t3.data.tolist() == [4, 10, 18]

        t3.backward(Tensor([-1, -2, -3]))

        assert t1.grad.data.tolist() == [-4, -10, -18]
        assert t2.grad.data.tolist() == [-1, -4, -9]

    def test_broadcast_mul(self):
        """Test broadcast add
        """

        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = Tensor([7, 8, 9], requires_grad=True)

        t3 = t1 * t2
        assert t3.data.tolist() == [[7, 16, 27], [28, 40, 54]]

        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[7, 8, 9], [7, 8, 9]]
        assert t2.grad.data.tolist() == [5, 7, 9]

    def test_broadcast_mul2(self):
        """Test broadcast add
        """

        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True) # (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad=True) # (1, 3)

        t3 = t1 * t2
        assert t3.data.tolist() == [[7, 16, 27], [28, 40, 54]]
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[7, 8, 9], [7, 8, 9]]
        assert t2.grad.data.tolist() == [[5, 7, 9]]