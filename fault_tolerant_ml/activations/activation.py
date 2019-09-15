"""Activation functions
"""
from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    """Abstract class for activation functions
    """
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, z):
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        """The activation function logic

        Args:
            z (ftml.Tensor): The independent variable to perform
                the activation function on
        """
        raise NotImplementedError

    @abstractmethod
    def grad(self, x, **kwargs):
        """The gradient of the activation function
        """
        raise NotImplementedError

    @abstractmethod
    def grad2(self, x, **kwargs):
        """Second derivative of the activation function
        """
        raise NotImplementedError

class Linear(Activation):
    """A linear activation function.
    """
    def __init__(self):
        super(Linear).__init__()

    def __str__(self):
        return "Linear"

    def fn(self, z):
        return z

    def grad(self, x, **kwargs):
        return 1

    def grad2(self, x, **kwargs):
        return 0

class Sigmoid(Activation):
    """
    A logistic sigmoid activation function.
    """
    def __init__(self):
        super(Sigmoid).__init__()

    def __str__(self):
        return "Sigmoid"

    def fn(self, z):
        return 1./(1. + np.exp(-z))

    def grad(self, x, **kwargs):
        return self.fn(x) * (1 - self.fn(x))

    def grad2(self, x, **kwargs):
        return self.grad(x) * (1 - 2 * self.fn(x))

class ReLU(Activation):
    """
    A rectified linear activation function.
    ReLU(x) =
        x   if x > 0
        0   otherwise
    ReLU units can be fragile during training and can "die". For example, a
    large gradient flowing through a ReLU neuron could cause the weights to
    update in such a way that the neuron will never activate on any datapoint
    again. If this happens, then the gradient flowing through the unit will
    forever be zero from that point on. That is, the ReLU units can
    irreversibly die during training since they can get knocked off the data
    manifold.
    For example, you may find that as much as 40% of your network can be "dead"
    (i.e. neurons that never activate across the entire training dataset) if
    the learning rate is set too high. With a proper setting of the learning
    rate this is less frequently an issue.
    - Andrej Karpathy
    """
    def __init__(self):
        super(ReLU).__init__()

    def __str__(self):
        return "ReLU"

    def fn(self, z):
        return np.clip(z, 0, np.inf)

    def grad(self, x, **kwargs):
        return (x > 0).astype(int)

    def grad2(self, x, **kwargs):
        return np.zeros_like(x)

class LeakyReLU(Activation):
    """
    'Leaky' version of a rectified linear unit (ReLU).
    f(x) =
        alpha * x   if x < 0
        x           otherwise
    Leaky ReLUs are designed to address the vanishing gradient problem in ReLUs
    by allowing a small non-zero gradient when x is negative.
    Parameters
    ----------
    alpha: float (default: 0.3)
        Activation slope when x < 0
    References
    ----------
    - [Rectifier Nonlinearities Improve Neural Network Acoustic Models](
    https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        super().__init__()

    def __str__(self):
        return "Leaky ReLU(alpha={})".format(self.alpha)

    def fn(self, z):
        _z = z.copy()
        _z[z < 0] = _z[z < 0] * self.alpha
        return _z

    def grad(self, x, **kwargs):
        out = np.ones_like(x)
        out[x < 0] *= self.alpha # pylint: disable=unsupported-assignment-operation
        return out

    def grad2(self, x, **kwargs):
        return np.zeros_like(x)

class Tanh(Activation):
    """
    A hyperbolic tangent activation function.
    """

    def __init__(self):
        super(Tanh).__init__()

    def __str__(self):
        return "Tanh"

    def fn(self, z):
        return np.tanh(z)

    def grad(self, x, **kwargs):
        return 1 - np.tanh(x) ** 2

    def grad2(self, x, **kwargs):
        return -2 * np.tanh(x) * self.grad(x)

def sigmoid(x):
    """Sigmoid calculation for the matrix passed into this function
    """
    return 1./(1. + np.exp(-x))

def tanh(x):
    """TanH function
    """
    return np.tanh(x)

def relu(x):
    """ReLu function
    """
    return x * (x > 0)

def sigmoid_der(x):
    """Derivative of Sigmoid
    """
    return sigmoid(x) * (1 - sigmoid(x))
