"""All loss functions
"""
from abc import ABC, abstractmethod

import numpy as np

class Loss(ABC):
    """Base class for all loss functions
    """
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def loss(self, y, y_pred, **kwargs):
        """Loss function 
        """
        raise NotImplementedError

    @abstractmethod
    def grad(self, y, y_pred, **kwargs):
        """Gradient function
        """
        raise NotImplementedError

class BinaryCrossEntropyLoss(Loss):
    """Cross Entropy loss

    loss = -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)    
    """
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, y, y_pred, reduce=True):
        return self.loss(y, y_pred, reduce=reduce)

    @staticmethod
    def loss(y, y_pred, **kwargs):
        """Returns cross entropy loss for each sample or across batch

        y (numpy.ndarray): Actual labels
        y_pred (numpy.ndarray): Predicted labels
        reduce (bool): Whether or not to return the average across the batch

        Returns:
            loss (float/numpy.ndarray): If reduce, then will return a float else a numpy array
        """
        reduce = kwargs.get("reduce", True)
        loss = -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)
        if reduce:
            loss = np.mean(loss)
        
        return loss

    @staticmethod
    def grad(y, y_pred, **kwargs):
        """Returns gradient for cross entropy loss

        y (numpy.ndarray): Actual labels
        y_pred (numpy.ndarray): Predicted labels

        Returns:
            grad (numpy.ndarray): Gradient tensor
        """
        grad = y_pred - y

        return grad

class CrossEntropyLoss(Loss):
    """Cross Entropy loss

    loss = -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)    
    """
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, y, y_pred, reduce=True):
        return self.loss(y, y_pred, reduce=reduce)

    @staticmethod
    def loss(y, y_pred, **kwargs):
        """Returns cross entropy loss for each sample or across batch

        y (numpy.ndarray): Actual labels
        y_pred (numpy.ndarray): Predicted labels
        reduce (bool): Whether or not to return the average across the batch

        Returns:
            loss (float/numpy.ndarray): If reduce, then will return a float else a numpy array
        """
        reduce = kwargs.get("reduce", True)
        # prevent taking the log of 0
        eps = np.finfo(float).eps

        # each example is associated with a single class; sum the negative log
        # probability of the correct label over all classes in the batch.
        loss = -np.sum(y * np.log(y_pred + eps), axis=1)
        if reduce:
            loss = np.mean(loss)
        
        return loss

    @staticmethod
    def grad(y, y_pred, **kwargs):
        """Returns gradient for cross entropy loss

        y (numpy.ndarray): Actual labels
        y_pred (numpy.ndarray): Predicted labels

        Returns:
            grad (numpy.ndarray): Gradient tensor
        """
        grad = y_pred - y

        return grad


class MSELoss(Loss):
    """Mean squared error
    """
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, y, y_pred, reduce=True):
        return self.loss(y, y_pred, reduce=reduce)

    @staticmethod
    def loss(y, y_pred, **kwargs):
        """Returns mean squared error loss for each sample or across batch

        y (numpy.ndarray): Actual labels
        y_pred (numpy.ndarray): Predicted labels
        reduce (bool): Whether or not to return the average across the batch

        Returns:
            loss (float/numpy.ndarray): If reduce, then will return a float else a numpy array
        """
        reduce = kwargs.get("reduce", True)
        loss = (1 / 2) * np.linalg.norm(y_pred - y) ** 2
        if reduce:
            loss = np.mean(loss)
        
        return loss

    @staticmethod
    def grad(y, y_pred, **kwargs):
        """Returns gradient for mean squared error loss for each sample or across batch

        y (numpy.ndarray): Actual labels
        y_pred (numpy.ndarray): Predicted labels
        z (numpy.ndarray): Pre non-linearity
        act_fn (dyn_fed.activations.Activation): The activation function for the output layer of the network

        Returns:
            grad (numpy.ndarray): Gradient tensor
        """
        grad = (y_pred - y)

        return grad

class HingeLoss(Loss):
    """Hinge loss function and grad
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def loss(y, y_pred, **kwargs):
        w = kwargs.get("w")
        reg_param = kwargs.get("reg_param")
        reduce = kwargs.get("reduce", True)
        loss_val = y * y_pred
        loss_val = 1 - loss_val if loss_val < 1 else np.zeros_like(loss_val)
        if reduce:
            loss_val = np.mean(loss_val)
        if reg_param is not None:
            loss_val += reg_param * (w ** 2)
        return loss_val

    @staticmethod
    def grad(y, y_pred, **kwargs):
        x = kwargs.get("x")
        w = kwargs.get("w")
        reg_param = kwargs.get("reg_param")
        assert x is not None and w is not None, "Expecting x and w, got None"
        loss_val = y * y_pred
        grads = -y * x if loss_val < 1 else np.zeros_like(loss_val)
        if reg_param is not None:
            grads = (2 * reg_param * w) + grads
        return grads


# Logistic regression
def single_cross_entropy_loss(y_pred, y):
    return -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)

def cross_entropy_loss(y_pred, y):
    return np.mean(single_cross_entropy_loss(y_pred, y))

def cross_entropy_gradient(X, e):
    # d_W = np.mean(X * e, axis=0).T
    d_W = np.dot(X.T, e)
    # d_W = np.sum(X * e, axis=0)
    return d_W

# Linear regression
def mse(y_pred, y):
    """Mean squared error cost function
    .. math::
        \frac{1}/{2m}(h(x) - y)^2
    """
    m, _ = np.shape(y_pred)
    z = ( y_pred - y )
    cost = 1.0 / (2.0 * m) * np.sum(z**2)
    return cost

def mse_gradient(X, e):
    d_W = np.mean(X * e, axis=0).T
    return d_W
