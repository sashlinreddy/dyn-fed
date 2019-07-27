import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    """Base class for all loss functions
    """
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def loss(self, y, y_pred):
        raise NotImplementedError

    @abstractmethod
    def grad(self, y, y_pred):
        raise NotImplementedError

class CrossEntropyLoss(Loss):
    """Cross Entropy loss

    loss = -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)    
    """
    def __init__(self):
        super().__init__()

    def __call__(self, y, y_pred, reduce=True):
        return self.loss(y, y_pred, reduce=reduce)

    @staticmethod
    def loss(y, y_pred, reduce=True):
        """Returns cross entropy loss for each sample or across batch

        y (numpy.ndarray): Actual labels
        y_pred (numpy.ndarray): Predicted labels
        reduce (bool): Whether or not to return the average across the batch

        Returns:
            loss (float/numpy.ndarray): If reduce, then will return a float else a numpy array
        """
        loss = -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)
        if reduce:
            loss = np.mean(loss)
        
        return loss

    @staticmethod
    def grad(y, y_pred):
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
    def __init__(self):
        super().__init__()

    def __call__(self, y, y_pred, reduce=True):
        return self.loss(y, y_pred, reduce=reduce)

    @staticmethod
    def loss(y, y_pred, reduce=True):
        """Returns mean squared error loss for each sample or across batch

        y (numpy.ndarray): Actual labels
        y_pred (numpy.ndarray): Predicted labels
        reduce (bool): Whether or not to return the average across the batch

        Returns:
            loss (float/numpy.ndarray): If reduce, then will return a float else a numpy array
        """
        loss = (1 / 2) * np.linalg.norm(y_pred - y) ** 2
        if reduce:
            loss = np.mean(loss)
        
        return loss

    @staticmethod
    def grad(y, y_pred, z, act_fn):
        """Returns gradient for mean squared error loss for each sample or across batch

        y (numpy.ndarray): Actual labels
        y_pred (numpy.ndarray): Predicted labels
        z (numpy.ndarray): Pre non-linearity
        act_fn (fault_tolerant_ml.activations.Activation): The activation function for the output layer of the network

        Returns:
            grad (numpy.ndarray): Gradient tensor
        """
        grad = (y_pred - y) * act_fn.grad(z)

        return grad


# Logistic regression
def single_cross_entropy_loss(y_pred, y):
    return -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)

def cross_entropy_loss(y_pred, y):
    return np.mean(single_cross_entropy_loss(y_pred, y))

def cross_entropy_gradient(X, e):
    # d_theta = np.mean(X * e, axis=0).T
    d_theta = np.dot(X.T, e)
    # d_theta = np.sum(X * e, axis=0)
    return d_theta

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
    d_theta = np.mean(X * e, axis=0).T
    return d_theta
