import numpy as np # Linear Algebra
import logging

class Optimizer(object):
    """Base class for machine learning optimizers
    
    Attributes:
        learning_rate (float): The rate at which the machine learning algorithm will learn      and how fast our descent method will go down the slope to find it's global minima
    """
    def __init__(self, loss, grad, learning_rate):
        self.learning_rate = learning_rate
        self.loss = loss
        self.grad = grad

        self._logger = logging.getLogger("ftml")

    def compute_loss(self, y, y_pred):
        """Compute loss to be overridden by children
        """
        raise NotImplementedError("Child must override this method")

    def compute_gradients(self, X, y, theta):
        """Compute gradients to be overridden by children
        """
        raise NotImplementedError("Child must override this method")

    def apply_gradients(self, d_theta, theta):
        """Compute gradients to be overridden by children
        """
        raise NotImplementedError("Child must override this method")

    def minimize(self, X, y, y_pred, theta):
        """Minimizes objective function to be overridden by children
        
        This function will perform the update of the parameters in order to get close to the  global minima
        """
        raise NotImplementedError("Child must override this method")

class SGDOptimizer(Optimizer):
    """Stochastic gradient descent optimizer
    """
    def __init__(self, loss, grad, learning_rate=0.1, **kwargs):
        super().__init__(loss, grad, learning_rate)
        self._role = None
        self._most_rep = None
        self._clip_norm = None
        self._clip_val = None
        self._n_most_rep = 0
        if "n_most_rep" in kwargs:
            self._n_most_rep = kwargs["n_most_rep"]
        if "role" in kwargs:
            self._role = kwargs["role"]
        if "clip_norm" in kwargs:
            self._clip_norm = kwargs["clip_norm"]
        if "clip_val" in kwargs:
            self._clip_val = kwargs["clip_val"]

    @property
    def most_rep(self):
        return self._most_rep

    @property
    def role(self):
        return self._role

    def compute_loss(self, y, y_pred):
        # Calculate loss between predicted and actual using selected loss function
        batch_loss = self.loss(y_pred, y)

        if self._role is not None:
            # Calculate most representative data points. We regard data points that have a 
            # high loss to be most representative
            if batch_loss.shape[1] > 1:
                temp = np.mean(batch_loss, axis=1)
                self._most_rep = np.argsort(-temp.flatten())[0:self._n_most_rep]
            else:
                self._most_rep = np.argsort(-batch_loss.flatten())[0:self._n_most_rep]
            # Calculate worker loss - this is aggregated
            batch_loss = np.mean(abs(batch_loss))

        return batch_loss

    def compute_gradients(self, X, y, y_pred, theta):
        """Computes gradients of parameter matrix
        
        Args:
            X (numpy.ndarray): Feature matrix
            theta (numpy.ndarray): Parameter matrix
        
        Returns:
            d_theta (numpy.ndarray): Gradient parameter matrix
        """
        # Calculate error/residuals
        e = (y_pred - y)
        # e = y_pred * (1 - y_pred) * e

        d_theta = self.grad(X, e)

        return d_theta

    def apply_gradients(self, d_theta, theta, N):
        """Applies gradients by updating parameter matrix
        
        Args:
            d_theta (numpy.ndarray): Gradient parameter matrix
        
        Returns:
            theta (numpy.ndarray): Updated parameter matrix
        """

        theta = theta - self.learning_rate * 1 / N * d_theta

        return theta

    def minimize(self, X, y, y_pred, theta, precomputed_gradients=None, N=None):
        """Minimizes gradients. Computes loss from actual and predicted, computes gradients and applies gradients
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Actual labels
            y_pred (numpy.ndarray): Predicted labels
            theta (numpy.ndarray): Parameter matrix
        
        Returns:
            theta (numpy.ndarray): Updated paramter matrix
            batch_loss (float): Loss for current predictions and labels
        """

        if precomputed_gradients is None:
            # Calculate loss
            batch_loss = self.compute_loss(y, y_pred)

            # Get gradients
            d_theta = self.compute_gradients(X, y, y_pred, theta)

            # Clip gradients to prevent getting too large
            if self._clip_norm is not None:
                d_theta = d_theta * self._clip_norm / np.linalg.norm(d_theta)

            # Apply them
            theta = self.apply_gradients(d_theta, theta, N)
            return theta, d_theta, batch_loss
        else:
            d_theta = precomputed_gradients
            # Apply them
            theta = self.apply_gradients(d_theta, theta, X.shape[0])
            return theta

class AdamOptimizer(Optimizer):
    """Adam gradient descent optimizer
    
    Attributes:
        learning_rate (float): Short description of attribute (default: 0.001)
        beta1 (float): First moment hyperparameter (default: 0.9)
        beta2 (float): Second moment hyperparameter (default: 0.999)
        epsilon (float): Some arbritrary epsilon so we do not divide by 0 (default: 1e-8)
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):
        super().__init__(learning_rate)

    def minimize(self, X, y, y_pred, theta):
        pass

class ParallelSGDOptimizer(Optimizer):
    """Parallel stochastic gradient descent optimizer

    Attributes:
        learning_rate (float): Rate at which the descent algorithm descends to find the         global minima (default: 0.1)
        n_most_representative (int): No. of most representative data points we wish to track    (default: 100)
    """
    def __init__(self, loss=None, grad=None, role="master", 
    learning_rate=0.1, n_most_representative=100):
        super().__init__(loss, grad, learning_rate)
        self.n_most_representative = n_most_representative
        self.role = role

        # If role is a worker loss and gradient function should not be none
        if self.role != "master":
            assert self.loss is not None
            assert self.grad is not None

    def compute_loss(self, y, y_pred):
        """Computes loss for parallel sgd
        
        Long description
        
        Args:
            loss (func): Loss function we will use to calculate batch loss
            y_pred (numpy.ndarray): Predictions
            y (numpy.ndarray): Actual values/labels
        
        Returns:
            batch_loss (float): Loss for current epoch
            most_representative (numpy.ndarray): Most representative data points. Data points   with a significant loss are considered to be the most representative
        """
        # Calculate loss for each point
        batch_loss = self.loss(y_pred, y)
        # Calculate most representative data points. We regard data points that have a 
        # high loss to be most representative
        most_representative = np.argsort(-batch_loss.flatten())[0:self.n_most_representative]
        # Calculate worker loss - this is aggregated
        batch_loss = np.mean(batch_loss)

        return batch_loss, most_representative

    def compute_gradients(self, X, y, y_pred, theta):
        """Computes gradients of parameter matrix
        
        Args:
            X (numpy.ndarray): Feature matrix
            theta (numpy.ndarray): Parameter matrix
        
        Returns:
            d_theta (numpy.ndarray): Gradient parameter matrix
        """
        
        # Calculate error/residuals
        e = (y_pred - y)

        n_features, n_classes = theta.shape
        d_theta = np.zeros((n_features, n_classes))

        # To be moved to optimizer
        for k in np.arange(n_classes):
            d_theta[:, k] = self.grad(X, e[:, np.newaxis, k])

        return d_theta

    def apply_gradients(self, d_theta, theta):
        """Applies gradients by updating parameter matrix
        
        Args:
            d_theta (numpy.ndarray): Gradient parameter matrix
        
        Returns:
            theta (numpy.ndarray): Updated parameter matrix
        """
        _, n_classes = theta.shape
        # Update the global parameters with weighted error
        for k in np.arange(n_classes):
            theta[:, k] = theta[:, k] - self.learning_rate * d_theta[:, k]

        return theta

    def minimize(self, X, y, y_pred, theta, d_theta=None):
        """Minimizes loss function
        
        Functions are split depending on the role. A master will only apply gradients and the worker will do the actual gradient computation, return the loss and most representative data points.
        
        Args:
            loss (func): Loss function we will use to calculate batch loss
            X (numpy.ndarray): Feature matrix
            y_pred (numpy.ndarray): Predictions
            theta (numpy.ndarray): Parameter matrix
            d_theta (numpy.ndarray): Optional gradient parameter matrix only required for the   master
        
        Returns:
            theta (numpy.ndarray): Returns theta if role is a master
            d_theta (numpy.ndarray): Returns computed gradients if role is a worker
            batch_loss (numpy.ndarray): Returns computed loss if role is a worker
            most_representative (numpy.ndarray): Returns most_representative data points if if role is a worker
        """
        if self.role == "master":
            assert d_theta is not None
            theta = self.apply_gradients(d_theta, theta)
            return theta
        else:
            # Role of the worker
            self._logger.info("Worker computing loss and gradients")
            batch_loss, most_representative = self.compute_loss(y, y_pred)
            d_theta = self.compute_gradients(X, y, y_pred, theta)
            return d_theta, batch_loss, most_representative
