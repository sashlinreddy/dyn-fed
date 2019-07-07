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

    @property
    def name(self):
        """Name of optimizer
        """
        raise NotImplementedError("Child must override this method")

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
        self._mu_g = 1.0 / self.learning_rate
        if "n_most_rep" in kwargs:
            self._n_most_rep = kwargs["n_most_rep"]
        if "role" in kwargs:
            self._role = kwargs["role"]
        if "clip_norm" in kwargs:
            self._clip_norm = kwargs["clip_norm"]
        if "clip_val" in kwargs:
            self._clip_val = kwargs["clip_val"]
        if "mu_g" in kwargs:
            self._mu_g = kwargs["mu_g"]

    @property
    def name(self):
        return "sgd"

    @property
    def most_rep(self):
        return self._most_rep

    @property
    def n_most_rep(self):
        return self._n_most_rep

    @property
    def mu_g(self):
        return self._mu_g

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

        d_theta = 1 / X.shape[0] * self.grad(X, e)

        # self._logger.debug(f"d_theta={d_theta} \n, d_theta.shape={d_theta.shape}")
        # assert np.all(d_theta == 0.0)

        return d_theta

    def apply_gradients(self, d_theta, theta, N, theta_g=None):
        """Applies gradients by updating parameter matrix
        
        Args:
            d_theta (numpy.ndarray): Gradient parameter matrix
        
        Returns:
            theta (numpy.ndarray): Updated parameter matrix
        """

        if self.role != "worker":
            theta = theta - self.learning_rate * d_theta
        else:
            theta = (
                (1 - self._mu_g * self.learning_rate) * theta - 
                (self.learning_rate * d_theta) + 
                (self._mu_g * theta_g * self.learning_rate)
            )

        return theta

    def minimize(self, X, y, y_pred, theta, precomputed_gradients=None, N=None, theta_g=None, **kwargs):
        """Minimizes gradients. Computes loss from actual and predicted, computes gradients and applies gradients
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Actual labels
            y_pred (numpy.ndarray): Predicted labels
            theta (numpy.ndarray): Parameter matrix
        
        Returns:
            theta (numpy.ndarray): Updated parameter matrix
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

            self._logger.info(f'n_samples={N}')
            # Apply them
            theta = self.apply_gradients(d_theta, theta, N, theta_g=theta_g)
            # theta = self.apply_gradients(d_theta, theta, X.shape[0])
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
    def __init__(self, loss, grad, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, **kwargs):
        super().__init__(loss, grad, learning_rate)

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m_t = None
        self.v_t = None
        self._role = None
        self._most_rep = None
        self._clip_norm = None
        self._clip_val = None
        self._n_most_rep = 0
        self._mu_g = 1.0 / self.learning_rate
        if "n_most_rep" in kwargs:
            self._n_most_rep = kwargs["n_most_rep"]
        if "role" in kwargs:
            self._role = kwargs["role"]
        if "mu_g" in kwargs:
            self._mu_g = kwargs["mu_g"]

    @property
    def name(self):
        return "adam"

    @property
    def most_rep(self):
        return self._most_rep

    @property
    def n_most_rep(self):
        return self._n_most_rep

    @property
    def mu_g(self):
        return self._mu_g

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

        d_theta = 1 / X.shape[0] * self.grad(X, e)

        return d_theta

    def apply_gradients(self, d_theta, theta, iteration, theta_g=None):
        """Applies gradients by updating parameter matrix
        
        Args:
            d_theta (numpy.ndarray): Gradient parameter matrix
        
        Returns:
            theta (numpy.ndarray): Updated parameter matrix
        """

        # if self.role != "worker":
        #     theta = theta - self.learning_rate * 1 / N * d_theta
        # else:
        #     theta = (
        #         (1 - self._mu_g * self.learning_rate) * theta - 
        #         (self.learning_rate * 1 / N * d_theta) + 
        #         (self._mu_g * theta_g * self.learning_rate)
        #     )

        if self.m_t is None:
            self.m_t = np.zeros_like(d_theta)
        if self.v_t is None:
            self.v_t = np.zeros_like(d_theta)

        # Update biased first moment estimate
        self.m_t = self.beta_1 * self.m_t + (1. - self.beta_1) * d_theta
        # Update biased second raw moment estimate
        self.v_t = self.beta_2 * self.v_t + (1. - self.beta_2) * d_theta**2

        # Bias correction
        m_t_corrected = self.m_t / (1. - self.beta_1**(iteration))
        v_t_corrected = self.v_t / (1. - self.beta_2**(iteration))

        # Update weights
        theta = theta - self.learning_rate * m_t_corrected / (np.sqrt(v_t_corrected) + self.epsilon)

        return theta

    def minimize(self, X, y, y_pred, theta, precomputed_gradients=None, N=None, theta_g=None, **kwargs):
        """Minimizes gradients. Computes loss from actual and predicted, computes gradients and applies gradients
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Actual labels
            y_pred (numpy.ndarray): Predicted labels
            theta (numpy.ndarray): Parameter matrix
        
        Returns:
            theta (numpy.ndarray): Updated parameter matrix
            batch_loss (float): Loss for current predictions and labels
        """

        if "iteration" in kwargs:
            iteration = kwargs["iteration"]

        if precomputed_gradients is None:
            # Calculate loss
            batch_loss = self.compute_loss(y, y_pred)

            # Get gradients
            d_theta = self.compute_gradients(X, y, y_pred, theta)

            self._logger.info(f'n_samples={N}')
            # Apply them
            theta = self.apply_gradients(d_theta, theta, iteration)
            # theta = self.apply_gradients(d_theta, theta, X.shape[0])
            return theta, d_theta, batch_loss
        else:
            d_theta = precomputed_gradients
            # Apply them
            theta = self.apply_gradients(d_theta, theta, X.shape[0])
            return theta