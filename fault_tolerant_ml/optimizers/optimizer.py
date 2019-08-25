"""Contains logic for all optimizers
"""
import logging

import numpy as np  # Linear Algebra


class OptimizerV1():
    """Base class for machine learning optimizers
    
    Attributes:
        loss (ftml.Loss): Loss function
        learning_rate (float): The rate at which the machine learning algorithm
        will learn and how fast our descent method will go down the slope to
        find it's global minima
    """
    def __init__(self, loss, learning_rate, **kwargs):
        self.learning_rate = learning_rate
        self.loss = loss

        self._role = None
        self._most_rep = None
        self._n_most_rep = 0
        self._mu_g = 1.0 / self.learning_rate
        if "n_most_rep" in kwargs:
            self._n_most_rep = kwargs["n_most_rep"]
        if "role" in kwargs:
            self._role = kwargs["role"]
        if "mu_g" in kwargs:
            self._mu_g = kwargs["mu_g"]

        self._logger = logging.getLogger(f"ftml.optimizers.{self.__class__.__name__}")

    def __repr__(self):
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate})"

    @property
    def name(self):
        """Name of optimizer
        """
        raise NotImplementedError("Child must override this method")

    @property
    def most_rep(self):
        """No. of most representative data points
        """
        return self._most_rep

    @property
    def n_most_rep(self):
        """No. of most representative data points
        """
        return self._n_most_rep

    @property
    def mu_g(self):
        """No. of most representative data points
        """
        return self._mu_g

    @property
    def role(self):
        """No. of most representative data points
        """
        return self._role

    def compute_loss(self, y, y_pred):
        """Compute loss to be overridden by children
        """
        raise NotImplementedError("Child must override this method")

    def compute_gradients(self, X, y, y_pred, W):
        """Compute gradients to be overridden by children
        """
        raise NotImplementedError("Child must override this method")

    def apply_gradients(self, W, **kwargs):
        """Compute gradients to be overridden by children
        """
        raise NotImplementedError("Child must override this method")

    def minimize(self, X, y, y_pred, W, **kwargs):
        """Minimizes objective function to be overridden by children
        
        This function will perform the update of the parameters in order to
        get close to the  global minima
        """
        raise NotImplementedError("Child must override this method")

class Optimizer():
    """Base class for machine learning optimizers
    
    Attributes:
        loss (ftml.Loss): Loss function
        learning_rate (float): The rate at which the machine learning algorithm
        will learn and how fast our descent method will go down the slope to
        find it's global minima
    """
    def __init__(self, loss, learning_rate, **kwargs):
        self.learning_rate = learning_rate
        self.loss = loss

        self._role = None
        self._most_rep = None
        self._n_most_rep = 0
        self._mu_g = 1.0 / self.learning_rate
        if "n_most_rep" in kwargs:
            self._n_most_rep = kwargs["n_most_rep"]
        if "role" in kwargs:
            self._role = kwargs["role"]
        if "mu_g" in kwargs:
            self._mu_g = kwargs["mu_g"]

        self._logger = logging.getLogger(f"ftml.optimizers.{self.__class__.__name__}")

    def __repr__(self):
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate})"

    @property
    def name(self):
        """Name of optimizer
        """
        raise NotImplementedError("Child must override this method")

    @property
    def most_rep(self):
        """No. of most representative data points
        """
        return self._most_rep

    @property
    def n_most_rep(self):
        """No. of most representative data points
        """
        return self._n_most_rep

    @property
    def mu_g(self):
        """No. of most representative data points
        """
        return self._mu_g

    @property
    def role(self):
        """No. of most representative data points
        """
        return self._role

    def compute_loss(self, y, y_pred):
        """Compute loss to be overridden by children
        """
        raise NotImplementedError("Child must override this method")

    def compute_gradients(self, model, y, y_pred):
        """Compute gradients to be overridden by children
        """
        raise NotImplementedError("Child must override this method")

    def apply_gradients(self, model):
        """Compute gradients to be overridden by children
        """
        raise NotImplementedError("Child must override this method")

    def minimize(self, model, y, y_pred, **kwargs):
        """Minimizes objective function to be overridden by children
        
        This function will perform the update of the parameters in order to
        get close to the  global minima
        """
        raise NotImplementedError("Child must override this method")

class SGD(Optimizer):
    """Stochastic gradient descent optimizer

    Attributes:
        loss (ftml.Losses): Loss function
        learning_rate (float): Learning rate
    """
    def __init__(self, loss, learning_rate=0.1, **kwargs):
        super().__init__(loss, learning_rate, **kwargs)
        # self.learning_rate = learning_rate
        # self.loss = loss

    @property
    def name(self):
        return "sgd"

    def compute_loss(self, y, y_pred):
        # Calculate loss between predicted and actual using selected loss function
        batch_loss = self.loss(y, y_pred, reduce=False)

        if self._role != "master":
            # Calculate most representative data points. We regard data points that have a 
            # high loss to be most representative
            if batch_loss.shape[1] > 1:
                temp = np.mean(batch_loss, axis=1).data
                self._most_rep = np.argsort(-temp.flatten())[0:self._n_most_rep]
            else:
                self._most_rep = np.argsort(-batch_loss.flatten())[0:self._n_most_rep]
            # Calculate worker loss - this is aggregated
            batch_loss = np.mean(abs(batch_loss))

        return batch_loss
        
    def compute_gradients(self, model, y, y_pred):
        """Compute gradients for each layer of the model and updates gradients for the parameters

        Args:
            y (fault_tolerant_ml.operators.Tensor): Actual labels
            y_pred (fault_tolerant_ml.operators.Tensor): Predicted labels
        """
        n_layers = len(model.layers)
        output_layer = model.layers[-1]
        m = model.layers[0].x.shape[0]
        # For each output unit, calculate it's error term
        delta = self.loss.grad(y, y_pred) * output_layer.act_fn.grad(output_layer.z)
        output_layer.W.grad = (1 / m) * output_layer.x.T @ delta
        output_layer.b.grad = (1 / m) * np.sum(delta, axis=0, keepdims=True)

        # For hidden units, calculate error term
        for i in np.arange(n_layers - 2, -1, -1):
            delta = (delta @ model.layers[i+1].W.T) * model.layers[i].act_fn.grad(model.layers[i].z)
            model.layers[i].W.grad = (1 / m) * (model.layers[i].x.T @ delta)
            model.layers[i].b.grad = (1 / m) * np.sum(delta, axis=0, keepdims=True)
            
    def apply_gradients(self, model):
        """Apply gradients to the parameters in each layer

        Args:
            model (fault_tolerant_ml.Model): Tensor model containing multiple layers
        """
        for layer in model.layers:
            layer.W.data = layer.W.data - self.learning_rate * layer.W.grad.data
            layer.b.data = layer.b.data - self.learning_rate * layer.b.grad.data

    def minimize(self, model, y, y_pred, **kwargs):

        # Compute loss
        batch_loss = self.compute_loss(y, y_pred)

        # Backprop
        self.compute_gradients(model, y, y_pred)

        # Update gradients
        self.apply_gradients(model)

        return batch_loss

class SGDOptimizer(OptimizerV1):
    """Stochastic gradient descent optimizer
    """
    def __init__(self, loss, learning_rate=0.1, **kwargs):
        super().__init__(loss, learning_rate, **kwargs)
        self._clip_norm = None
        self._clip_val = None
        if "clip_norm" in kwargs:
            self._clip_norm = kwargs["clip_norm"]
        if "clip_val" in kwargs:
            self._clip_val = kwargs["clip_val"]

    def __repr__(self):
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate}, mu_g={self.mu_g})"

    @property
    def name(self):
        return "sgd"

    def compute_loss(self, y, y_pred):
        # Calculate loss between predicted and actual using selected loss function
        batch_loss = self.loss(y, y_pred, reduce=False)

        if self._role != "master":
            # Calculate most representative data points. We regard data points that have a 
            # high loss to be most representative
            if batch_loss.shape[1] > 1:
                temp = np.mean(batch_loss, axis=1).data
                self._most_rep = np.argsort(-temp.flatten())[0:self._n_most_rep]
            else:
                self._most_rep = np.argsort(-batch_loss.flatten())[0:self._n_most_rep]
            # Calculate worker loss - this is aggregated
            batch_loss = np.mean(abs(batch_loss))

        return batch_loss

    def compute_gradients(self, X, y, y_pred, W):
        """Computes gradients of parameter matrix
        
        Args:
            X (numpy.ndarray): Feature matrix
            W (numpy.ndarray): Parameter matrix
        
        Returns:
            d_W (numpy.ndarray): Gradient parameter matrix
        """
        # Calculate error/residuals
        cost_prime = self.loss.grad(y, y_pred)
        # Calculate error term
        delta = y_pred * (1 - y_pred) * cost_prime
        d_W = 1 / X.shape[0] * (X.T @ delta)

        # self._logger.debug(f"d_W={d_W} \n, d_W.shape={d_W.shape}")
        # assert np.all(d_W == 0.0)

        return d_W

    def apply_gradients(self, W, **kwargs):
        """Applies gradients by updating parameter matrix
        
        Args:
            d_W (numpy.ndarray): Gradient parameter matrix
        
        Returns:
            W (numpy.ndarray): Updated parameter matrix
        """
        W_g = kwargs.get("W_g")

        if self.role != "worker":
            W = W - self.learning_rate * W.grad
        else:
            W = (
                (1 - self._mu_g * self.learning_rate) * W - 
                (self.learning_rate * W.grad) + 
                (self._mu_g * W_g * self.learning_rate)
            )

        return W

    def minimize(self, X, y, y_pred, W, **kwargs):
        """Minimizes gradients. Computes loss from actual and predicted, 
        computes gradients and applies gradients
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Actual labels
            y_pred (numpy.ndarray): Predicted labels
            W (numpy.ndarray): Parameter matrix
        
        Returns:
            W (numpy.ndarray): Updated parameter matrix
            batch_loss (float): Loss for current predictions and labels
        """
        N = kwargs.get("N")
        precomputed_gradients = kwargs.get("precomputed_gradients")
        W_g = kwargs.get("W_g")
        
        if precomputed_gradients is None:
            # Calculate loss
            batch_loss = self.compute_loss(y, y_pred)

            # Get gradients
            W.grad = self.compute_gradients(X, y, y_pred, W)

            self._logger.info(f'n_samples={N}')
            # Apply them
            W = self.apply_gradients(W, W_g=W_g)
            return W, batch_loss
        else:
            W.grad = precomputed_gradients
            # Apply them
            W = self.apply_gradients(W)
            return W

class AdamOptimizer(OptimizerV1):
    """Adam gradient descent optimizer
    
    Attributes:
        learning_rate (float): Short description of attribute (default: 0.001)
        beta1 (float): First moment hyperparameter (default: 0.9)
        beta2 (float): Second moment hyperparameter (default: 0.999)
        epsilon (float): Some arbritrary epsilon so we do not divide by 0 (default: 1e-8)
    """
    def __init__(self, loss, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-08, **kwargs):
        super().__init__(loss, learning_rate, **kwargs)

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m_t = None
        self.v_t = None

    def __repr__(self):
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate}, mu_g={self.mu_g})"

    @property
    def name(self):
        return "adam"

    def compute_loss(self, y, y_pred):
        # Calculate loss between predicted and actual using selected loss function
        batch_loss = self.loss(y, y_pred, reduce=False)

        if self._role != "master":
            # Calculate most representative data points. We regard data points that have a 
            # high loss to be most representative
            if batch_loss.shape[1] > 1:
                temp = np.mean(batch_loss, axis=1).data
                self._most_rep = np.argsort(-temp.flatten())[0:self._n_most_rep]
            else:
                self._most_rep = np.argsort(-batch_loss.flatten())[0:self._n_most_rep]
            # Calculate worker loss - this is aggregated
            batch_loss = np.mean(abs(batch_loss))

        return batch_loss

    def compute_gradients(self, X, y, y_pred, W):
        """Computes gradients of parameter matrix
        
        Args:
            X (numpy.ndarray): Feature matrix
            W (numpy.ndarray): Parameter matrix
        
        Returns:
            d_W (numpy.ndarray): Gradient parameter matrix
        """
        # Calculate error/residuals
        cost_prime = self.loss.grad(y, y_pred)
        # Calculate error term
        delta = y_pred * (1 - y_pred) * cost_prime
        d_W = 1 / X.shape[0] * (X.T @ delta)

        return d_W

    def apply_gradients(self, W, **kwargs):
        """Applies gradients by updating parameter matrix
        
        Args:
            d_W (numpy.ndarray): Gradient parameter matrix
        
        Returns:
            W (numpy.ndarray): Updated parameter matrix
        """

        iteration = kwargs.get("iteration")
        # W_g = kwargs.get("W_g")

        if self.m_t is None:
            self.m_t = W.grad.zeros_like()
        if self.v_t is None:
            self.v_t = W.grad.zeros_like()

        # Update biased first moment estimate
        self.m_t = self.beta_1 * self.m_t + (1. - self.beta_1) * W.grad
        # Update biased second raw moment estimate
        self.v_t = self.beta_2 * self.v_t + (1. - self.beta_2) * W.grad**2

        # Bias correction
        m_t_corrected = self.m_t / (1. - self.beta_1**(iteration))
        v_t_corrected = self.v_t / (1. - self.beta_2**(iteration))

        # Update weights
        W = W - self.learning_rate * m_t_corrected / (np.sqrt(v_t_corrected) + self.epsilon)

        return W

    def minimize(self, X, y, y_pred, W, **kwargs):
        """Minimizes gradients. Computes loss from actual and predicted,
        computes gradients and applies gradients
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Actual labels
            y_pred (numpy.ndarray): Predicted labels
            W (numpy.ndarray): Parameter matrix
        
        Returns:
            W (numpy.ndarray): Updated parameter matrix
            batch_loss (float): Loss for current predictions and labels
        """
        N = kwargs.get("N")
        precomputed_gradients = kwargs.get("precomputed_gradients")
        # W_g = kwargs.get("W_g")
        iteration = kwargs.get("iteration")

        if precomputed_gradients is None:
            # Calculate loss
            batch_loss = self.compute_loss(y, y_pred)

            # Get gradients
            W.grad = self.compute_gradients(X, y, y_pred, W)

            self._logger.info(f'n_samples={N}')
            # Apply them
            W = self.apply_gradients(W, iteration=iteration)
            return W, batch_loss
        else:
            W.grad = precomputed_gradients
            # Apply them
            W = self.apply_gradients(W)
            return W
