import numpy as np

# Local
from ..base import BaseEstimator, ClassifierMixin
from fault_tolerant_ml.utils import maths

class LinearModel(BaseEstimator):
    """Base class for Linear Models
    """
    def __init__(self):
        self.theta = None

    def _hypothesis(self, X):
        # return np.dot(X, self.theta)
        return X @ self.theta

    def fit(self, X, y):
        raise NotImplementedError("Child must override")

    def predict(self, X):
        return self._hypothesis(X)

class LinearRegression(LinearModel):

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

class LinearClassifierMixin(ClassifierMixin):

    def __init__(self):
        pass

    def _hypothesis(self, X):
        # s = np.dot(X, self.theta)
        self._logger.debug(f"type(X)={type(X)}, type(self.theta)={type(self.theta)}")
        self._logger.debug(f"X.dtype={X.dtype}, self.theta.dtype={self.theta.dtype}")
        s = X @ self.theta
        return maths.sigmoid(s)

    def predict(self, X):
        scores = self._hypothesis(X)
        # labels = np.argmax(scores, axis=1)
        return scores