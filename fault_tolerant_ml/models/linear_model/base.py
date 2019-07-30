import numpy as np

# Local
from ..base import BaseEstimator, ClassifierMixin
from fault_tolerant_ml.utils import maths

class LinearModel(BaseEstimator):
    """Base class for Linear Models
    """
    def __init__(self):
        self.W = None

    def _hypothesis(self, X):
        # return np.dot(X, self.W)
        return X @ self.W

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
        # s = np.dot(X, self.W)
        self._logger.debug(f"type(X)={type(X)}, type(self.W)={type(self.W)}")
        self._logger.debug(f"X.dtype={X.dtype}, self.W.dtype={self.W.dtype}")
        s = X @ self.W
        return maths.sigmoid(s)

    def predict(self, X):
        scores = self._hypothesis(X)
        # labels = np.argmax(scores, axis=1)
        return scores