import numpy as np

# Local
from ..base import BaseEstimator, ClassifierMixin
from ..utils import maths_utils

class LinearModel(BaseEstimator):
    """Base class for Linear Models
    """
    def __init__(self):
        self.theta = None

    def _hypothesis(self, X):
        return np.dot(X, self.theta)

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
        s = np.dot(X, self.theta)
        return maths_utils.sigmoid(s)

    def predict(self, X):
        scores = self._hypothesis(X)
        # labels = np.argmax(scores, axis=1)
        return scores