"""Base class for estimators
"""
from fault_tolerant_ml.utils import maths
from ..base import BaseEstimator, ClassifierMixin


class LinearModel(BaseEstimator):
    """Base class for Linear Models
    """
    def __init__(self):
        super().__init__(optimizer=None)
        self.W = None

    def _hypothesis(self, X):
        # return np.dot(X, self.W)
        return X @ self.W

    def fit(self, X, y):
        """Fit data and train
        """
        raise NotImplementedError("Child must override")

    def predict(self, X):
        """Predict labels for given dataset
        """
        return self._hypothesis(X)

class LinearRegression(LinearModel):
    """Linear Regression
    """
    def __init__(self):
        super(LinearRegression).__init__()

    def fit(self, X, y):
        pass

class LinearClassifierMixin(ClassifierMixin):
    """LinearClassiferMixin
    """
    def __init__(self):
        super(LinearClassifierMixin).__init__()

    def _hypothesis(self, X):
        # s = np.dot(X, self.W)
        self._logger.debug(f"type(X)={type(X)}, type(self.W)={type(self.W)}")
        self._logger.debug(f"X.dtype={X.dtype}, self.W.dtype={self.W.dtype}")
        s = X @ self.W
        return maths.sigmoid(s)

    def predict(self, X):
        """Predict labels given dataset X
        """
        scores = self._hypothesis(X)
        # labels = np.argmax(scores, axis=1)
        return scores
