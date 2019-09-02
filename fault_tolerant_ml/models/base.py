"""Base class for models
"""
from __future__ import absolute_import, division, print_function

import logging

from fault_tolerant_ml.distribute.strategy import LocalStrategy
from fault_tolerant_ml.metrics import accuracy_score


class BaseEstimator():
    """Base Estimator
    """
    def __init__(self, optimizer, strategy=None, **kwargs):
        self.encode_name = None
        if "encode_name" in kwargs:
            self.encode_name = kwargs["encode_name"]
        self.optimizer = optimizer

        if strategy is None:
            self.strategy = LocalStrategy(config={})
        else:
            self.strategy = strategy

        self._logger = logging.getLogger(f"ftml.models.{self.__class__.__name__}")

class ClassifierMixin():
    """ClassiferMixin for classfication models
    """
    def __init__(self):
        
        self._logger = logging.getLogger(f"ftml.models.{self.__class__.__name__}")

    def score(self, X, y):
        """Returns score of corresponding model given X and y
        """
        return accuracy_score(y, self.predict(X))

class RegressorMixin():
    """RegressorMixin for regression models
    """
    def __init__(self):
        pass

    def score(self, X, y):
        """Returns score of coressponding model given X and y
        """
