import logging

from fault_tolerant_ml.metrics import accuracy_score
from fault_tolerant_ml.distribute.strategy import LocalStrategy

class BaseEstimator(object):

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

    def encode(self):
        return f"{self.strategy.n_workers}-{self.strategy.scenario}-{self.strategy.remap}-{self.strategy.quantize}-{self.optimizer.n_most_rep}-{self.strategy.comm_period}-{self.optimizer.mu_g}-{self.strategy.send_gradients}"

class ClassifierMixin(object):

    def __init__(self):
        
        self._logger = logging.getLogger(f"ftml.models.{self.__class__.__name__}")

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

class RegressorMixin(object):

    def __init__(self):
        pass

    def score(self, X, y):
        pass