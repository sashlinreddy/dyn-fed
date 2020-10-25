"""Logistic regression class
"""
import time

import numpy as np

from dyn_fed.distribute import Master, Worker
from dyn_fed.metrics import accuracy_scorev2

# Local
from ..base import BaseEstimator
from .base import LinearClassifierMixin


class LogisticRegression(BaseEstimator, LinearClassifierMixin):
    """Logistic regression class

    Attributes:
        max_iter (int): Max no. of iterations
        shuffle (bool): Whether or not to shuffle dataset
        iter (int): Current iteration
        verbose (int): Verbose for logging
        encode_name (str): Encoded name for logging
    """
    def __init__(self, optimizer, strategy, max_iter=300,
                 shuffle=True, verbose=10, encode_name=None):
        super().__init__(optimizer, strategy, encode_name=encode_name)
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.iter = 0
        self.verbose = verbose

        # Model params
        self.W: np.ndarray = None
        self.classes_: np.ndarray = None

        self._setup()

    def _setup(self):

        if self.strategy.name == "local":
            pass # TODO: Local strategy
        elif self.strategy.name == "master_worker":
            if self.strategy.role == "server":
                # Setup server
                self._master = Master(
                    model=self
                )

                self._logger.info("Connecting server sockets")
                self._master.connect()
                # setattr(server, "train_iter", train_iter)
                # time.sleep(1)
            else:

                time.sleep(2)

                self._worker = Worker(
                    model=self,
                    verbose=self.verbose,
                    identity=self.strategy.identity
                )

                self._worker.connect()

    def _fit_local(self, X=None, y=None):
        """Training for local strategy
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y.argmax(axis=1))
        n_classes = len(self.classes_)

        # Initialize parameters
        self.W = (np.random.randn(n_features, n_classes) * 0.01).astype(np.float32)
        
        i = 0
        delta = 1.0
        while i < self.max_iter:
            
            # if self.shuffle:
            #     idxs = np.arange(X.shape[0])
            #     np.random.shuffle(idxs)
            #     X = X[idxs]
            #     y = y[idxs]
                
            # Create a copy of W so we can calculate change in W
            W_p = self.W.copy()
            # Get predictions for current W
            y_pred = self.predict(X)
            # Calculate and apply gradients
            self.W, epoch_loss = self.optimizer.minimize(X, y, y_pred, self.W)
            # Calculate change in W
            delta = np.max(np.abs(W_p - self.W))
            acc = accuracy_scorev2(y.data, y_pred.data)

            if i % 100 == 0:
                self._logger.info(
                    f"Iteration={i}, delta={delta:.3f}, "
                    f"loss={epoch_loss:.3f}, train acc={acc:.3f}"
                )

            i += 1

    def _fit_mw(self, X=None, y=None):
        """Training logistic regression using the server client strategy
        """
        if self.strategy.role == "server":
            # Master training
            self._master.start(X)
        else:
            # Worker training
            self._worker.start()

    def fit(self, X=None, y=None):
        """Training for estimating parameters
        """
        if self.strategy.name == 'local':
            self._fit_local(X, y)
        elif self.strategy.name == "master_worker":
            self._fit_mw(X, y)
        
    def predict_proba(self, X):
        """Returns predicted probabilities
        """

    def predict_log_proba(self, X):
        """Predicts predicted log probabilities
        """
