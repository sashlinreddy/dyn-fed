import numpy as np
import logging
import time

# Local
from ..base import BaseEstimator
from .base import LinearClassifierMixin
from fault_tolerant_ml.metrics import accuracy_scorev2
from fault_tolerant_ml.distribute import Master, Worker

class LogisticRegression(BaseEstimator, LinearClassifierMixin):

    def __init__(self, optimizer, strategy, max_iter=300, shuffle=True, verbose=10, encode_name=None):
        super().__init__(optimizer, strategy, encode_name=encode_name)
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.iter = 0
        self.verbose = verbose

        self._logger = logging.getLogger(f"ftml.{self.__class__.__name__}")

        self._setup()

    def _setup(self):

        if self.strategy.name == "local":
            pass # TODO: Local strategy
        elif self.strategy.name == "master_worker":
            if self.strategy.role == "master":
                # Setup master
                self._master = Master(
                    model=self,
                    verbose=self.verbose,
                )

                self._logger.info("Connecting master sockets")
                self._master.connect()
                # setattr(master, "train_iter", train_iter)
                # time.sleep(1)
            else:

                time.sleep(2)

                self._worker = Worker(
                    model=self,
                    verbose=self.verbose,
                    id=self.strategy.identity
                )

                self._worker.connect()

    def _fit_local(self, X=None, y=None):
        """Training for local strategy
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y.argmax(axis=1))
        n_classes = len(self.classes_)

        # Initialize parameters
        self.theta = (np.random.randn(n_features, n_classes) * 0.01).astype(np.float32)
        
        i = 0
        delta = 1.0
        while i < self.max_iter:
            
            # if self.shuffle:
            #     idxs = np.arange(X.shape[0])
            #     np.random.shuffle(idxs)
            #     X = X[idxs]
            #     y = y[idxs]
                
            # Create a copy of theta so we can calculate change in theta
            theta_p = self.theta.copy()
            # Get predictions for current theta
            y_pred = self.predict(X)
            # Calculate and apply gradients
            self.theta, d_theta, epoch_loss = self.optimizer.minimize(X, y, y_pred, self.theta)
            # Calculate change in theta
            delta = np.max(np.abs(theta_p - self.theta))
            acc = accuracy_scorev2(y, y_pred)

            if i % 100 == 0:
                print(f"Iteration={i}, delta={delta:.3f}, loss={epoch_loss:.3f}, train acc={acc:.3f}")

            i += 1

    def _fit_mw(self, X=None, y=None):
        """Training logistic regression using the master worker strategy
        """
        if self.strategy.role == "master":
            # Master training
            self._master.train(X)
        else:
            # Worker training
            self._worker.train()

    def fit(self, X=None, y=None):
        """Training for estimating parameters
        """
        if self.strategy.name == 'local':
            self._fit_local(X, y)
        elif self.strategy.name == "master_worker":
            self._fit_mw(X, y)
        
    def predict_proba(self, X):
        pass

    def predict_log_proba(self, X):
        pass

    def plot_metrics(self):
        """Plots metrics
        """
        if self.strategy.name == "master_worker":
            self._master.plot_metrics()