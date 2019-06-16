import numpy as np

# Local
from ..base import BaseEstimator
from .base import LinearClassifierMixin
from ..metrics_temp import accuracy_scorev2

class LogisticRegression(BaseEstimator, LinearClassifierMixin):

    def __init__(self, optimizer, max_iter=300, shuffle=True):
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.iter = 0

    def fit(self, X, y):
        
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

    def predict_proba(self, X):
        pass

    def predict_log_proba(self, X):
        pass