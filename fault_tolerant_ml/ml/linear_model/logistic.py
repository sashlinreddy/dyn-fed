from fault_tolerant_ml.ml.base import BaseEstimator
from fault_tolerant_ml.ml.linear_model.base import LinearClassifierMixin

class LogisticRegression(BaseEstimator, LinearClassifierMixin):

    def __init__(self, optimizer, max_iter):
        self.optimizer = optimizer
        self.max_iter = max_iter

    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Initialize parameters
        self.theta = np.random.randn(n_features, n_classes)
        
        i = 0
        delta = 1.0
        while i < self.max_iter:
            
            # Create a copy of theta so we can calculate change in theta
            theta_p = self.theta.copy()
            # Get predictions for current theta
            y_pred = self.predict(X)
            # Calculate and apply gradients
            self.theta, epoch_loss = self.optimizer.minimize(X, y, y_pred, self.theta)
            # Calculate change in theta
            delta = np.max(np.abs(theta_p - self.theta))

            print(f"Iteration={i}, delta={delta}, loss={epoch_loss}")

            i += 1

    def predict_proba(self, X):
        pass

    def predict_log_proba(self, X):
        pass