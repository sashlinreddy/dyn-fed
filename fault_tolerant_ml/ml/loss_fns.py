import numpy as np

# Logistic regression
def single_cross_entropy_loss(y_pred, y):
    return -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)

def cross_entropy_loss(y_pred, y):
    return np.mean(single_cross_entropy_loss(y_pred, y))

def cross_entropy_gradient(X, e):
    # d_theta = np.mean(X * e, axis=0).T
    d_theta = np.dot(X.T, e)
    # d_theta = np.sum(X * e, axis=0)
    return d_theta

# Linear regression
def mse(y_pred, y):
    """Mean squared error cost function
    .. math::
        \frac{1}/{2m}(h(x) - y)^2
    """
    m, _ = np.shape(y_pred)
    z = ( y_pred - y )
    cost = 1.0 / (2.0 * m) * np.sum(z**2)
    return cost

def mse_gradient(X, e):
    d_theta = np.mean(X * e, axis=0).T
    return d_theta
