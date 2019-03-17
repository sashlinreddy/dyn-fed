import numpy as np

# Logistic regression
def single_cross_entropy_loss(h, y):
    return -y * np.log(h) - (1 - y) * np.log(1 - h)

def cross_entropy_loss(h, y):
    return np.mean(single_cross_entropy_loss(h, y))

def cross_entropy_gradient(X, e):
    d_theta = np.mean(X * e, axis=0).T
    return d_theta

# Linear regression
def mse(h, y):
    """Mean squared error cost function
    .. math::
        \frac{1}/{2m}(h(x) - y)^2
    """
    m, _ = np.shape(h)
    z = ( h - y )
    cost = 1.0 / (2.0 * m) * np.sum(z**2)
    return cost

def mse_gradient(X, e):
    d_theta = np.mean(X * e, axis=0).T
    return d_theta
