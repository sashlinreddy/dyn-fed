import numpy as np
from fault_tolerant_ml.ml.activations import act_fns as af

# Logistic regression
def log_hypothesis(X, theta):
    """Hypothesis for logistic regression
    """
    s = np.dot(X, theta)
    return af.sigmoid(s)

# Linear regression
def lin_hypothesis(X, theta):
    """Hypothesis for linear regression - convex optimization problem
    """
    s = np.dot(X, theta)
    return s
