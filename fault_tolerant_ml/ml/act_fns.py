import numpy as np

# Logistic regression
def sigmoid(s):
    """
    Sigmoid calculation for the matrix passed into this function
    """
    return 1./(1. + np.exp(-s))