import numpy as np

# Logistic regression
def sigmoid(s):
    """
    Sigmoid calculation for the matrix passed into this function
    """
    s = np.clip(s, -100, 100)
    # print(s)
    return 1./(1. + np.exp(-s))