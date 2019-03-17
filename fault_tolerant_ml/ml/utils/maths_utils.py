import numpy as np

def sigmoid(s):
    """
    Sigmoid calculation for the matrix passed into this function
    """
    return 1./(1. + np.exp(-s))