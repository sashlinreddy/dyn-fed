import numpy as np

def sigmoid(s):
    """
    Sigmoid calculation for the matrix passed into this function
    """
    # print(s)
    return 1./(1. + np.exp(-s))