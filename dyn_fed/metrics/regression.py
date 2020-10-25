"""Regression metrics
"""
import numpy as np

def rmse(y, y_pred):
    """Returns accuracy score V2
    """
    return np.sqrt(np.mean((y - y_pred)**2))
