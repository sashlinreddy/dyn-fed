"""Metrics for evaluating models
"""
import sys

import numpy as np


def print_confusion_matrix(confusion_matrix, targets):
    """This function performs formatted printing of the confusion matrix.

    Args:
        confusion_matrix (numpy.ndarray): Confusion matrix of learned hypothesis
        targets (numpy.ndarray): Actual targets
    """
    rows, cols = confusion_matrix.shape
    print("Confusion Matrix:")
    sys.stdout.write('%-8s '%' ')
    for r in np.arange(0, rows):
        sys.stdout.write('%-8s '%get_key(r, targets))
    sys.stdout.write('\n')
    for r in np.arange(0, rows):
        sys.stdout.write('%-8s '%get_key(r, targets))
        for c in np.arange(0, cols):
            sys.stdout.write('%-8d '%confusion_matrix[r, c])
        sys.stdout.write('\n')
    total_samples = np.sum(confusion_matrix)
    empirical_error = np.float((total_samples - np.trace(confusion_matrix))) / total_samples
    sys.stdout.write('empirical error = %.4f'%(empirical_error))
    sys.stdout.write('\n')


def get_key(key_value, targets):
    """Return key from dict
    """
    for key in targets.keys():
        if targets[key] == key_value:
            return key

def accuracy(X, y, W, hypothesis):
    """Return accuracy
    """
    y_pred = hypothesis(X, W)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    return np.sum(y_pred == y) / y.shape[0]
