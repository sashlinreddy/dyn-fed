"""Metrics to assess performance on classification task given class prediction"""

import numpy as np

def confusion_matrix(y, y_pred):
    """Tests the learned hypothesis.
    
    This function tests the learned hypothesis, and produces
    the confusion matrix as the output. The diagonal elements of the confusion
    matrix refer to the number of correct classifications and non-diagonal elements
    fefer to the number of incorrect classifications.

    Args:
        y (numpy.ndarray): Label matrix
        y_pred (numpy.ndarray): Predictions

    Returns:
        confusion_matrix (numpy.ndarray): Confusion matrix of learned hypothesis
    """

    n_labels = np.unique(y.argmax(axis=1)).size

    confusion_matrix = np.zeros((n_labels, n_labels), dtype=int)

    if n_labels > 2:
        y_pred_class_index = np.argmax(y_pred, axis=1)
        y_class_index = np.argmax(y, axis=1)
        rows = y_pred_class_index.shape[0]
        for i in np.arange(0,rows):
            confusion_matrix[y_class_index[i], y_pred_class_index[i]] = confusion_matrix[y_class_index[i], y_pred_class_index[i]] + 1
    else:
        y_pred[y_pred>0.5] = 1
        y_pred[y_pred<=0.5] = 0
        rows = y_pred.shape[0]
        for i in np.arange(0,rows):
            confusion_matrix[int(y[i]), int(y_pred[i])] = confusion_matrix[int(y[i]), int(y_pred[i])] + 1

    return confusion_matrix

def accuracy_score(y, y_pred):
    y_pred[y_pred>0.5] = 1
    y_pred[y_pred<=0.5] = 0
    return np.sum(y_pred==y) / y.shape[0]