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
        conf_matrix (numpy.ndarray): Confusion matrix of learned hypothesis
    """

    n_labels = np.unique(y.argmax(axis=1)).size

    conf_matrix = np.zeros((n_labels, n_labels), dtype=int)

    if n_labels > 2:
        y_pred_class_index = np.argmax(y_pred, axis=1)
        y_class_index = np.argmax(y, axis=1)
        rows = y_pred_class_index.shape[0]
        for i in np.arange(0, rows):
            conf_matrix[y_class_index[i], y_pred_class_index[i]] = \
                conf_matrix[y_class_index[i], y_pred_class_index[i]] + 1
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        rows = y_pred.shape[0]
        for i in np.arange(0, rows):
            conf_matrix[int(y[i]), int(y_pred[i])] = \
                conf_matrix[int(y[i]), int(y_pred[i])] + 1

    return conf_matrix

def accuracy_score(y, y_pred):
    """Returns accuracy score
    """
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    return np.sum(y_pred == y) / y.shape[0]

def accuracy_scorev2(y, y_pred):
    """Returns accuracy score V2
    """
    y_pred_ = y_pred.argmax(axis=1)
    y_ = y.argmax(axis=1)
    return np.sum(y_pred_ == y_) / y_.shape[0]