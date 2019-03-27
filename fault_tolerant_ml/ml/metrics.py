import numpy as np
import sys
from . import hypotheses as hyp

def print_confusion_matrix(confusion_matrix, targets):
    """This function performs formatted printing of the confusion matrix.

    Args:
        confusion_matrix (numpy.ndarray): Confusion matrix of learned hypothesis
        targets (numpy.ndarray): Actual targets
    """
    rows, cols = confusion_matrix.shape
    print("Confusion Matrix:")
    sys.stdout.write('%-8s '%' ')
    for r in np.arange(0,rows):
        sys.stdout.write('%-8s '%get_key(r, targets))
    sys.stdout.write('\n')
    for r in np.arange(0,rows):
        sys.stdout.write('%-8s '%get_key(r, targets))
        for c in np.arange(0,cols):
            sys.stdout.write('%-8d '%confusion_matrix[r,c])
        sys.stdout.write('\n')
    total_samples = np.sum(confusion_matrix)
    empirical_error = np.float((total_samples - np.trace(confusion_matrix))) / total_samples
    sys.stdout.write('empirical error = %.4f'%(empirical_error))
    sys.stdout.write('\n')


def get_key(key_value, targets):
    for key in targets.keys():
        if targets[key] == key_value:
            return key

def test_hypothesis(X, y, theta):
    """Tests the learned hypothesis.
    
    This function tests the learned hypothesis, and produces
    the confusion matrix as the output. The diagonal elements of the confusion
    matrix refer to the number of correct classifications and non-diagonal elements
    fefer to the number of incorrect classifications.

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Label matrix
        theta (numpy.ndarray): dxc matrix where d is the number of features and c is the        number of classes

    Returns:
        confusion_matrix (numpy.ndarray): Confusion matrix of learned hypothesis
    """
    h = hyp.log_hypothesis(X, theta)

    binary_classification = False
    if theta.shape[1] == 1:
        binary_classification = True

    if not binary_classification:
        num_features, num_classes = theta.shape
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        h_class_index = np.argmax(h,axis=1)
        y_class_index = np.argmax(y,axis=1)
        rows = h_class_index.shape[0]
        for i in np.arange(0,rows):
            confusion_matrix[y_class_index[i], h_class_index[i]] = confusion_matrix[y_class_index[i], h_class_index[i]] + 1
    else:
        h[h>0.5] = 1
        h[h<=0.5] = 0
        num_classes = 2
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        rows = h.shape[0]
        for i in np.arange(0,rows):
            confusion_matrix[int(y[i]), int(h[i])] = confusion_matrix[int(y[i]), int(h[i])] + 1

    return confusion_matrix

def accuracy(X, y, theta, hypothesis):
    y_pred = hypothesis(X, theta)
    y_pred[y_pred>0.5] = 1
    y_pred[y_pred<=0.5] = 0
    return np.sum(y_pred==y) / y.shape[0]

