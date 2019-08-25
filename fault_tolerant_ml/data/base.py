"""Base class for all ftml Datasets
"""
from __future__ import absolute_import, print_function

import numpy as np
from sklearn.model_selection import train_test_split


class Dataset():
    """Base class for all datasets for fault_tolerant_ml
    """

    # pylint: disable=too-many-instance-attributes
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.n_samples: int = 0
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    @staticmethod
    def one_hot(labels):
        """Returns one hot encoded representation of data 
        
        Args:
            labels (numpy.ndarray): nx1 numpy vector of labels to be one-hot
            encoded where n is the number of samples in the dataset
        
        Returns:
            one_hot (numpy.ndarray): nxd one-hot encoded numpy matrix
            where d is the number of unique labels in the 1d numpy vector
        """
        unique_labels = np.unique(labels)
        return (unique_labels == labels[:, np.newaxis]).astype(int)

    @staticmethod
    def next_batch(X, y, batch_size, shuffle=True, overlap=0.0):
        """
        Returns a batch of the data of size batch_size. If the
        batch_size=1 then it is normal stochastic gradient descent
        Args:
            X (ftml.Tensor): Entire feature dataset
            y (ftml.Tensor): Entire label dataset
            batch_size (int): selected batch_size; default: 1
            shuffle (bool): Whether or not to shuffle data batch (default: True)
            overlap (float): Percentage of overlap for partitions (default: 0.0)

        Returns:
            X_batch: random batch of X
            y_batch: random batch of y
        """
        if shuffle:
            idxs = np.arange(X.shape[0])
            np.random.shuffle(idxs)
            X = X[idxs]
            y = y[idxs]

        # Calculate no. of points that will overlap given a percentage
        n_overlap = int((1 + overlap) * batch_size)

        # loop over our dataset `X` in mini-batches of size `batchSize`
        for i in np.arange(0, X.shape[0], batch_size):
            # yield a tuple of the current batched data and labels
            # yield (X[i:i + batch_size], y[i:i + batch_size])
            if i > 0:
                start = i
            else:
                start = 0
            end = start + n_overlap
            X_batch = X[start:end]
            y_batch = y[start:end]
            if end > X.shape[0] and overlap > 0.0:
                # We need to circle back and stack the points from worker 1 onto these points
                end = end - X.shape[0]
                X_batch = np.vstack([X_batch, X[0:end]])
                y_batch = np.vstack([y_batch, X[0:end]])
            yield X_batch, y_batch

    def prepare_data(self):
        """Overridden function to prepare data
        """
        raise NotImplementedError("Child required to implement prepare data method")

    def update_xy(self, idxs):
        """Update x and y given new indices
        """
        X_train = self.X_train[idxs]
        y_train = self.y_train[idxs]
        self.n_samples = X_train.shape[0]
        return X_train, y_train

    def do_split(self, X, y, test_size=0.3):
        """Returns a dictionary of the train test split numpy arrays
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        data = {'training' : {'features' : X_train, 'labels' : y_train},
                'testing' : {'features' : X_test, 'labels' : y_test}}

        return data
