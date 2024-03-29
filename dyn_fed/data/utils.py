"""Utilities for handling data
"""
import numpy as np
from sklearn.model_selection import train_test_split

from dyn_fed.operators import Tensor


def next_batch(X, y, batch_size, shuffle=True, overlap=0.0):
    """
    Returns a batch of the data of size batch_size. If the
    batch_size=1 then it is normal stochastic gradient descent
    Args:
        X (dfl.Tensor): Entire feature dataset
        y (dfl.Tensor): Entire label dataset
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
            # We need to circle back and stack the points from client 1 onto these points
            end = end - X.shape[0]
            X_batch = Tensor(np.vstack([X_batch.data, X.data[0:end]]))
            y_batch = Tensor(np.vstack([y_batch.data, y.data[0:end]]))
        yield X_batch, y_batch

def apply_noniid(X, y, n_partitions):
    """Apply noniid transformation. Makes sure that there is at least 
    2 classes in each partition. Imitates the original federated learning paper
    """
    # Get indices of first encountered unique values
    n_samples = X.shape[0]
    batch_size = int(np.ceil(n_samples / n_partitions))
    _, idx, counts = np.unique(y, return_index=True, return_counts=True)
    if batch_size < np.min(counts):
        X_part = np.split(X, idx[1:])
        y_part = np.split(y, idx[1:])

        for i in np.arange(1, len(y_part), 2):
            # Determine indices to swap
            swap_size = np.min([X_part[i-1].shape[0] // 2, X_part[i].shape[0] // 2])
            p_im1 = np.random.choice(X_part[i-1].shape[0], swap_size)
            p_i = np.random.choice(X_part[i].shape[0], swap_size)
            # Swap X data
            tmp = X_part[i-1][p_im1].copy()
            X_part[i-1][p_im1] = X_part[i][p_i].copy()
            X_part[i][p_i] = tmp
            # Swap y data
            tmp = y_part[i-1][p_im1].copy()
            y_part[i-1][p_im1] = y_part[i][p_i].copy()
            y_part[i][p_i] = tmp

        X = np.vstack(X_part)
        y = np.hstack(y_part)

    return X, y


def next_batch_unbalanced(X, y, n_partitions, shuffle=True):
    """Imitates federated learning unbalanced partitioning
    """
    # Shuffle data
    if shuffle:
        idxs = np.arange(X.shape[0])
        np.random.shuffle(idxs)
        X = X[idxs]
        y = y[idxs]
    # Pick random partition indices and sort them in ascending order
    partition_idxs = np.sort(
        np.random.randint(0, X.shape[0], size=(n_partitions - 1,))
    )
    # Insert 0 index if not in the array
    if 0 not in partition_idxs:
        partition_idxs = np.insert(partition_idxs, 0, 0)
    if X.shape[0] not in partition_idxs:
        partition_idxs = np.append(partition_idxs, X.shape[0])

    # Iterate through unbalanced partitions
    for i in np.arange(partition_idxs.shape[0]-1):
        start, end = partition_idxs[i], partition_idxs[i + 1]
        X_batch = X[start:end]
        y_batch = y[start:end]
        yield X_batch, y_batch


def update_xy(X, y, idxs):
    """Update x and y given new indices
    """
    X_new = X[idxs]
    y_new = y[idxs]
    n_samples = X_new.shape[0]
    return X_new, y_new, n_samples

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

def do_split(X, y, test_size=0.3):
    """Returns a dictionary of the train test split numpy arrays
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    data = {'training' : {'features' : X_train, 'labels' : y_train},
            'testing' : {'features' : X_test, 'labels' : y_test}}

    return data
