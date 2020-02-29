"""Mnist dataset
"""
import gzip
import struct
from pathlib import Path
from typing import Optional

import numpy as np

from dyn_fed.operators import Tensor
from dyn_fed.data.utils import one_hot

from .base import Dataset

def _preprocess(X_train, y_train, X_test, y_test, onehot: Optional[bool]=False):
    """Scales data between 0 and 1 and one-hot encodes labels
    """
    # Scale data
    fac = 255 * 0.99 + 0.01
    X_train = X_train / fac
    X_test = X_test / fac

    # One hot encode labels
    if onehot:
        y_train = one_hot(y_train)
        y_test = one_hot(y_test)
        # we don't want zeroes and ones in the labels neither:
        y_train = np.where(y_train == 0, 0.01, 0.99)
        y_test = np.where(y_test == 0, 0.01, 0.99)

    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    # self.y_train[self.y_train==0] = 0.01
    # self.y_train[self.y_train==1] = 0.99
    # self.y_test[self.y_test==0] = 0.01
    # self.y_test[self.y_test==1] = 0.99

    return X_train, y_train, X_test, y_test

def _unarchive(fname):
    """Unarchive filename
    """
    with gzip.open(fname) as f:
        _, _, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def load_data(path: Optional[str]=None,
              noniid: Optional[bool]=False,
              onehot=False,
              reshape: Optional[bool]=False):
    """Load mnist data - returns numpy train and test sets
    """
    # For now just using this path
    if path is None:
        path = Path(__file__).resolve().parents[2]/'data/mnist'

    x_train_fname = path/'train-images-idx3-ubyte.gz'
    y_train_fname = path/'train-labels-idx1-ubyte.gz'
    x_test_fname = path/'t10k-images-idx3-ubyte.gz'
    y_test_fname = path/'t10k-labels-idx1-ubyte.gz'

    X_train = _unarchive(x_train_fname)
    y_train = _unarchive(y_train_fname)
    X_test = _unarchive(x_test_fname)
    y_test = _unarchive(y_test_fname)

    # Should use this path in future
    # os.path.join(os.path.expanduser('~'), '.dfl')

    if reshape:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    X_train, y_train, X_test, y_test = _preprocess(
        X_train,
        y_train,
        X_test,
        y_test,
        onehot=onehot
    )

    if noniid:
        # Undo one hot encoding
        labels = np.argmax(y_train, axis=1)
        # Sort and get indices
        noniid_idxs = np.argsort(labels)
        X_train = X_train[noniid_idxs]
        y_train = y_train[noniid_idxs]

    return X_train, y_train, X_test, y_test
    

class MNist(Dataset):
    """Class to make mnist dataset accessible and overrides Dataset

    Attributes:
        filepath (str): Path to mnist dataset
        noniid (bool): Whether or not to make dataset noniid
    """
    def __init__(self, filepath, noniid=False):
        super().__init__(filepath)

        self._noniid = noniid
        self.class_names = [
            "Zero", "One", "Two", "Three", "Four", "Five",
            "Six", "Seven", "Eight", "Nine"
        ]

        self.prepare_data()
        self.n_samples, self.n_features = self.X_train.shape
        self.n_classes = len(self.class_names)

    
    def __repr__(self):
        return (
            f"<{self.__class__.__name__} X_train={self.X_train.shape}, "
            f"y_train={self.y_train.shape}, X_test={self.X_test.shape}, "
            f"y_test={self.y_test.shape}>"
        )

    def load_data(self, filepath):
        """Reads mnist dataset
        """
        with gzip.open(filepath) as f:
            _, _, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

    def preprocess(self):
        """Scales data between 0 and 1 and one-hot encodes labels
        """
        # Scale data
        fac = 255 * 0.99 + 0.01
        self.X_train = self.X_train / fac
        self.X_test = self.X_test / fac

        # One hot encode labels
        self.y_train = one_hot(self.y_train)
        self.y_test = one_hot(self.y_test)
        # we don't want zeroes and ones in the labels neither:
        self.y_train = np.where(self.y_train == 0, 0.01, 0.99)
        self.y_test = np.where(self.y_test == 0, 0.01, 0.99)

        self.X_train = self.X_train.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)

        self.X_test = self.X_test.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)
        # self.y_train[self.y_train==0] = 0.01
        # self.y_train[self.y_train==1] = 0.99
        # self.y_test[self.y_test==0] = 0.01
        # self.y_test[self.y_test==1] = 0.99
        self.X_train = Tensor(self.X_train)
        self.y_train = Tensor(self.y_train)
        self.X_test = Tensor(self.X_test)
        self.y_test = Tensor(self.y_test)

    def prepare_data(self):
        """Reads in, reshapes, scales and one-hot encodes data
        """
        # Read in train/test data
        self.X_train = self.load_data(self.filepath["train"]["images"])
        self.y_train = self.load_data(self.filepath["train"]["labels"])
        self.X_test = self.load_data(self.filepath["test"]["images"])
        self.y_test = self.load_data(self.filepath["test"]["labels"])

        # Reshape data
        self.X_train = self.X_train.reshape(self.X_train.shape[0], -1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], -1)

        # Preprocess data
        self.preprocess()

        if self._noniid:
            # Undo one hot encoding
            labels = np.argmax(self.y_train.data, axis=1)
            # Sort and get indices
            noniid_idxs = np.argsort(labels)
            self.X_train.data = self.X_train.data[noniid_idxs]
            self.y_train.data = self.y_train.data[noniid_idxs]

class FashionMNist(MNist):
    """Class to make fashion mnist dataset accessible and inherits from MNist

    Attributes:
        filepath (str): Path to mnist dataset
        class_names (list): List of classname
    """
    def __init__(self, filepath, noniid=False):
        super().__init__(filepath, noniid=noniid)
        self.class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", 
            "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
            ]
