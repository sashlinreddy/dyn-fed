"""Mnist dataset
"""
import gzip
import struct

import numpy as np

from fault_tolerant_ml.operators import Tensor
from fault_tolerant_ml.data.utils import one_hot

from .base import Dataset


class MNist(Dataset):
    """Class to make mnist dataset accessible and overrides Dataset

    Attributes:
        filepath (str): Path to mnist dataset
        class_names (list): List of classname
    """
    def __init__(self, filepath, fashion=False, noniid=True):
        super().__init__(filepath)

        self._noniid = noniid
        self.class_names = [
            "Zero", "One", "Two", "Three", "Four", "Five",
            "Six", "Seven", "Eight", "Nine"
        ]

        if fashion:
            self.class_names = [
                "T-shirt/top", "Trouser", "Pullover", "Dress", 
                "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
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

    def read_data(self, filepath):
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
        self.X_train = self.read_data(self.filepath["train"]["images"])
        self.y_train = self.read_data(self.filepath["train"]["labels"])
        self.X_test = self.read_data(self.filepath["test"]["images"])
        self.y_test = self.read_data(self.filepath["test"]["labels"])

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
