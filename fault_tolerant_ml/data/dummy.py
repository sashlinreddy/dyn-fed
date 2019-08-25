"""Dummy model
"""
from __future__ import print_function, absolute_import
import numpy as np

from .base import Dataset


class DummyData(Dataset):
    """Dummy dataset

    Attributes:
        filepath (str): No filepath needed
        n_samples (int): No. of desired samples
        n_features (int): No. of desired features
        n_classes (int): No. of desired classes
        lower_bound (int): Lower bound
        upper_bound (int): Upper bound
    """
    def __init__(self, filepath="", n_samples=100, n_features=10,
                 n_classes=1, lower_bound=None, upper_bound=None):
        super().__init__(filepath)
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def to_string(self):
        """Returns byte string of array
        """
        return self.X.tostring()

    def __repr__(self):
        return f'<DummyData X={self.X.shape}, y={self.y.shape}>'

    def prepare_data(self):
        """No need to prepare this dataset
        """

    def transform(self):
        """Transforms datset
        """
        # X = np.random.randint(self.lower_bound, self.upper_bound,
        # size=(self.n_samples,), dtype=np.int32)
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randn(self.n_samples, self.n_classes)

        return self.X, self.y
