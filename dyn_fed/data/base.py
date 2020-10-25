"""Base class for all dfl Datasets
"""
from __future__ import absolute_import, print_function
import gzip
import struct

import numpy as np

def _unarchive(fname):
    """Unarchive filename
    """
    with gzip.open(fname) as f:
        _, _, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
class Dataset():
    """Base class for all datasets for dyn_fed
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

    def prepare_data(self):
        """Overridden function to prepare data
        """
        raise NotImplementedError("Child required to implement prepare data method")
