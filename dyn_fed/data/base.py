"""Base class for all dfl Datasets
"""
from __future__ import absolute_import, print_function



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
