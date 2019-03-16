import struct
import gzip
import numpy as np

# Local
from .base_model import BaseData

class MNist(BaseData):

    def __init__(self, filepath):
        super().__init__(filepath)

        self.prepare_data()

    def read_data(self, filepath):

        with gzip.open(filepath) as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

    def prepare_data(self):
        # Read in train/test data
        self.X_train = self.read_data(self.filepath["train"]["images"])
        self.y_train = self.read_data(self.filepath["train"]["labels"])
        self.X_test = self.read_data(self.filepath["test"]["images"])
        self.y_test = self.read_data(self.filepath["test"]["labels"])

        # One hot encode labels
        self.y_train = MNist._one_hot(self.y_train)
        self.y_test = MNist._one_hot(self.y_test)
