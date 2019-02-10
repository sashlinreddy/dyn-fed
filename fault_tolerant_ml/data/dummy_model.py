from .base_model import BaseData
import numpy as np

class DummyData(BaseData):

    def __init__(self, filepath, n_samples=100, lower_bound=0, upper_bound=10):
        super().__init__(filepath)
        self.n_samples = n_samples
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.X = self.prepare_data()

    def to_string(self):
        return self.X.tostring()

    def prepare_data(self):

        X = np.random.randint(self.lower_bound, self.upper_bound, size=(self.n_samples,), dtype=np.int32)

        return X