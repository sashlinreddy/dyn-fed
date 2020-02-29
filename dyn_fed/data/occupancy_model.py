"""Occupancy dataset
"""
from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from dyn_fed.preprocessing import preprocessor as pre
from dyn_fed.data.utils import do_split

from .base import Dataset


class OccupancyData(Dataset):
    """Occupancy dataset
    """
    def __init__(self, filepath, n_stacks=1):
        super().__init__(filepath)
        self.n_stacks = n_stacks
        self.target_names = {
            'occupied': 1,
            'unoccupied': 0
        }

        self.feature_names = [
            "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"
        ]
        self.n_samples: int = 0
        self.n_features: int = 0
        self.n_classes: int = 0
        self.raw_data = self.load_data()
        self.X, self.y = self.prepare_data()
        self.transform()

    def __repr__(self):
        return f'<{self.__class__.__name__} X={self.X.shape}, y={self.y.shape}>'

    def load_data(self):
        """Reads dataset from disk
        """
        return pd.read_csv(self.filepath, delimiter=',')

    def prepare_data(self):
        features = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
        X = self.raw_data[features].values
        y = self.raw_data[["Occupancy"]].values

        sm = SMOTE(random_state=42)

        X_res, y_res = sm.fit_sample(X, y.flatten())
        y_res = y_res[:, np.newaxis]

        X_res = np.tile(X_res, (self.n_stacks, 1))
        y_res = np.tile(y_res, (self.n_stacks, 1))

        return X_res, y_res

    def transform(self):
        """Transforms dataset and prepares it
        """
        # Train test split
        data = do_split(self.X, self.y)
        # Normalize data
        _ = pre.normalize_datasets(data)

        self.X_train = data["training"]["features"]
        self.y_train = data["training"]["labels"]
        self.X_test = data["testing"]["features"]
        self.y_test = data["testing"]["labels"]

        # Initialize the parameter matrix
        self.n_samples, self.n_features = self.X_train.shape
        _, self.n_classes = self.y_train.shape
