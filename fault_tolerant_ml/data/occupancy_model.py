from .base_model import BaseData
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import datasets

from fault_tolerant_ml.ml import preprocessor as pre
class OccupancyData(BaseData):

    def __init__(self, filepath, n_stacks=1):
        super().__init__(filepath)
        self.n_stacks = n_stacks
        self.target_names = {'occupied'   :   1,
                        'unoccupied' :   0}

        self.feature_names = ["Temperature","Humidity","Light","CO2","HumidityRatio"]

        self.raw_data = self.read_data()
        self.X, self.y = self.prepare_data()

    def __repr__(self):
        return f'<{self.__class__.__name__} X={self.X.shape}, y={self.y.shape}>'

    def read_data(self):
        return pd.read_csv(self.filepath, delimiter=',')

    def prepare_data(self):

        X = self.raw_data[["Temperature","Humidity","Light","CO2","HumidityRatio"]].values
        y = self.raw_data[["Occupancy"]].values

        sm = SMOTE(random_state=42)

        X_res, y_res = sm.fit_sample(X, y.flatten())
        y_res = y_res[:, np.newaxis]

        X_res = np.tile(X_res, (self.n_stacks, 1))
        y_res = np.tile(y_res, (self.n_stacks, 1))

        return X_res, y_res

    def transform(self):

        # Train test split
        data = self.do_split(self.X, self.y)
        # Normalize data
        norm_params = pre.normalize_datasets(data)

        self.X_train = data["training"]["features"]
        self.y_train = data["training"]["labels"]
        self.X_test = data["testing"]["features"]
        self.y_test = data["testing"]["labels"]

        # Initialize the parameter matrix
        self.n_samples, self.n_features = self.X_train.shape
        _, self.n_classes = self.y_train.shape