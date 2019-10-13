"""Downloads year msd dataset from
https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD#
"""
import os
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    """Load Year Million Song dataset
    """
    # dirname = "year-msd"
    # origin = (
    #     "https://archive.ics.uci.edu/ml/machine-learning-databases"
    #     "/00203/YearPredictionMSD.txt.zip"
    # )
    # path = get_file(dirname, origin=origin, extract=True)
    home = os.path.expanduser('~')
    home = Path(home)
    f_path = home/'.ftml/datasets/year-msd/YearPredictionMSD.txt'

    df = pd.read_csv(f_path, header=None)

    train = df.iloc[:463_715]
    test = df.iloc[463_715:]

    x_train, y_train = train.iloc[:, 1:].values, train.iloc[:, [0]].values
    x_test, y_test = test.iloc[:, 1:].values, test.iloc[:, [0]].values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (x_train, y_train), (x_test, y_test)
