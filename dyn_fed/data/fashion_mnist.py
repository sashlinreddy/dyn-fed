"""Load fashion mnist dataset
"""
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from dyn_fed.operators import Tensor
from dyn_fed.data.base import _unarchive
from dyn_fed.data.utils import one_hot

def _preprocess(X_train,
                y_train,
                X_test,
                y_test,
                onehot: Optional[bool]=False,
                test_only: Optional[bool]=False,
                convert_types: Optional[bool]=False):
    """Scales data between 0 and 1 and one-hot encodes labels
    """
    data: Tuple
    # Scale data
    fac = 255 * 0.99 + 0.01
    if not test_only:
        X_train = X_train / fac
        X_test = X_test / fac

        # One hot encode labels
        if onehot:
            y_train = one_hot(y_train)
            y_test = one_hot(y_test)
            # we don't want zeroes and ones in the labels neither:
            if convert_types:
                y_train = np.where(y_train == 0, 0.01, 0.99)
                y_test = np.where(y_test == 0, 0.01, 0.99)
                y_train = y_train.astype(np.float32)
                y_test = y_test.astype(np.float32)
            else:
                y_train = y_train.astype(np.uint8)
                y_test = y_test.astype(np.uint8)

        X_train = X_train.astype(np.float32)    
        # y_train = y_train.astype(np.float32)

        X_test = X_test.astype(np.float32)
        # y_test = y_test.astype(np.float32)
        # if onehot:
        #     y_train = y_train.astype(np.bool)
        #     y_test = y_test.astype(np.bool)
        # self.y_train[self.y_train==0] = 0.01
        # self.y_train[self.y_train==1] = 0.99
        # self.y_test[self.y_test==0] = 0.01
        # self.y_test[self.y_test==1] = 0.99

        data = (X_train, y_train, X_test, y_test)

    else:
        X_test = X_test / fac

        # One hot encode labels
        if onehot:
            y_test = one_hot(y_test)
            # we don't want zeroes and ones in the labels neither:
            # y_test = np.where(y_test == 0, 0.01, 0.99)
            if convert_types:
                y_test = np.where(y_test == 0, 0.01, 0.99)
                y_test = y_test.astype(np.float32)
            else:
                y_test = y_test.astype(np.uint8)

        X_test = X_test.astype(np.float32)
        data = (X_test, y_test)

    return data

def load_data(path: Optional[str]=None,
              noniid: Optional[bool]=False,
              onehot=False,
              convert_types=False,
              reshape: Optional[bool]=False,
              test_only: Optional[bool]=False,
              tensorize: Optional[bool]=False,
              rgb_channel: Optional[bool]=False):
    """Load mnist data - returns numpy train and test sets
    """
    # For now just using this path
    if path is None:
        path = Path(__file__).resolve().parents[2]/'data/fashion-mnist'

    x_train_fname = path/'train-images-idx3-ubyte.gz'
    y_train_fname = path/'train-labels-idx1-ubyte.gz'
    x_test_fname = path/'t10k-images-idx3-ubyte.gz'
    y_test_fname = path/'t10k-labels-idx1-ubyte.gz'

    data: Tuple

    if not test_only:
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
            onehot=onehot,
            convert_types=convert_types
        )

        if noniid:
            # Undo one hot encoding
            labels = y_train
            if onehot:
                labels = np.argmax(y_train, axis=1)
            # Sort and get indices
            noniid_idxs = np.argsort(labels)
            X_train = X_train[noniid_idxs]
            y_train = y_train[noniid_idxs]

        if tensorize:
            X_train = Tensor(X_train)
            y_train = Tensor(y_train)
            X_test = Tensor(X_test)
            y_test = Tensor(y_test)

        if rgb_channel:
            X_train = X_train[..., np.newaxis]
            X_test = X_test[..., np.newaxis]

        data = (X_train, y_train, X_test, y_test)
    else:
        X_test = _unarchive(x_test_fname)
        y_test = _unarchive(y_test_fname)

        # Should use this path in future
        # os.path.join(os.path.expanduser('~'), '.dfl')

        if reshape:
            X_test = X_test.reshape(X_test.shape[0], -1)

        X_test, y_test = _preprocess(
            None,
            None,
            X_test,
            y_test,
            onehot=onehot,
            test_only=test_only,
            convert_types=convert_types
        )

        if tensorize:
            X_test = Tensor(X_test)
            y_test = Tensor(y_test)

        if rgb_channel:
            X_test = X_test[..., np.newaxis]

        data = (X_test, y_test)

    return data
    