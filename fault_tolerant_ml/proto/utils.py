"""Protocol buffer python utilities
"""
from __future__ import print_function

import numpy as np

from fault_tolerant_ml.proto import ftml_pb2


def reconstruct_numpy(data, dtype, shape):
    """Reconstruct numpy array from buffer given dtype and shape

    Args:
        data (byte string): Byte array to be reconstructed
        dtype (byte string): Data type of reconstructed array
        shape (byte string): Shape of reconstructed array

    Returns:
        arr (numpy.ndarray): Reconstructed array of shape `shape` and type `dtype`
    """
    # Reconstruct numpy array
    buf = memoryview(data)
    arr = np.frombuffer(buf, dtype=dtype)
    arr = arr.reshape(shape)

    return arr.copy()

def reconstruct_setup(msg):
    """Reconstruct numpy array from buffer given dtype and shape

    Args:
        data (byte string): Byte array to be reconstructed
        dtype (byte string): Data type of reconstructed array
        shape (byte string): Shape of reconstructed array

    Returns:
        arr (numpy.ndarray): Reconstructed array of shape `shape` and type `dtype`
    """
    setup = ftml_pb2.Setup()
    setup.ParseFromString(msg)

    # pylint: disable=no-member
    X = reconstruct_numpy(
        setup.X.data,
        setup.X.dtype,
        (setup.X.rows, setup.X.columns)
    )

    y = reconstruct_numpy(
        setup.y.data,
        setup.y.dtype,
        (setup.y.rows, setup.y.columns)
    )
    
    return X, y, setup.n_samples, setup.state
