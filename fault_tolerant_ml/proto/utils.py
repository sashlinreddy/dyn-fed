"""Protocol buffer python utilities
"""
from __future__ import print_function

import logging

import numpy as np

from fault_tolerant_ml.proto import ftml_pb2

logger = logging.getLogger('ftml.proto.utils')

def setup_to_string(X, y, n_samples, state):
    """Generate setup data buffer message

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Label matrix
        n_samples (int): Total no. of samples
        state (int): State of learning system (MAP, DIST_PARAMS, REDUCE, etc)

    Returns:
        buffer (byte string): Byte string of objects
    """
    X_proto = ftml_pb2.Tensor(
        data=X.tostring(),
        rows=X.shape[0],
        columns=X.shape[1],
        dtype=X.dtype.str
    )

    y_proto = ftml_pb2.Tensor(
        data=y.tostring(),
        rows=y.shape[0],
        columns=y.shape[1],
        dtype=y.dtype.str
    )

    sent_msg = ftml_pb2.Setup(
        n_samples=n_samples,
        state=state,
        X=X_proto,
        y=y_proto
    )

    buffer = sent_msg.SerializeToString()
    
    return buffer

def setup_reponse_to_string(svd_idx):
    """Generate setup data buffer message

    Args:
        svd_idx (float): Worker SVD 95 percentile index

    Returns:
        buffer (byte string): Byte string of objects
    """
    sent_msg = ftml_pb2.SetupResponse(
        svd_idx=svd_idx
    )

    buffer = sent_msg.SerializeToString()
    
    return buffer

def comms_setup_to_string(n_iterations, comm_interval, comm_every_iter):
    """Generate setup data buffer message

    Args:
        n_iterations (int): Worker comm period

    Returns:
        buffer (byte string): Byte string of objects
    """
    sent_msg = ftml_pb2.CommSetup(
        n_iterations=n_iterations,
        comm_interval=comm_interval,
        comm_every_iter=comm_every_iter
    )

    buffer = sent_msg.SerializeToString()
    
    return buffer

def params_to_string(model_layers):
    """Parameters protobuf serialization

    Args:
        model_layers (list): List of model layers (ftml.Layer)

    Returns:
        msg (byte string): Serialized parameters
    """
    layers = []
    for layer in model_layers:
        W = ftml_pb2.Tensor(
            data=layer.W.data.tostring(),
            rows=layer.W.shape[0],
            columns=layer.W.shape[1],
            dtype=layer.W.dtype.str
        )
        b = ftml_pb2.Tensor(
            data=layer.b.data.tostring(),
            rows=layer.b.shape[0],
            columns=layer.b.shape[1],
            dtype=layer.b.dtype.str
        )
        layers.append(
            ftml_pb2.Parameter(
                W=W,
                b=b
            )
        )

    msg = ftml_pb2.Subscription(
        layers=layers
    )

    buffer = msg.SerializeToString()

    return buffer


def params_response_to_string(model_layers, most_rep, loss):
    """Parameter response protobuf serialization

    Args:
        model_layers (list): List of model layers (ftml.Layer)
        most_rep (np.ndarray): Most representative data points
        loss (float): Loss for corresponding epoch

    Returns:
        msg (byte string): Serialized parameters
    """
    layers = []
    for layer in model_layers:
        W = ftml_pb2.Tensor(
            data=layer.W.data.tostring(),
            rows=layer.W.shape[0],
            columns=layer.W.shape[1],
            dtype=layer.W.dtype.str
        )
        b = ftml_pb2.Tensor(
            data=layer.b.data.tostring(),
            rows=layer.b.shape[0],
            columns=layer.b.shape[1],
            dtype=layer.b.dtype.str
        )
        layers.append(
            ftml_pb2.Parameter(
                W=W,
                b=b
            )
        )

    if most_rep.ndim < 2:
        most_rep = most_rep[:, np.newaxis]

    mr = ftml_pb2.Tensor(
        data=most_rep.tostring(),
        rows=most_rep.shape[0],
        columns=most_rep.shape[1],
        dtype=most_rep.dtype.str
    )

    msg = ftml_pb2.SubscriptionResponse(
        layers=layers,
        most_rep=mr,
        loss=loss
    )

    buffer = msg.SerializeToString()

    return buffer

def parse_numpy_from_string(data, dtype, shape):
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

def parse_setup_from_string(msg):
    """Reconstruct protocol buffer message

    Args:
        msg (byte string): Byte array to be reconstructed

    Returns:
        X (numpy.ndarray): Reconstructed feature matrix
        y (numpy.ndarray): Reconstructed label matrix
        n_samples (int): Reconstructed total no. of samples
        state (int): Reconstructured state of learning system
            (MAP, DIST_PARAMS, REDUCE, etc)
    """
    setup = ftml_pb2.Setup()
    setup.ParseFromString(msg)

    # pylint: disable=no-member
    X = parse_numpy_from_string(
        setup.X.data,
        setup.X.dtype,
        (setup.X.rows, setup.X.columns)
    )

    y = parse_numpy_from_string(
        setup.y.data,
        setup.y.dtype,
        (setup.y.rows, setup.y.columns)
    )
    
    return X, y, setup.n_samples, setup.state

def parse_setup_response_from_string(msg):
    """Reconstruct protocol buffer message

    Args:
        msg (byte string): Byte array to be reconstructed

    Returns:
        svd_idx (float): Worker SVD 95 percentile index
    """
    # pylint: disable=no-member
    setup_response = ftml_pb2.SetupResponse()
    setup_response.ParseFromString(msg)

    
    return setup_response.svd_idx

def parse_comm_setup_from_string(msg):
    """Reconstruct protocol buffer message

    Args:
        msg (byte string): Byte array to be reconstructed

    Returns:
        n_iterations (int): Worker comm iterations
    """
    # pylint: disable=no-member
    comm_setup = ftml_pb2.CommSetup()
    comm_setup.ParseFromString(msg)

    n_iterations = comm_setup.n_iterations
    comm_interval = comm_setup.comm_interval
    comm_every_iter = comm_setup.comm_every_iter
    
    return n_iterations, comm_interval, comm_every_iter

def parse_params_from_string(msg):
    """Parameters protobuf deserialization

    Args:
        msg (byte string): Byte array to be reconstructed

    Returns:
        layers (list): List of parameters received
    """
    subscription = ftml_pb2.Subscription()
    subscription.ParseFromString(msg)

    layers = []

    # pylint: disable=no-member
    for layer in subscription.layers:
        W = parse_numpy_from_string(
            layer.W.data,
            layer.W.dtype,
            (layer.W.rows, layer.W.columns)
        )

        b = parse_numpy_from_string(
            layer.b.data,
            layer.b.dtype,
            (layer.b.rows, layer.b.columns)
        )
        layers.append([W, b])

    return layers

def parse_params_response_from_string(msg):
    """Parameters protobuf deserialization

    Args:
        msg (byte string): Byte array to be reconstructed

    Returns:
        layers (list): List of parameters received
    """
    subscription_response = ftml_pb2.SubscriptionResponse()
    subscription_response.ParseFromString(msg)

    layers = []

    # pylint: disable=no-member
    for layer in subscription_response.layers:
        W = parse_numpy_from_string(
            layer.W.data,
            layer.W.dtype,
            (layer.W.rows, layer.W.columns)
        )

        b = parse_numpy_from_string(
            layer.b.data,
            layer.b.dtype,
            (layer.b.rows, layer.b.columns)
        )
        layers.append([W, b])

    most_rep = parse_numpy_from_string(
        subscription_response.most_rep.data,
        subscription_response.most_rep.dtype,
        (subscription_response.most_rep.rows, subscription_response.most_rep.columns)
    )

    loss = subscription_response.loss

    return layers, most_rep, loss
    