"""Protocol buffer python utilities
"""
from __future__ import print_function

import logging

import numpy as np

from dyn_fed.proto import dfl_pb2

logger = logging.getLogger('dfl.proto.utils')

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
    X_proto = dfl_pb2.Tensor(
        data=X.tostring(),
        shape=X.shape,
        dtype=X.dtype.str,
    )

    y_proto = dfl_pb2.Tensor(
        data=y.tostring(),
        shape=y.shape,
        dtype=y.dtype.str
    )

    sent_msg = dfl_pb2.Setup(
        n_samples=n_samples,
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
    sent_msg = dfl_pb2.SetupResponse(
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
    sent_msg = dfl_pb2.CommSetup(
        n_iterations=n_iterations,
        comm_interval=comm_interval,
        comm_every_iter=comm_every_iter
    )

    buffer = sent_msg.SerializeToString()    
    return buffer

def params_to_string(model_layers):
    """Parameters protobuf serialization

    Args:
        model_layers (list): List of model layers (dfl.Layer)

    Returns:
        msg (byte string): Serialized parameters
    """
    layers = []
    for layer in model_layers:
        logger.debug(f"layer dtype={layer.W.dtype.str}")
        W = dfl_pb2.Tensor(
            data=layer.W.data.tostring(),
            shape=layer.W.shape,
            dtype=layer.W.dtype.str
        )
        b = dfl_pb2.Tensor(
            data=layer.b.data.tostring(),
            shape=layer.b.shape,
            dtype=layer.b.dtype.str
        )
        param = dfl_pb2.Parameter(
            W=W,
            b=b
        )
        layers.append(param)

    msg = dfl_pb2.Subscription(
        layers=layers
    )

    buffer = msg.SerializeToString()

    return buffer

def params_to_stringv2(trainable_vars):
    """Parameters protobuf serialization

    Args:
        model_layers (list): List of model layers (dfl.Layer)

    Returns:
        msg (byte string): Serialized parameters
    """
    weights = []
    for var in trainable_vars:
        # logger.debug(f"layer dtype={layer.W.dtype.str}")
        var_npy = var.numpy()
        weight = dfl_pb2.Tensor(
            data=var_npy.tostring(),
            shape=var_npy.shape,
            dtype=var_npy.dtype.str
        )

        weights.append(weight)

    msg = dfl_pb2.SubscriptionV2(
        trainable_weights=weights
    )

    buffer = msg.SerializeToString()

    return buffer


def params_response_to_string(model_layers, most_rep, loss):
    """Parameter response protobuf serialization

    Args:
        model_layers (list): List of model layers (dfl.Layer)
        most_rep (np.ndarray): Most representative data points
        loss (float): Loss for corresponding epoch

    Returns:
        msg (byte string): Serialized parameters
    """
    layers = []
    for layer in model_layers:
        W = dfl_pb2.Tensor(
            data=layer.W.data.tostring(),
            shape=layer.W.shape,
            dtype=layer.W.dtype.str
        )
        b = dfl_pb2.Tensor(
            data=layer.b.data.tostring(),
            shape=layer.b.shape,
            dtype=layer.b.dtype.str
        )
        layers.append(
            dfl_pb2.Parameter(
                W=W,
                b=b
            )
        )

    if most_rep.ndim < 2:
        most_rep = most_rep[:, np.newaxis]

    mr = dfl_pb2.Tensor(
        data=most_rep.tostring(),
        shape=most_rep.shape,
        dtype=most_rep.dtype.str
    )

    msg = dfl_pb2.SubscriptionResponse(
        layers=layers,
        most_rep=mr,
        loss=loss
    )

    buffer = msg.SerializeToString()

    return buffer

def params_response_to_stringv2(trainable_vars, loss):
    """Parameter response protobuf serialization

    Args:
        model_layers (list): List of model layers (dfl.Layer)
        most_rep (np.ndarray): Most representative data points
        loss (float): Loss for corresponding epoch

    Returns:
        msg (byte string): Serialized parameters
    """
    weights = []
    for var in trainable_vars:
        var_npy = var.numpy()
        weight = dfl_pb2.Tensor(
            data=var_npy.tostring(),
            shape=var_npy.shape,
            dtype=var_npy.dtype.str
        )
        weights.append(weight)


    msg = dfl_pb2.SubscriptionResponseV2(
        trainable_weights=weights,
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
    setup = dfl_pb2.Setup()
    setup.ParseFromString(msg)

    # pylint: disable=no-member
    X = parse_numpy_from_string(
        setup.X.data,
        setup.X.dtype,
        setup.X.shape
    )

    y = parse_numpy_from_string(
        setup.y.data,
        setup.y.dtype,
        setup.y.shape
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
    setup_response = dfl_pb2.SetupResponse()
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
    comm_setup = dfl_pb2.CommSetup()
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
    subscription = dfl_pb2.Subscription()
    subscription.ParseFromString(msg)

    layers = []

    # pylint: disable=no-member
    for layer in subscription.layers:
        W = parse_numpy_from_string(
            layer.W.data,
            layer.W.dtype,
            layer.W.shape
        )

        b = parse_numpy_from_string(
            layer.b.data,
            layer.b.dtype,
            layer.b.shape
        )
        layers.append([W, b])

    return layers

def parse_params_from_stringv2(msg):
    """Parameters protobuf deserialization

    Args:
        msg (byte string): Byte array to be reconstructed

    Returns:
        layers (list): List of parameters received
    """
    subscription = dfl_pb2.SubscriptionV2()
    subscription.ParseFromString(msg)

    weights = []

    # pylint: disable=no-member
    for trainable_weight in subscription.trainable_weights:
        weight = parse_numpy_from_string(
            trainable_weight.data,
            trainable_weight.dtype,
            trainable_weight.shape
        )

        weights.append(weight)

    return weights

def parse_params_response_from_string(msg):
    """Parameters protobuf deserialization

    Args:
        msg (byte string): Byte array to be reconstructed

    Returns:
        layers (list): List of parameters received
    """
    subscription_response = dfl_pb2.SubscriptionResponse()
    subscription_response.ParseFromString(msg)

    layers = []

    # pylint: disable=no-member
    for layer in subscription_response.layers:
        W = parse_numpy_from_string(
            layer.W.data,
            layer.W.dtype,
            layer.W.shape
        )

        b = parse_numpy_from_string(
            layer.b.data,
            layer.b.dtype,
            layer.b.shape
        )
        layers.append([W, b])

    most_rep = parse_numpy_from_string(
        subscription_response.most_rep.data,
        subscription_response.most_rep.dtype,
        subscription_response.most_rep.shape
    )

    loss = subscription_response.loss

    return layers, most_rep, loss

def parse_params_response_from_stringv2(msg):
    """Parameters protobuf deserialization

    Args:
        msg (byte string): Byte array to be reconstructed

    Returns:
        layers (list): List of parameters received
    """
    subscription_response = dfl_pb2.SubscriptionResponseV2()
    subscription_response.ParseFromString(msg)

    weights = []

    # pylint: disable=no-member
    for trainable_weight in subscription_response.trainable_weights:
        weight = parse_numpy_from_string(
            trainable_weight.data,
            trainable_weight.dtype,
            trainable_weight.shape
        )

        
        weights.append(weight)

    loss = subscription_response.loss

    return weights, loss
    