"""Utility functions specifically for distributed modelling environment
"""

from __future__ import absolute_import, print_function

import logging

import numpy as np

from fault_tolerant_ml.utils import zhelpers

logger = logging.getLogger("ftml.distribute.utils")

def decode_msg(worker_params):
    """Decodes a single zmq message containing W and b

    Args:
        worker_params (list of byte strings): W and b params received from corresponding worker

    Returns:
        W (np.ndarray): Parameter matrix as numpy array
        b (np.ndarray): Bias matrix
    """
    # Get data for correponding layer
    Wdata, Wdtype, Wshape, bdata, bdtype, bshape = worker_params

    W = zhelpers.reconstruct_array(Wdata, Wdtype, Wshape)
    b = zhelpers.reconstruct_array(bdata, bdtype, bshape)

    return [W, b]

def decode_params(n_layers, parameters, worker_params, n_items=6):
    """Collect all parameters across workers to master and decode

    Args:
        model (ftml.Model): Fault tolerant model
        parameters (np.ndarray): Tensor where we store the collected messages
        worker_params (byte string): Multipart message received from worker
        n_items (int): Length of message received from worker that we need
        to decode (default: 6). 3 messages for the W tensor 
        (data, dtype, shape) and 3 for the bias vector (data, dtype, shape)

    Returns:
        parameters (np.ndarray): Populated parameter tensor
    """

    # Decode multipart message for each layer
    for j, k in zip(np.arange(n_layers), np.arange(0, len(worker_params), n_items)):
        
        W, b = decode_msg(worker_params[k:k+n_items])
        
        parameters[j][0] = W
        parameters[j][1] = b

    return parameters
