# encoding: utf-8
"""Helper module for example applications. Mimics ZeroMQ Guide's zhelpers.h.
"""
from __future__ import print_function

import binascii
import os

import numpy as np
import zmq


def socket_set_hwm(socket, hwm=-1):
    """libzmq 2/3/4 compatible sethwm"""
    try:
        socket.sndhwm = socket.rcvhwm = hwm
    except AttributeError:
        socket.hwm = hwm


def dump(msg_or_socket):
    """Receives all message parts from socket, printing each frame neatly"""
    if isinstance(msg_or_socket, zmq.Socket):
        # it's a socket, call on current message
        msg = msg_or_socket.recv_multipart()
    else:
        msg = msg_or_socket
    print("----------------------------------------")
    for part in msg:
        print("[%03d]" % len(part), end=' ')
        # is_text = True
        try:
            print(part.decode('ascii'))
        except UnicodeDecodeError:
            print(r"0x%s" % (binascii.hexlify(part).decode('ascii')))


def set_id(zsocket):
    """Set simple random printable identity on socket"""
    identity = u"%04x-%04x" % (np.random.randint(0, 0x10000), np.random.randint(0, 0x10000))
    zsocket.setsockopt_string(zmq.IDENTITY, identity) # pylint: disable=no-member


def zpipe(ctx):
    """build inproc pipe for talking to threads

    mimic pipe used in czmq zthread_fork.

    Returns a pair of PAIRs connected via inproc
    """
    a = ctx.socket(zmq.PAIR) # pylint: disable=no-member
    b = ctx.socket(zmq.PAIR) # pylint: disable=no-member
    a.linger = b.linger = 0
    a.hwm = b.hwm = 1
    iface = "inproc://%s" % binascii.hexlify(os.urandom(8))
    a.bind(iface)
    b.connect(iface)
    return a, b

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(A.dtype),
        shape=A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

def multipart_params(data_list):
    """Prep model parameters to send to workers.

    Get parameters from each layer and append the data, dtype and shape a list
    to send. Also quantize parameters if quantize flag = 1
    """
    multipart = []
    for data in data_list:
        multipart.append(data.tostring())
        multipart.append(data.dtype.str.encode())
        multipart.append(str(data.shape).encode())

    return multipart

def reconstruct_array(data, dtype, shape):
    """Reconstruct numpy array from buffer given dtype and shape

    Args:
        data (byte string): Byte array to be reconstructed
        dtype (byte string): Data type of reconstructed array
        shape (byte string): Shape of reconstructed array

    Returns:
        arr (numpy.ndarray): Reconstructed array of shape `shape` and type `dtype`
    """
    # Decode shape byte string
    shape = shape.decode()
    shape = eval(shape) # pylint: disable=eval-used
    # Reconstruct numpy array
    buf = memoryview(data)
    arr = np.frombuffer(buf, dtype=dtype)
    arr = arr.reshape(shape)

    return arr.copy()
