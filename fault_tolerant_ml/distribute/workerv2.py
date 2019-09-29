"""Client example using zmqstreams and heartbeating. To be used with server.py
"""
import logging
import time
import uuid

import numpy as np
import zmq
from zmq import devices
from zmq.eventloop import zmqstream
from tornado import ioloop

from fault_tolerant_ml.proto.utils import parse_setup_from_string

# pylint: disable=no-member
class WorkerV2():
    """Client class
    """
    def __init__(self, model, identity=None, verbose="INFO"):
        self.io_loop = None
        self.context = None
        self.sub = None
        self.push = None
        self.ctrl = None
        self.loop = None
        self.identity = \
            str(uuid.uuid4()) if identity is None else f"worker-{identity}"

        self.model = model

        # Model variables
        self.X = None
        self.y = None
        self.n_samples: int = None
        self.n_features: int = None
        self.n_classes: int = None
        self.state = None

        self._logger = logging.getLogger(f"ftml.distribute.{self.__class__.__name__}")

        self._logger.info("Setting up...")

    def _connect(self):
        """Connect to sockets
        """
        self._logger.info("Connecting sockets...")
        self.loop = ioloop.IOLoop()
        self.context = zmq.Context()

        dev = devices.ThreadDevice(zmq.FORWARDER, zmq.SUB, zmq.DEALER)
        dev.setsockopt_in(zmq.SUBSCRIBE, b"")
        dev.setsockopt_out(zmq.IDENTITY, self.identity.encode())
        dev.connect_in('tcp://127.0.0.1:5564')
        dev.connect_out('tcp://127.0.0.1:5561')
        dev.start()

        subscriber = self.context.socket(zmq.SUB) # pylint: disable=no-member
        subscriber.connect("tcp://localhost:5560")
        # subscriber.connect(f"tcp://{master_ip_address}:5563")
        subscriber.setsockopt(zmq.SUBSCRIBE, b"") # pylint: disable=no-member

        self.push = self.context.socket(zmq.PUSH) # pylint: disable=no-member
        self.push.connect("tcp://localhost:5567")
        # push_socket.connect(f"tcp://{master_ip_address}:5562")

        ctrl_socket = self.context.socket(zmq.DEALER) # pylint: disable=no-member
        ctrl_socket.setsockopt_string(zmq.IDENTITY, self.identity) # pylint: disable=no-member
        ctrl_socket.connect("tcp://localhost:5566")
        # ctrl_socket.connect(f"tcp://{master_ip_address}:5565")

        self.sub = zmqstream.ZMQStream(subscriber, self.loop)
        self.ctrl = zmqstream.ZMQStream(ctrl_socket, self.loop)

        self.ctrl.on_recv(self.recv_work)
        # wait for connections
        time.sleep(1)

    def start(self):
        """Start session
        """
        try:
            self._connect()
            self.loop.start()
        except KeyboardInterrupt:
            self._logger.info("Keyboard quit")
        except zmq.ZMQError:
            self._logger.info("ZMQError")
        finally:
            self.kill()

    def recv_work(self, msg):
        """Receive work
        """
        self._logger.info("Receiving work...")
        cmd = msg[0]
        if cmd == b"WORK_DATA":
            # buf = memoryview(msg[1])
            # arr = np.frombuffer(buf, dtype=np.float)
            # self._logger.info(f"Data.shape={arr.shape}")
            X, y, n_samples, state = parse_setup_from_string(msg[1])

            self.n_samples = n_samples
            self.n_features = X.shape[1]
            self.n_classes = y.shape[1]
            self.state = state
            self.X = X
            self.y = y

            self._logger.info(f"X.shape={self.X.shape}, y.shape={self.y.shape}")
            # self.ctrl.stop_on_recv()
            # After receiving data we can recv params
            self.sub.on_recv(self.recv_params)

    def recv_params(self, msg):
        """Recv params
        """
        _ = msg[0] # Topic
        cmd = msg[1]
        if cmd == b"WORK_PARAMS":
            self._logger.info("Receiving params...")
            buf = memoryview(msg[2])
            arr = np.frombuffer(buf, dtype=np.float).copy()
            self._logger.info(f"Params.shape={arr.shape}")

            # Do some work
            arr = arr * 2
            A = np.random.random((2**11, 2**11))
            tic = time.time()
            np.dot(A, A.transpose())
            self._logger.info("blocked for %.3f s", (time.time() - tic))

            self.push.send_multipart(
                [b"WORK", self.identity.encode(), arr.tostring()]
            )

        if cmd == b"EXIT":
            self._logger.info("Ending session")
            self.kill()

    def kill(self):
        """Kills sockets
        """
        self._logger.info("Cleaning up")
        self.loop.stop()
