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

logging.basicConfig(level="INFO")
logger = logging.getLogger()

# pylint: disable=no-member

class WorkerV2():
    """Client class
    """
    def __init__(self, worker_id):
        self.io_loop = None
        self.sub = None
        self.push = None
        self.ctrl = None
        self.loop = None
        self.worker_id = worker_id

        # Model variables
        self.X = None
        self.y = None
        self.n_samples: int = None
        self.n_features: int = None
        self.n_classes: int = None
        self.state = None

        logger.info("Setting up...")

    def _connect(self):
        """Connect to sockets
        """
        logger.info("Connecting sockets...")
        self.loop = ioloop.IOLoop()
        context = zmq.Context()

        dev = devices.ThreadDevice(zmq.FORWARDER, zmq.SUB, zmq.DEALER)
        dev.setsockopt_in(zmq.SUBSCRIBE, b"")
        dev.setsockopt_out(zmq.IDENTITY, identity.encode())
        dev.connect_in('tcp://127.0.0.1:5564')
        dev.connect_out('tcp://127.0.0.1:5565')
        dev.start()

        subscriber = context.socket(zmq.SUB) # pylint: disable=no-member
        subscriber.connect("tcp://localhost:5563")
        # subscriber.connect(f"tcp://{master_ip_address}:5563")
        subscriber.setsockopt(zmq.SUBSCRIBE, b"") # pylint: disable=no-member

        self.push = context.socket(zmq.PUSH) # pylint: disable=no-member
        self.push.connect("tcp://localhost:5562")
        # push_socket.connect(f"tcp://{master_ip_address}:5562")

        ctrl_socket = context.socket(zmq.DEALER) # pylint: disable=no-member
        ctrl_socket.setsockopt_string(zmq.IDENTITY, identity) # pylint: disable=no-member
        ctrl_socket.connect("tcp://localhost:5566")
        # ctrl_socket.connect(f"tcp://{master_ip_address}:5565")

        self.sub = zmqstream.ZMQStream(subscriber, self.loop)
        self.ctrl = zmqstream.ZMQStream(ctrl_socket, self.loop)

        self.ctrl.on_recv(self.recv_work)
        # wait for connections
        time.sleep(1)

        self.loop.start()

    def start(self):
        """Start session
        """
        self._connect()

    def kill(self):
        """Kills sockets
        """
        self.sub.close()
        self.ctrl.close()
        self.loop.stop()

    def recv_work(self, msg):
        """Receive work
        """
        logger.info("Receiving work...")
        cmd = msg[0]
        if cmd == b"WORK_DATA":
            # buf = memoryview(msg[1])
            # arr = np.frombuffer(buf, dtype=np.float)
            # logger.info(f"Data.shape={arr.shape}")
            X, y, n_samples, state = parse_setup_from_string(msg[1])

            self.n_samples = n_samples
            self.n_features = X.shape[1]
            self.n_classes = y.shape[1]
            self.state = state
            self.X = X
            self.y = y

            logger.info(f"X.shape={self.X.shape}, y.shape={self.y.shape}")
            # self.ctrl.stop_on_recv()
            # After receiving data we can recv params
            self.sub.on_recv(self.recv_params)

    def recv_params(self, msg):
        """Recv params
        """
        _ = msg[0] # Topic
        cmd = msg[1]
        if cmd == b"WORK_PARAMS":
            logger.info("Receiving params...")
            buf = memoryview(msg[2])
            arr = np.frombuffer(buf, dtype=np.float).copy()
            logger.info(f"Params.shape={arr.shape}")

            # Do some work
            arr = arr * 2
            A = np.random.random((2**11, 2**11))
            tic = time.time()
            np.dot(A, A.transpose())
            logger.info("blocked for %.3f s", (time.time() - tic))

            self.push.send_multipart(
                [b"WORK", self.worker_id.encode(), arr.tostring()]
            )

        if cmd == b"EXIT":
            logger.info("Ending session")
            self.kill()


if __name__ == "__main__":

    time.sleep(1)
    identity = str(uuid.uuid4())

    try:

        worker = WorkerV2(identity)
        worker.start()

    except KeyboardInterrupt:
        logger.info("Keyboard quit")
    except zmq.ZMQError:
        logger.info("ZMQError")
    finally:
        worker.kill()
