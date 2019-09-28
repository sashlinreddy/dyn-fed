"""Master with heartbeater
"""
import logging
import signal
import time
import os

import gevent
import numpy as np
import zmq.green as zmq

from fault_tolerant_ml.data import MNist
from fault_tolerant_ml.data.utils import next_batch
from fault_tolerant_ml.distribute.heartbeater import Heartbeater
from fault_tolerant_ml.distribute.states import (COMPLETE, MAP, MAP_PARAMS,
                                                 START)
from fault_tolerant_ml.proto.utils import setup_to_string

logging.basicConfig(level="DEBUG")
logger = logging.getLogger('ftml')

# pylint: disable=no-member

class MasterV2():
    """Master class
    """
    def __init__(self, n_workers, period=1000):
        # Sockets
        self.heart_pub_socket = None
        self.heart_ctrl_socket = None
        self.pub_socket = None
        self.ctrl_socket = None
        self.pull_socket = None

        # Polling
        self.poller = None
        self.period = period
        self.state = START

        # Distribute variables
        self.model = None
        # self.strategy = self.model.strategy
        self.n_workers = n_workers

        # Model variables
        self.n_iterations = 5
        self.iteration = 0
        self.X = None
        self.y = None

        self.heartbeater = Heartbeater(self.n_workers, period)

    def _connect(self):
        """Connect sockets
        """
        context = zmq.Context()

        # Heart sockets
        self.heart_pub_socket = context.socket(zmq.PUB)
        self.heart_pub_socket.bind("tcp://*:5564")

        self.heart_ctrl_socket = context.socket(zmq.ROUTER)
        self.heart_ctrl_socket.setsockopt_string(zmq.IDENTITY, 'MASTER')
        self.heart_ctrl_socket.bind("tcp://*:5565")

        # Normal sockets
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:5563")

        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind("tcp://*:5562")

        self.ctrl_socket = context.socket(zmq.ROUTER)
        self.ctrl_socket.setsockopt_string(zmq.IDENTITY, 'MASTER')
        self.ctrl_socket.bind("tcp://*:5566")

        self.poller = self._setup_poller()

    def _setup_poller(self):
        """Setup poller
        """
        poller = zmq.Poller()
        poller.register(self.pull_socket, zmq.POLLIN | zmq.POLLERR)
        poller.register(self.ctrl_socket, zmq.POLLIN | zmq.POLLERR)
        poller.register(self.pub_socket, zmq.POLLOUT | zmq.POLLERR)

        poller.register(self.heart_pub_socket, zmq.POLLOUT | zmq.POLLERR)
        poller.register(self.heart_ctrl_socket, zmq.POLLIN | zmq.POLLERR)

        return poller

    def setup(self, X, y):
        """Setup master with data
        """
        self.X = X
        self.y = y

    def heart_loop(self):
        """Heart loop
        """
        logger.info("Starting heart beater")
        while self.state != COMPLETE:
            # Send beat
            self.state = self.heartbeater.beat(self.heart_pub_socket, self.state)
            # Receive responses
            gevent.sleep(1)
            events = dict(self.poller.poll())
            while (self.heart_ctrl_socket in events) and \
                (events.get(self.heart_ctrl_socket) == zmq.POLLIN):
                events = dict(self.poller.poll())
                if (self.heart_ctrl_socket in events) and \
                    (events.get(self.heart_ctrl_socket) == zmq.POLLIN):
                    # Handle pong
                    msg = self.heart_ctrl_socket.recv_multipart()
                    self.heartbeater.handle_pong(msg)

    def start(self):
        """Start server
        """
        self._connect()

        gevent.signal(signal.SIGQUIT, gevent.kill)

        heart_loop = gevent.spawn(self.heart_loop)
        server_loop = gevent.spawn(self.train_loop)
        
        gevent.joinall([
            heart_loop,
            server_loop
        ])

    def _map(self):
        """Map data to workers
        """
        if self.state == MAP:
            logger.info("Sending work to workers")
            # First map data
            n_samples = self.X.shape[0]

            logger.debug(f"State={self.state}")

            batch_size = int(np.ceil(n_samples / len(self.heartbeater.hearts)))
            batch_gen = next_batch(
                self.X,
                self.y,
                batch_size,
                shuffle=False,
                overlap=0.0
            )

            for heart in self.heartbeater.hearts:
                X_batch, y_batch = next(batch_gen)
                X_batch = X_batch.data
                y_batch = y_batch.data
                logger.debug(f"X.shape={X_batch.shape}, y.shape={y_batch.shape}")

                msg = [setup_to_string(X_batch, y_batch, n_samples, self.state)]
                multipart = [heart, b"WORK_DATA"]
                multipart.extend(msg)
                self.ctrl_socket.send_multipart(multipart)

            # A = np.random.randn(10, 1)
            # logger.info(f"Hearts={self.heartbeater.hearts}")
            # for heart in self.heartbeater.hearts:
            #     logger.info(f"Work to {heart}")
            #     msg = A.tostring()
            #     self.ctrl_socket.send_multipart([heart, b"WORK_DATA", msg])

            self.state = MAP_PARAMS
        if self.state == MAP_PARAMS:
            # Map params
            B = np.random.randn(5, 1)
            msg = B.tostring()
            logger.info("Sending params")
            self.pub_socket.send_multipart([b"", b"WORK_PARAMS", msg])

    def _reduce(self):
        """Reduce params from workers
        """
        if self.state == MAP_PARAMS:
            logger.info("Recv work")
            # Recv work
            i = 0
            hearts = len(self.heartbeater.hearts)
            while i < hearts:
                gevent.sleep(0.000001)
                events = dict(self.poller.poll())
                if (self.pull_socket in events) and \
                    (events.get(self.pull_socket) == zmq.POLLIN):
                    msg = self.pull_socket.recv_multipart()
                    cmd = msg[0]
                    worker = msg[1]
                    content = msg[2]
                    if cmd == b"WORK":
                        buf = memoryview(content)
                        arr = np.frombuffer(buf, dtype=np.float).copy()
                        logger.info(
                            f"Received work from {worker}, content.shape={arr.shape}"
                        )

                    i += 1

                if hearts > len(self.heartbeater.hearts):
                    logger.info(
                        f"Changed no of hearts from {hearts} to {self.heartbeater.hearts}"
                        )
                    hearts = len(self.heartbeater.hearts)

            logger.info(f"Iteration={self.iteration}")

            self.iteration += 1

    def train_loop(self):
        """Machine learning training loop
        """
        while self.iteration < self.n_iterations:
            # Need to have a small sleep to enable gevent threading
            gevent.sleep(0.00000001)
            
            # Send data or params
            self._map()

            # Aggregate params
            self._reduce()

        self.done()

    def done(self):
        """Sends exit signal to workers
        """
        time.sleep(1)
        self.pub_socket.send_multipart([b"", b"EXIT"])
        self.state = COMPLETE

    def kill(self):
        """Kills sockets
        """
        self.poller.unregister(self.pull_socket)
        self.poller.unregister(self.pub_socket)
        self.poller.unregister(self.ctrl_socket)
        self.poller.unregister(self.heart_pub_socket)
        self.poller.unregister(self.heart_ctrl_socket)

        self.pull_socket.close()
        self.pub_socket.close()
        self.ctrl_socket.close()
        self.heart_ctrl_socket.close()
        self.heart_pub_socket.close()

if __name__ == "__main__":

    try:
        data_dir = "data/"
        filepaths = {
            "train": {
                "images": os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
                "labels": os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
            },
            "test": {
                "images": os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
                "labels": os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
            }
        }
        data = MNist(filepaths)

        master = MasterV2(n_workers=4)
        master.setup(data.X_train, data.y_train)
        master.start()

    except KeyboardInterrupt:
        logger.info("Keyboard quit")
    except zmq.ZMQError:
        logger.info("ZMQError")
    finally:
        master.kill()
