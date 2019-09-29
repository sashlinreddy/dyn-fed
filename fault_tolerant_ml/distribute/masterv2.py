"""Master with heartbeater
"""
import logging
import signal
import time

import gevent
import numpy as np
import zmq.green as zmq

from fault_tolerant_ml.data.utils import next_batch
from fault_tolerant_ml.distribute.heartbeater import Heartbeater
from fault_tolerant_ml.distribute.states import (COMPLETE, MAP, MAP_PARAMS,
                                                 START)
from fault_tolerant_ml.proto.utils import (params_to_string,
                                           parse_params_response_from_string,
                                           setup_to_string)

# pylint: disable=no-member

class MasterV2():
    """Master class
    """
    def __init__(self, model, period=1000):
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
        self.model = model
        self.strategy = self.model.strategy
        self.n_workers = self.strategy.n_workers

        # Model variables
        self.n_iterations = 5
        self.iteration = 0
        self.X = None
        self.y = None
        self.X_valid = None
        self.y_valid = None

        self.heartbeater = Heartbeater(self.n_workers, period)

        self._logger = logging.getLogger(f"ftml.distribute.{self.__class__.__name__}")

    def _connect(self):
        """Connect sockets
        """
        context = zmq.Context()

        # Heart sockets
        self.heart_pub_socket = context.socket(zmq.PUB)
        self.heart_pub_socket.bind("tcp://*:5564")

        self.heart_ctrl_socket = context.socket(zmq.ROUTER)
        self.heart_ctrl_socket.setsockopt_string(zmq.IDENTITY, 'MASTER')
        self.heart_ctrl_socket.bind("tcp://*:5561")

        # Normal sockets
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:5560")

        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind("tcp://*:5567")

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

    def setup(self, X, y, X_valid=None, y_valid=None):
        """Setup master with data
        """
        self.X = X
        self.y = y
        self.X_valid = X_valid
        self.y_valid = y_valid

    def heart_loop(self):
        """Heart loop
        """
        self._logger.info("Starting heart beater")
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

        self.kill()

    def _map(self):
        """Map data to workers
        """
        if self.state == MAP:
            self._logger.info("Sending work to workers")
            # First map data
            n_samples = self.X.shape[0]

            self._logger.debug(f"State={self.state}")

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
                self._logger.debug(f"X.shape={X_batch.shape}, y.shape={y_batch.shape}")

                msg = [setup_to_string(X_batch, y_batch, n_samples, self.state)]
                multipart = [heart, b"WORK_DATA"]
                multipart.extend(msg)
                self.ctrl_socket.send_multipart(multipart)

            self.state = MAP_PARAMS
        if self.state == MAP_PARAMS:
            # Map params
            msg = [params_to_string(self.model.layers)]
            multipart = [b"", b"WORK_PARAMS"]
            multipart.extend(msg)
            # B = np.random.randn(5, 1)
            # msg = B.tostring()
            self._logger.info("Sending params")
            self.pub_socket.send_multipart(multipart)

    def _reduce(self):
        """Reduce params from workers
        """
        if self.state == MAP_PARAMS:
            self._logger.info("Recv work")
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
                        parameters, mr, loss = \
                            parse_params_response_from_string(content)
                        # buf = memoryview(content)
                        # arr = np.frombuffer(buf, dtype=np.float).copy()
                        self._logger.info(
                            f"Received work from {worker}, mr.shape={mr.shape}"
                        )

                    i += 1

                if hearts > len(self.heartbeater.hearts):
                    self._logger.info(
                        f"Changed no of hearts from {hearts} to {self.heartbeater.hearts}"
                        )
                    hearts = len(self.heartbeater.hearts)

            self._logger.info(f"Iteration={self.iteration}")

            self.iteration += 1

    def train_loop(self):
        """Machine learning training loop
        """
        try:
            while self.iteration < self.n_iterations:
                # Need to have a small sleep to enable gevent threading
                gevent.sleep(0.00000001)
                
                # Send data or params
                self._map()

                # Aggregate params
                self._reduce()

            self.done()
        except KeyboardInterrupt:
            self._logger.info("Keyboard quit")
        except zmq.ZMQError:
            self._logger.info("ZMQError")
            self.done()
        finally:
            self._logger.info("Exiting peacefully")
            # self.kill()

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
