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

from fault_tolerant_ml.operators import Tensor
from fault_tolerant_ml.proto.utils import (parse_params_from_string,
                                           parse_setup_from_string,
                                           params_response_to_string)


# pylint: disable=no-member
class WorkerV2():
    """Client class
    """
    def __init__(self, model, identity=None):
        self.context = None
        self.sub = None
        self.push = None
        self.ctrl = None
        self.loop = None
        self.identity = \
            str(uuid.uuid4()) if identity is None else f"worker-{identity}"

        self.model = model
        self.strategy = self.model.strategy

        # Model variables
        self.X = None
        self.y = None
        self.n_samples: int = None
        self.n_features: int = None
        self.n_classes: int = None
        self.state = None

        # Executor params
        self.remap = self.model.strategy.remap
        self.quantize = self.model.strategy.quantize
        self.comm_period = self.model.strategy.comm_period
        self.send_gradients = self.model.strategy.send_gradients

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

    def _do_work(self, X, y, W_g=None):
        """Worker doing the heavy lifting of calculating gradients and
        calculating loss

        Args:
            W_g (numpy.ndarray): Global parameters

        Returns:
            batch_loss (float): Loss for this iteration
            most_representative (numpy.ndarray): Vector of most representative
            data samples for a particular iteration. Most representative is 
            determined by the data points that have the highest loss.
        """
        # Get predictions
        y_pred = self.model.forward(X)

        batch_loss = self.model.optimizer.minimize(
            self.model,
            y, 
            y_pred, 
            iteration=self.model.iter + 1,
            N=X.shape[0],
            W_g=None)
        most_representative = self.model.optimizer.most_rep
        
        return batch_loss, most_representative

    def _training_loop(self):
        """Perform training loop

        Returns:
            epoch_loss (float): Loss for corresponding epoch
            most_representative(np.ndarray): Most representative data points
        """
        for _ in range(self.comm_period):
            epoch_loss = 0.0
            n_batches = 0
            # self._logger.debug(
            #     f"layers[0].W.data before mini={self.model.layers[0].W.data}"
            # )
            for start in range(0, self.X.shape[0], self.model.batch_size):
                end = start + self.model.batch_size

                X_batch = self.X[start:end]
                y_batch = self.y[start:end]
                # Each worker does work and we get the resulting parameters
                batch_loss, most_representative = \
                self._do_work(
                    X_batch,
                    y_batch,
                    W_g=None
                )
                epoch_loss += batch_loss.data
                n_batches += 1
            epoch_loss /= n_batches
            self._logger.info(
                f"iteration = {self.model.iter}, Loss = {epoch_loss:7.4f}"
            )

            self.model.iter += 1

        return epoch_loss, most_representative

    def start(self):
        """Start session
        """
        try:
            self._connect()
            self.loop.start()
        except KeyboardInterrupt:
            self._logger.info("Keyboard quit")
            self.kill()
        except zmq.ZMQError:
            self._logger.info("ZMQError")
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
            self.X = Tensor(self.X)
            self.y = Tensor(self.y)

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
            parameters = parse_params_from_string(msg[2])
            # buf = memoryview(msg[2])
            # arr = np.frombuffer(buf, dtype=np.float).copy()
            for i in np.arange(self.model.n_layers):
                self.model.layers[i].W.data = parameters[i][0]
                self.model.layers[i].b.data = parameters[i][1]

            self._logger.info(
                f"W[0].shape.shape={self.model.layers[0].W.data.shape}"
            )

            # Do some work
            tic = time.time()
            epoch_loss, most_representative = self._training_loop()
            self._logger.info(f"Batch_loss={epoch_loss}")
            self._logger.info("blocked for %.3f s", (time.time() - tic))

            data = [
                params_response_to_string(
                    self.model.layers,
                    most_representative,
                    epoch_loss
                )
            ]

            multipart = [b"WORK", self.identity.encode()]
            multipart.extend(data)

            self.push.send_multipart(multipart)

        if cmd == b"EXIT":
            self._logger.info("Ending session")
            self.kill()

    def kill(self):
        """Kills sockets
        """
        self._logger.info("Cleaning up")
        self.loop.stop()
