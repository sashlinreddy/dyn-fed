"""Contains all worker logic for fault tolerant ml

All worker devices will contain worker logic
"""

import zmq.green as zmq
import time
import numpy as np
import uuid
import logging
import click
import os
# from dotenv import find_dotenv, load_dotenv

# Local
from fault_tolerant_ml.utils import setup_logger
from fault_tolerant_ml.utils import zhelpers
from fault_tolerant_ml.ml import hypotheses, loss_fns
from fault_tolerant_ml.ml.optimizer import ParallelSGDOptimizer
from fault_tolerant_ml.tools import TFLogger

class Worker(object):
    """Worker class for distributed machine learning system
    
    Attributes:
        worker_id (str): Unique identifier for worker
        subscriber (zmq.Socket): zmq.SUB socket which subscribes to all master published messages
        connected (bool): Whether or not the worker is connected successfully to the master
    """
    def __init__(self, verbose):

        self.worker_id = str(uuid.uuid4())
        self.subscriber = None
        self.connected = False

        # Model variables
        self.hypothesis = hypotheses.log_hypothesis
        self.gradient = loss_fns.cross_entropy_gradient

        self._logger = setup_logger(filename=f'log-{self.worker_id}.log', level=verbose)
        self.tf_logger = None

        if "LOGDIR" in os.environ:
            logdir = os.path.join(os.environ["LOGDIR"], f"tf/{self.worker_id}")
            self.tf_logger = TFLogger(logdir)

    def connect(self):
        """Prepare our context, push socket and publisher
        """
        self.context    = zmq.Context()
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect("tcp://localhost:5563")
        self.subscriber.setsockopt(zmq.SUBSCRIBE, b"")

        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.connect("tcp://localhost:5562")

        self.ctrl_socket = self.context.socket(zmq.DEALER)
        self.ctrl_socket.setsockopt_string(zmq.IDENTITY, self.worker_id)
        self.ctrl_socket.connect("tcp://localhost:5565")

    def setup_poller(self):
        """Register necessary sockets for poller
        """
        poller = zmq.Poller()
        poller.register(self.subscriber, zmq.POLLIN | zmq.POLLERR)
        poller.register(self.ctrl_socket, zmq.POLLIN | zmq.POLLERR)

        return poller

    def do_work(self, X, y, theta):
        """Worker doing the heavy lifting of calculating gradients and calculating loss

        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Label vector
            theta (numpy.ndarray): Parameter matrix

        Returns:
            d_theta (numpy.ndarray): Gradient matrix for parameters
            batch_loss (float): Loss for this iteration
            most_representative (numpy.ndarray): Vector of most representative data samples     for a particular iteration. Most representative is determined by the data         points that have the highest loss.
        """

        # Get predictions
        y_pred = self.hypothesis(X, theta)

        d_theta, batch_loss, most_representative = self.optimizer.minimize(X, y, y_pred, theta)
        
        return d_theta, batch_loss, most_representative

    def receive_data(self, start=True):
        """Receives data from worker

        Receives and makes sense of the data received from the master. Also initializes the optimizer since we receive the learning rate from the master at this stage.

        Args:
            start (bool): Whether or not we received the data at the beginning of the workers   life span. 

        Returns:
            n_samples (int): No. of samples in the dataset for a worker
            n_features (int): No. of features in the dataset
            n_classes (int): No. of classes/labels
        """
        data, dtype, shape = self.ctrl_socket.recv_multipart()
        shape = shape.decode()
        data = np.frombuffer(data, dtype=dtype)
        data = data.reshape(eval(shape))

        # Receive shape of X, y so we can reshape
        _, n_samples, n_features, n_classes, scenario, n_most_representative, learning_rate, delay = self.ctrl_socket.recv_multipart()
        n_samples = int(n_samples.decode())
        n_features = int(n_features.decode())
        n_classes = int(n_classes.decode())
        self.scenario = int(scenario.decode())
        self.n_most_representative = int(n_most_representative.decode())
        self.learning_rate = float(learning_rate.decode())
        self.delay = int(delay.decode())

        self.optimizer = ParallelSGDOptimizer(loss=loss_fns.single_cross_entropy_loss, grad=self.gradient, role="worker", learning_rate=self.learning_rate)

        if self.scenario == 2 and not start:
            self.X, self.y = np.vstack([self.X, data[:, :n_features]]), np.vstack([self.y, data[:, -n_classes:]])
            self._logger.debug(f"New data shape={self.X.shape}")
        else:
            self.X, self.y = data[:, :n_features], data[:, -n_classes:]

        # Check if we need to add a new axis if the dimension of y is not 2d
        if len(self.y.shape) < 2:
            self.y = self.y[:, np.newaxis]
        self._logger.info(f"Received data, X.shape={self.X.shape}, y.shape={self.y.shape}")
        self.have_work = True

        return n_samples, n_features, n_classes
                    

    def start(self):
        """Training for the worker

        Boots up the worker to start receiving data. Thereafter, the worker does the heavy lifting by computing the gradients of the parameter matrix. This is returned to the master, where the master will aggregate gradients and apply them to the global theta. The parameters will be distributed back to the worker and this occurs iteratively, to find the global minima for the parameter matrix.
        """
        poller = self.setup_poller()
        # poller.register(self.push_socket, zmq.POLLOUT)

        self._logger.info('Started Worker %s' % self.worker_id)

        try:
            start = time.time()
            i = 0
            self.scenario = 0
            n_samples = 0
            n_features = 0
            n_classes = 0

            # self.starter.send_multipart([b"READY", self.worker_id.encode()])
            # self.ctrl_socket.send(b"READY")

            while True:

                if self.connected:

                    events = dict(poller.poll())

                    if (self.ctrl_socket in events) and (events.get(self.ctrl_socket) == zmq.POLLIN):
                        command = self.ctrl_socket.recv(flags=zmq.SNDMORE)
                        self._logger.debug(f"Command={command}")

                        if command == b"WORK":
                            n_samples, n_features, n_classes = self.receive_data(start=False)

                        if command == b"HEARTBEAT":
                            self.ctrl_socket.send(b"PONG")
                            self._logger.debug("PONG")

                    if (self.subscriber in events) and (events.get(self.subscriber) == zmq.POLLIN):
                        # Read envelope with address
                        self._logger.debug("Receiving contents")
                        contents = self.subscriber.recv_multipart()
                        address = contents[0]
                        cmd = contents[1]
                        msg = contents[2:]
                        packet_size = np.sum([len(m) for m in contents])

                        self._logger.debug(f"Packet size={packet_size} bytes")

                        if cmd == b"EXIT":
                            self._logger.info("Received EXIT command")
                            break
                        else:

                            if self.scenario != 1:

                                # Receive parameter matrix on the subscriber socket
                                if cmd == b"WORKNODELAY":
                                    self.delay = 1
                                data, dtype, shape = msg
                                shape = shape.decode()

                                # Reconstruct numpy array
                                buf = memoryview(data)
                                theta = np.frombuffer(buf, dtype=dtype)
                                theta = theta.reshape(eval(shape))
                                self._logger.info(f"theta.shape{theta.shape}")
                                
                                theta = theta.copy()
                            elif self.scenario == 1:

                                # Receive numpy struct array
                                # msg = self.subscriber.recv(flags=flags, copy=True, track=False)
                                buf = memoryview(msg[0])

                                # Reconstruct theta matrix from min, max, no. of intervals and which bins
                                # each parameter value falls in
                                struct_field_names = ["min_val", "max_val", "interval", "bins"]
                                struct_field_types = [np.float32, np.float32, np.int32, 'b']
                                struct_field_shapes = [1, 1, 1, ((n_features, n_classes))]
                                dtype=(list(zip(struct_field_names, struct_field_types, struct_field_shapes)))
                                                                
                                data = np.frombuffer(buf, dtype=dtype)
                                min_theta_val, max_theta_val, interval, theta_bins = data[0]

                                # Generate lineared space vector
                                bins = np.linspace(min_theta_val, max_theta_val, interval)
                                theta = bins[theta_bins].reshape(n_features, n_classes)                             

                            theta_g = theta.copy()
                            count = 1
                            while True:
                            # Each worker does work and we get the resulting gradients
                                d_theta, batch_loss, most_representative = self.do_work(self.X, self.y, theta)
                                self._logger.debug(f"iteration = {i}, Loss = {batch_loss:7.4f}")

                                # Update the global parameters with weighted error
                                for k in np.arange(n_classes):
                                    theta[:, k] = theta[:, k] - self.learning_rate * d_theta[:, k]

                                # Let global theta influence local theta
                                for k in np.arange(theta.shape[1]):
                                    theta[:, k] = (self.learning_rate) * theta[:, k] + (1 - self.learning_rate) * theta_g[:, k]

                                if self.tf_logger is not None:
                                    self.tf_logger.histogram(f"theta={self.worker_id}", theta, i, bins=400)

                                i += 1
                                if count == self.delay:
                                    break
                                count += 1

                            # Get messages ready to send by converting them to bytes format. We do not
                            # need to send the shape since the gradients have the same shape as theta which
                            # the master already owns
                            msg = d_theta.tostring()
                            loss = str(batch_loss).encode()
                            mr = most_representative.tostring()

                            # self.push_socket.send(b"WORK")
                            # self._logger.debug("Sent work command")
                            self.push_socket.send_multipart([b"WORK", self.worker_id.encode(), msg, loss, mr])

                            self._logger.debug("Sent work back to master")
                else:

                    self._logger.info("Connecting to server")
                    self.push_socket.send_multipart([b"CONNECT", self.worker_id.encode()])
                    self._logger.debug("Connected")
                    self.connected = True

                    command = self.ctrl_socket.recv(flags=zmq.SNDMORE)

                    if command == b"WORK":
                        n_samples, n_features, n_classes = self.receive_data()

                    # Receive X, y
                    # n_samples, n_features, n_classes = self.receive_data()
                    # poller.register(self.ctrl_socket, zmq.POLLIN)

            end = time.time()

            self._logger.info("Time taken for %d iterations is %7.6fs" % (i-1, end-start))
        except KeyboardInterrupt as e:
            self._logger.info("Keyboard quit")
        except zmq.ZMQError:
            self._logger.info("ZMQError")
        finally:
            poller.unregister(self.subscriber)
            self.kill()

    def kill(self):
        """Kills sockets
        """
        self.subscriber.close()
        self.ctrl_socket.close()
        # self.context.term()