"""Contains all worker logic for fault tolerant ml

All worker devices will contain worker logic
"""

import json
import logging
import os
import time
import uuid

import numpy as np
import zmq.green as zmq

from fault_tolerant_ml.distribute import Coordinator
from fault_tolerant_ml.distribute.states import REMAP
from fault_tolerant_ml.operators import Tensor
from fault_tolerant_ml.proto.utils import (params_response_to_string,
                                           parse_params_from_string,
                                           parse_setup_from_string)
from fault_tolerant_ml.tools import TFLogger
from fault_tolerant_ml.utils import setup_logger
from fault_tolerant_ml.utils.maths import reconstruct_approximation


class Worker(object):
    """Worker class for distributed machine learning system
    
    Attributes:
        worker_id (str): Unique identifier for worker
        subscriber (zmq.Socket): zmq.SUB socket which subscribes to all master published messages
        connected (bool): Whether or not the worker is connected successfully to the master
    """
    def __init__(self, model, verbose, identity=None):

        self.worker_id = str(uuid.uuid4()) if identity is None else f"worker-{identity}"
        self.subscriber = None
        self.connected = False

        self.model = model
        self.strategy = self.model.strategy
        self.state = None

        # Executor params
        self.remap = self.model.strategy.remap
        self.quantize = self.model.strategy.quantize
        self.comm_period = self.model.strategy.comm_period
        self.send_gradients = self.model.strategy.send_gradients

        # ZMQ variables
        self.context: zmq.Context = None
        self.ctrl_socket: zmq.Socket = None
        self.push_socket: zmq.Socket = None

        # Model params
        self.n_samples: int = 0
        self.n_features: int = 0
        self.n_classes: int = 0
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.have_work: bool = False

        setup_logger(filename=f'log-{self.worker_id}.log', level=verbose)
        self._logger = logging.getLogger(f"ftml.distribute.{self.__class__.__name__}")

        ip_filename = "ip_config.json"
        if "SLURM_JOBID" in os.environ:
            slurm_job_id = os.environ["SLURM_JOBID"]
            ip_filename = f"ip_config_{slurm_job_id}.json"

        full_path = os.path.join(self.strategy.config_folder, ip_filename)
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                ip_config = json.load(f)
        else:
            # file_watcher = FileWatcher(self.model.strategy.shared_folder, full_path)
            # file_found = file_watcher.run(timeout=30)
            # if file_found:
            #     with open(full_path, "r") as f:
            #         ip_config = json.load(f)
            # else:
            raise FileNotFoundError("IP Config file not found")

        self.master_ip_address = ip_config["ipAddress"]

        self.encoded_name = self.model.encode_name

        self._tf_logger = None
        self.coordinator = Coordinator()

    def _setup_poller(self):
        """Register necessary sockets for poller
        """
        poller = zmq.Poller()
        poller.register(self.subscriber, zmq.POLLIN | zmq.POLLERR)
        poller.register(self.ctrl_socket, zmq.POLLIN | zmq.POLLERR)

        return poller

    def _receive_data(self, start=True):
        """Receives data from worker

        Receives and makes sense of the data received from the master.
        Also initializes the optimizer since we receive the learning rate
        from the master at this stage.

        Args:
            start (bool): Whether or not we received the data at the beginning
            of the workers life span. 

        Returns:
            n_samples (int): No. of samples in the dataset for a worker
            n_features (int): No. of features in the dataset
            n_classes (int): No. of classes/labels
        """
        msg = self.ctrl_socket.recv_multipart() # pylint: disable=unbalanced-tuple-unpacking
        X, y, n_samples, state = parse_setup_from_string(msg[0])

        # Receive shape of X, y so we can reshape
        self.n_samples = n_samples
        self.n_features = X.shape[1]
        self.n_classes = y.shape[1]
        self.state = state

        if "TFDIR" in os.environ:
            logdir = os.path.join(os.environ["TFDIR"], f"tf/{self.encoded_name}/{self.worker_id}")
            self._tf_logger = TFLogger(logdir)

        self._logger.debug(f"Data size={X.shape}")

        if self.remap == 1 and not start and state == REMAP:
            self._logger.debug(f"self.X.shape={self.X.shape}, X.shape={X.shape}")
            self.X, self.y = (
                np.vstack([self.X.data, X]),
                np.vstack([self.y.data, y])
            )
            self._logger.debug(f"New data shape={self.X.shape}")
        else:
            self.X, self.y = X, y

        # Check if we need to add a new axis if the dimension of y is not 2d
        if len(self.y.shape) < 2:
            self.y = self.y[:, np.newaxis]

        # Tensorize the retrieved data
        self.X = Tensor(self.X)
        self.y = Tensor(self.y)

        # Cap batch size to number of samples we have
        if self.model.batch_size > self.X.shape[0]:
            self.model.batch_size = self.X.shape[0]

        self._logger.info(f"Received data, X.shape={self.X.shape}, y.shape={self.y.shape}")
        self.have_work = True

    def _read_envelope(self):
        """Read envelope and return command and message

        Returns:
            cmd (bytes): Command whether to EXIT or WORK
            msg (bytes): Parameter tensor from master in bytes
        """
        self._logger.debug("Receiving contents")
        contents = self.subscriber.recv_multipart()
        _ = contents[0]
        cmd = contents[1]
        msg = contents[2:]
        packet_size = np.sum([len(m) for m in contents])

        self._logger.debug(f"Packet size={packet_size} bytes")

        return cmd, msg

    def _parse_msg(self, cmd, msg):
        """Parse contents received from master

        Args:
            cmd (byte): Command received from master
            msg (byte array): Parameter Tensor as bytes

        Returns:
            parameters (np.ndarray): Parameter tensor as a numpy array
        """
        if self.quantize != 1:
            # Receive parameter matrix on the subscriber socket
            if cmd == b"WORKNODELAY":
                self.comm_period = 1

            parameters = parse_params_from_string(msg[0])

            for i in np.arange(self.model.n_layers):
                self.model.layers[i].W.data = parameters[i][0]
                self.model.layers[i].b.data = parameters[i][1]
            
        elif self.quantize == 1:
            
            # TODO: Recontruct approximation for quantized for many layers
            # Receive numpy struct array
            buf = memoryview(msg[0])

            # Reconstruct W matrix from min, max, no. of 
            # intervals and which bins each parameter value
            # falls in
            shape = (self.n_features, self.n_classes)
            W = reconstruct_approximation(buf, shape, r_dtype=np.float32) # pylint: disable=unused-variable

        # W = Tensor(W, is_param=True)
        # W_g = W.copy()
        # self.model.layers[0].W = W
        # self.model.layers[0].W = W

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
            W_g=W_g)
        most_representative = self.model.optimizer.most_rep
        
        return batch_loss, most_representative

    def _training_loop(self):
        """Perform training loop

        Returns:
            epoch_loss (float): Loss for corresponding epoch
            most_representative(np.ndarray): Most representative data points
        """
        count = 1
        while True:
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
            # self._logger.info(
            # f"iteration = {self.model.iter}, Loss = {batch_loss:7.4f}"
            # )
            self._logger.info(
                f"iteration = {self.model.iter}, Loss = {epoch_loss:7.4f}"
            )

            # Log to tensorboard
            if self._tf_logger is not None:
                self._tf_logger.histogram(
                    f"W={self.worker_id}",
                    self.model.layers[0].W.data,
                    self.model.iter, bins=400
                )
                self._tf_logger.histogram(
                    f"d_W={self.worker_id}",
                    self.model.layers[0].W.grad.data,
                    self.model.iter, bins=400
                )
                self._tf_logger.scalar(
                    f"loss-{self.worker_id}",
                    batch_loss,
                    self.model.iter
                )

            self.model.iter += 1
            if count == self.comm_period:
                break
            count += 1

        return epoch_loss, most_representative

    def _prep_multipart(self, data):
        """Prepare multipart message
        """
        multipart = [b"WORK", self.worker_id.encode()]
        multipart.extend(data)
        return multipart

    def _check_subscriber_events(self):
        """Check subscriber events

        Args:
            events (dict): Dictionary of zmq events

        Returns:
            session_end (bool): Whether or not exit command was received
        """
        # Read envelope with address
        cmd, msg = self._read_envelope()
        session_end = False

        if cmd == b"EXIT":
            # End session
            self._logger.info("Received EXIT command")
            session_end = True
        else:
            # Perform session
            # Parse parameters
            self._parse_msg(cmd, msg)
            # Do training
            epoch_loss, most_representative = self._training_loop()

            self._logger.debug(f"Most_rep.shape={most_representative.shape}")

            # Get messages ready to send by converting them to
            # bytes format. We do not need to send the shape 
            # since the gradients have the same shape as W which
            # the master already owns
            if self.quantize:
                # d_W = linspace_quantization(d_W, interval=100)
                # self.model.layers[0].W = \
                # linspace_quantization(self.model.layers[0].W, interval=100)
                # d_W.data = linspace_quantization(d_W.data, interval=100)
                # self.model.layers[0].W.grad.data = \
                # linspace_quantization(
                # self.model.layers[0].W.grad.data, 
                # interval=100
                # )
                # self.model.layers[0].W.data = \
                # linspace_quantization(self.model.layers[0].W.data, interval=100)
                pass

            self._logger.debug(f"Send gradients flag={self.send_gradients}")
            # msg = self.model.layers[0].W.tostring()
            
            if self.send_gradients:
                # msg = d_W.tostring()
                # msg = self.model.layers[0].W.grad.tostring()
                data = self.model.parameters(grad=True)

            data = [params_response_to_string(self.model.layers, most_representative, epoch_loss)]

            multipart = self._prep_multipart(data)
            self.push_socket.send_multipart(multipart)

            self._logger.debug("Sent work back to master")

        return session_end

    def _handle_dealer_events(self, events):
        """Handle events on control socket (zmq.Dealer)

        Args:
            events (dict): Dictionary of zmq.Events
        """
        if (self.ctrl_socket in events) \
            and (events.get(self.ctrl_socket) == zmq.POLLIN):
            command = self.ctrl_socket.recv(flags=zmq.SNDMORE)
            self._logger.debug(f"Command={command}")

            if command == b"WORK":
                self._receive_data(start=False)

            if command == b"HEARTBEAT":
                self.ctrl_socket.send(b"PONG")
                self._logger.debug("PONG")

    def _handle_subscriber_events(self, events):
        """Handles subscriber events

        Args:
            events (dict): Dictionary of zmq.Events

        Returns:
            session_end (bool): Whether or not exit command was received
        """
        session_end = False
        if (self.subscriber in events) and (events.get(self.subscriber) == zmq.POLLIN):
            # Check subsriber event
            session_end = self._check_subscriber_events()
        return session_end


    def connect(self):
        """Prepare our context, push socket and publisher
        """
        self.context = zmq.Context()
        self.subscriber = self.context.socket(zmq.SUB) # pylint: disable=no-member
        # self.subscriber.connect("tcp://localhost:5563")
        self.subscriber.connect(f"tcp://{self.master_ip_address}:5563")
        self.subscriber.setsockopt(zmq.SUBSCRIBE, b"") # pylint: disable=no-member

        self.push_socket = self.context.socket(zmq.PUSH) # pylint: disable=no-member
        # self.push_socket.connect("tcp://localhost:5562")
        self.push_socket.connect(f"tcp://{self.master_ip_address}:5562")

        self.ctrl_socket = self.context.socket(zmq.DEALER) # pylint: disable=no-member
        self.ctrl_socket.setsockopt_string(zmq.IDENTITY, self.worker_id) # pylint: disable=no-member
        # self.ctrl_socket.connect("tcp://localhost:5565")
        self.ctrl_socket.connect(f"tcp://{self.master_ip_address}:5565")

        self._logger.info(
            f"Connected to ip address {self.master_ip_address}, "
            f"on ports 5563, 5562 & 5565"
        )

    def start(self):
        """Worker session

        Boots up the worker to start receiving data. Thereafter, the worker
        does the heavy lifting by computing the gradients of the parameter
        matrix. This is returned to the master, where the master will aggregate
        gradients and apply them to the global W. The parameters will be
        distributed back to the worker and this occurs iteratively, to find the
        global minima for the parameter matrix.
        """
        poller = self._setup_poller()

        self._logger.info('Started Worker %s', self.worker_id)

        try:
            start_time = time.time()
            self.n_samples = 0
            self.n_features = 0
            self.n_classes = 0

            # self.starter.send_multipart([b"READY", self.worker_id.encode()])
            # self.ctrl_socket.send(b"READY")

            while True:
                if self.connected:
                    events = dict(poller.poll())

                    self._handle_dealer_events(events)

                    session_end = self._handle_subscriber_events(events)
                    if session_end:
                        break
                else:
                    self._logger.info("Connecting to server")
                    self.push_socket.send_multipart([b"CONNECT", self.worker_id.encode()])
                    self._logger.info("Connected")
                    self.connected = True

                    command = self.ctrl_socket.recv(flags=zmq.SNDMORE)

                    if command == b"WORK":
                        self._receive_data()

            elapsed_time = time.time() - start_time

            self._logger.info(
                "Time taken for %d iterations is %7.6fs",
                self.model.iter-1, elapsed_time
            )
        except KeyboardInterrupt:
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
