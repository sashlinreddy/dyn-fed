"""Client example using zmqstreams and heartbeating. To be used with server.py
"""
import logging
import time
import uuid
import os
import json

import numpy as np
import zmq
from zmq import devices
from zmq.eventloop import zmqstream
from tornado import ioloop

from dyn_fed.operators import Tensor
from dyn_fed.metrics import accuracy_scorev2
from dyn_fed.proto.utils import (params_response_to_string,
                                           parse_params_from_string,
                                           parse_setup_from_string,
                                           setup_reponse_to_string,
                                           parse_comm_setup_from_string)
from dyn_fed.utils.maths import arg_svd
from dyn_fed.lib.io.file_io import FileWatcher


# pylint: disable=no-member
class Client():
    """Client class
    """
    def __init__(self, model, identity=None):
        self.sub = None
        self.push = None
        self.ctrl = None
        self.loop = None
        self.identity = \
            str(uuid.uuid4()) if identity is None else f"worker-{identity}"

        # Model variables
        self.model = model
        self.config = self.model.strategy.config
        self.X = None
        self.y = None
        self.X_valid = None # To check for overfitting/underfitting.
        self.y_valid = None # To check for overfitting/underfitting
        self.n_samples: int = None
        self.n_features: int = None
        self.n_classes: int = None
        self.check_overfitting = self.config.model.check_overfitting

        # Distribute variables
        self.state = None
        self.comm_iterations = None
        self.start_comms_iter = 0
        self.comm_interval = 1
        self.comm_every_iter = 1
        self.subscribed = False
        self.prev_params = None
        # Assume all workers have violated the dynamic operator threshold
        self.violated = True

        self._logger = logging.getLogger(f"dfl.distribute.{self.__class__.__name__}")

        self._logger.info("Setting up...")

    def _load_master_ip(self):
        """Load master IP from shared folder
        """
        self._logger.info("Loading in Master IP")
        ip_filename = "ip_config.json"
        if "SLURM_JOBID" in os.environ:
            slurm_job_id = os.environ["SLURM_JOBID"]
            ip_filename = f"ip_config_{slurm_job_id}.json"

        full_path = os.path.join(self.config.executor.config_folder, ip_filename)
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                ip_config = json.load(f)
        else:
            file_watcher = FileWatcher(self.config.executor.config_folder, full_path)
            file_found = file_watcher.run(timeout=30)
            if file_found:
                self._logger.info("Found IP Address file. Loading...")
                with open(full_path, "r") as f:
                    ip_config = json.load(f)
            else:
                raise FileNotFoundError("IP Config file not found")

        master_ip_address = ip_config["ipAddress"]

        return master_ip_address

    def _connect(self):
        """Connect to sockets
        """
        master_ip_address = self._load_master_ip()
        self._logger.info(f"Connecting sockets on {master_ip_address}")
        self.loop = ioloop.IOLoop()
        context = zmq.Context()

        dev = devices.ThreadDevice(zmq.FORWARDER, zmq.SUB, zmq.DEALER)
        dev.setsockopt_in(zmq.SUBSCRIBE, b"")
        dev.setsockopt_out(zmq.IDENTITY, self.identity.encode())
        dev.connect_in(f'tcp://{master_ip_address}:5564')
        dev.connect_out(f'tcp://{master_ip_address}:5561')
        dev.start()

        subscriber = context.socket(zmq.SUB) # pylint: disable=no-member
        subscriber.connect(f"tcp://{master_ip_address}:5560")
        # subscriber.connect(f"tcp://{master_ip_address}:5563")
        subscriber.setsockopt(zmq.SUBSCRIBE, b"") # pylint: disable=no-member

        self.push = context.socket(zmq.PUSH) # pylint: disable=no-member
        self.push.connect(f"tcp://{master_ip_address}:5567")
        # push_socket.connect(f"tcp://{master_ip_address}:5562")

        ctrl_socket = context.socket(zmq.DEALER) # pylint: disable=no-member
        ctrl_socket.setsockopt_string(zmq.IDENTITY, self.identity) # pylint: disable=no-member
        ctrl_socket.connect(f"tcp://{master_ip_address}:5566")
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
            W_g=W_g)
        most_representative = self.model.optimizer.most_rep
        
        return batch_loss, most_representative

    def _check_metrics(self):
        """Checks metrics on training and validation dataset

        Returns:
            train_acc (float): Training accuracy
            test_acc (float): Test accuracy
        """
        y_pred = self.model.forward(self.X_valid)
        y_train_pred = self.model.forward(self.X)
        
        test_loss = self.model.optimizer.compute_loss(
            self.y_valid.data,
            y_pred.data,
            reduce=True
        )

        train_acc = accuracy_scorev2(self.y.data, y_train_pred.data)
        test_acc = accuracy_scorev2(self.y_valid.data, y_pred.data)

        return train_acc, test_acc, test_loss

    def _training_loop(self):
        """Perform training loop

        Returns:
            epoch_loss (float): Loss for corresponding epoch
            most_representative(np.ndarray): Most representative data points
        """
        self.prev_params = [p.data.copy() for p in self.model.parameters()]

        for _ in range(self.config.comms.interval):
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
                epoch_loss += batch_loss
                n_batches += 1
            epoch_loss /= n_batches

            new_params = self.model.parameters()
            delta = np.max([
                np.linalg.norm(o - n.data)**2
                for o, n in zip(self.prev_params, new_params)
            ])

            # Check metrics
            if self.check_overfitting:
                train_acc, test_acc, test_loss = self._check_metrics()
                self._logger.info(
                    f"iteration={self.model.iter}, delta={delta:7.4f}, "
                    f"train_loss={epoch_loss:7.4f}, test_loss={test_loss:7.4f}, "
                    f"train acc={train_acc*100:7.4f}%, "
                    f"test acc={test_acc*100:7.4f}%"
                )
            else:
                self._logger.info(
                    f"iteration={self.model.iter}, train_loss={epoch_loss:7.4f}, delta={delta}"
                )

            self.model.iter += 1

        return epoch_loss, most_representative, delta

    def _recv_comm_info(self, msg):
        """Receive communication information
        """
        self.comm_iterations, self.comm_interval, self.comm_every_iter = \
                parse_comm_setup_from_string(msg[1])
        self.start_comms_iter = self.model.max_iter - \
            self.comm_iterations
            
        self._logger.debug(
            f"Comm iterations={self.comm_iterations}, "
            f"Comm interval={self.comm_interval}"
        )

    def recv_work(self, msg):
        """Receive work
        """
        self._logger.info("Receiving work...")
        cmd = msg[0]

        if cmd == b"WORK_DATA":
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

            if self.config.comms.mode == 1 or \
                self.config.distribute.aggregate_mode == 3:
                tic = time.time()
                idx_95 = arg_svd(X)
                self._logger.info(
                    f"Time to calculate svd idx {(time.time() - tic):.3f} s"
                )

                # Send back idx_95 to determine dynamic communication strategy
                data = [setup_reponse_to_string(idx_95)]
                multipart = [b"SVD", self.identity.encode()]
                multipart.extend(data)

                self.push.send_multipart(multipart)
            if self.config.comms.mode != 1:
                if not self.subscribed:
                    # After receiving data we can recv params
                    self.sub.on_recv(self.recv_params)
                    self.subscribed = True

        if cmd == b"COMM_INFO":
            self._logger.debug("Receiving communication info")
            self._recv_comm_info(msg)

            if not self.subscribed:
                # After receiving data we can recv params
                self.sub.on_recv(self.recv_params)
                self.subscribed = True

    def update_params(self, parameters):
        """Update model parameters

        Args:
            parameters (list of model parameters for each layer (if ANN))
        """
        for i in np.arange(self.model.n_layers):
            self.model.layers[i].W.data = parameters[i][0]
            self.model.layers[i].b.data = parameters[i][1]

    def recv_params(self, msg):
        """Recv params
        """
        _ = msg[0] # Topic
        cmd = msg[1]
        if cmd == b"WORK_PARAMS":
            self._logger.info("Receiving params...")
            # Simulating https://arxiv.org/abs/1807.03210
            # If delta greater than threshold then we communicate paramaters
            # back to server, else we continue
            # If you are less than the dynamic operator threshold, then you skip
            # communicating back to coordinator
            if self.violated:
                parameters = parse_params_from_string(msg[2])
                packet_size = len(msg[2])
                self._logger.debug(f"Packet size of params={packet_size}")
                self.update_params(parameters)
            else:
                self._logger.debug("Delta < threshold - no need to update params")

            # Do some work
            tic = time.time()
            # Initialize to some low number so that all clients are communicating
            delta = 0.0
            
            epoch_loss, most_representative, delta = self._training_loop()
            if self.config.comms.mode == 3:
                # Only do this for dynamic averaging technique
                if delta < self.config.distribute.delta_threshold:
                    self.violated = False
                # If exceeds threshold then will communicate
                else:
                    self._logger.debug(
                        f"Sending work, delta >= threshold="
                        f"{self.config.distribute.delta_threshold}"
                    )
                    self.violated = True

            self._logger.info("blocked for %.3f s", (time.time() - tic))

            data = [
                params_response_to_string(
                    self.model.layers,
                    most_representative,
                    epoch_loss
                )
            ]

            send_work = (self.model.iter - 1) % self.comm_interval == 0
            self._logger.debug(f"send_work={send_work}, {self.model.iter}, {self.comm_interval}")
            send_work = send_work or (self.model.iter >= self.comm_every_iter)
            send_work = send_work and self.violated # If work is violated, we send the work
            # send_work = send_work or (self.model.iter <= self.comm_every_iter)
            self._logger.debug(f"Send work={send_work}")
            if send_work:
                multipart = [b"WORK", self.identity.encode()]
                multipart.extend(data)
                self.push.send_multipart(multipart)
            elif not send_work and self.config.comms.mode == 3:
                self._logger.debug(f"Skipping sending work")
                multipart = [b"SKIP", self.identity.encode()]
                self.push.send_multipart(multipart)

        if cmd == b"EXIT":
            self._logger.info("Ending session")
            self.kill()

    def setup(self, X_valid=None, y_valid=None):
        """Setup master with data
        """
        self.X_valid = X_valid
        self.y_valid = y_valid

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
        except Exception as e:
            self._logger.exception(e)
            self.kill()

    def kill(self):
        """Kills sockets
        """
        self._logger.info("Cleaning up")
        self.loop.stop()
