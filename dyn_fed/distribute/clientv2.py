"""Client example using zmqstreams and heartbeating. To be used with server.py

Using tensorflow.

Currently doesn't seem to speed up. Not sure what's wrong with
tf.function in eager execution mode.
"""
import logging
import time
import uuid
import os
import json
import datetime

import tensorflow as tf

import numpy as np
import zmq
from zmq import devices
from zmq.eventloop import zmqstream
from tornado import ioloop

from dyn_fed.distribute.watch_dog import ModelWatchDog
from dyn_fed.proto.utils import (params_response_to_stringv2,
                                 params_skip_response_to_string,
                                 parse_params_from_stringv2,
                                 parse_setup_from_string,
                                 setup_reponse_to_string,
                                 parse_comm_setup_from_string,
                                 parse_numpy_from_string)
from dyn_fed.utils.maths import arg_svd
from dyn_fed.lib.io.file_io import FileWatcher


# pylint: disable=no-member
class ClientV2():
    """Client class
    """
    def __init__(self, model, optimizer, strategy):
        self.sub = None
        self.push = None
        self.ctrl = None
        self.loop = None
        identity = strategy.identity
        self.identity = \
            str(uuid.uuid4()) if identity is None else f"client-{identity}"

        # Model variables
        self.model = model
        self.optimizer = optimizer
        self.strategy = strategy
        self.config = self.strategy.config
        self.train_dataset = None
        self.test_dataset = None
        self.loss_func = None
        self.epoch_accuracy = None
        self.epoch_loss_avg = None
        self.iter = 0
        self.max_iter = self.config.model.n_iterations

        self.n_samples: int = None

        # Distribute variables
        self.state = None
        self.comm_iterations = None
        self.start_comms_iter = 0
        self.comm_interval = 1
        self.comm_every_iter = 1
        self.subscribed = False
        self.test_loss_avg = None
        self.test_acc_avg = None
        self.prev_params = None
        self.model_watchdog: ModelWatchDog = None
        # Assume all clients have violated the dynamic operator threshold
        self.violated = True

        self._logger = logging.getLogger(f"dfl.distribute.{self.__class__.__name__}")

        self._logger.info("Setting up...")

    def _load_master_ip(self):
        """Load server IP from shared folder
        """
        self._logger.info("Loading in Master IP")
        datestr = datetime.datetime.now().strftime("%Y%m%d")
        ip_filename = f"ip_config_{datestr}.json"
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

        self._logger.info(f"Connected")       

    def _check_metrics(self):
        """Checks metrics on training and validation dataset

        Returns:
            train_acc (float): Training accuracy
            test_acc (float): Test accuracy
        """
        def model_validate(features, labels):
            predictions = self.model(features, training=False)
            v_loss = self.loss_func(labels, predictions)

            self.test_loss_avg(v_loss)
            self.test_acc_avg(labels, predictions)

        for x, y in self.test_dataset:
            model_validate(x, y)

        test_loss = self.test_loss_avg.result()
        test_acc = self.test_acc_avg.result()

        return test_acc, test_loss 

    def _training_loop(self):
        """Perform training loop

        Returns:
            epoch_loss (float): Loss for corresponding epoch
            most_representative(np.ndarray): Most representative data points
        """

        @tf.function
        def train_loop(x, y):

            # Calculate gradients
            with tf.GradientTape() as t:
                # training=training is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                predictions = self.model(x, training=True)
                loss = self.loss_func(y, predictions)

            grads = t.gradient(loss, self.model.trainable_variables)

            # Optimize the model
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Track progress
            self.epoch_loss_avg(loss)

            # # Compare predicted label to actual
            self.epoch_accuracy.update_state(y, predictions)

        for _ in range(self.config.comms.interval):
            start = time.time()
            for x, y in self.train_dataset:
                train_loop(x, y)
            end = time.time()
            elapsed = end - start
            self._logger.debug(f"Gradient calc elapsed time={elapsed:7.4f}")


            epoch_loss = self.epoch_loss_avg.result()
            epoch_train_acc = self.epoch_accuracy.result()

            new_params = [w.numpy() for w in self.model.trainable_weights]
            self.model_watchdog.calculate_divergence([new_params])

            if self.config.model.check_overfitting:
                test_acc, test_loss = self._check_metrics()
                self._logger.info(
                    f"iteration = {self.iter}, "
                    f"delta={self.model_watchdog.divergence:7.4f}, "
                    f"train_loss = {epoch_loss:7.4f}, test_loss={test_loss:7.4f}, "
                    f"train_acc={epoch_train_acc*100:7.4}%, "
                    f"test_acc={test_acc*100:7.4f}%"
                )
            else:
                self._logger.info(
                    f"iteration={self.iter}, train_loss={epoch_loss:7.4f}, "
                    f"delta={self.model_watchdog.divergence}"
                )

            self.epoch_loss_avg.reset_states()
            self.epoch_accuracy.reset_states()
            self.test_loss_avg.reset_states()
            self.test_acc_avg.reset_states()

            self.iter += 1

        return epoch_loss

    def _recv_comm_info(self, msg):
        """Receive communication information
        """
        self.comm_iterations, self.comm_interval, self.comm_every_iter = \
                parse_comm_setup_from_string(msg[1])
        self.start_comms_iter = self.max_iter - \
            self.comm_iterations
            
        self._logger.debug(
            f"Comm iterations={self.comm_iterations}, "
            f"Comm interval={self.comm_interval}"
        )
    
    def setup(self, train_dataset, test_dataset=None):
        """Setup server with train and test data and train steps
        """
        self.train_dataset = train_dataset
        
        if self.config.model.check_overfitting:
            X_test, y_test = test_dataset
            n_samples = X_test.shape[0]
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            test_dataset = test_dataset.batch(n_samples)
            self.test_dataset = test_dataset

        # Define loss function
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.epoch_loss_avg = tf.keras.metrics.Mean()
        self.epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.test_loss_avg = tf.keras.metrics.Mean()
        self.test_acc_avg = tf.keras.metrics.SparseCategoricalAccuracy()

    def recv_work(self, msg):
        """Receive work
        """
        self._logger.info("Receiving work...")
        cmd = msg[0]

        if cmd == b"WORK_DATA":
            X, y, n_samples, state = parse_setup_from_string(msg[1])

            self.n_samples = n_samples
            self.state = state
            # self.X = X
            # self.y = y
            self.train_dataset = tf.data.Dataset.from_tensor_slices(
                (X, y)
            )
            self.train_dataset = (
                self.train_dataset.shuffle(self.config.data.shuffle_buffer_size)
                .batch(self.config.data.batch_size)
            )

            self.model_watchdog = ModelWatchDog(
                n_samples=n_samples,
                n_classes=np.max(y) + 1,
                delta_threshold=self.config.distribute.delta_threshold
            )

            # self._logger.info(f"X.shape={self.X.shape}, y.shape={self.y.shape}")
            self._logger.info(f"X.shape={X.shape}, y.shape={y.shape}")

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

    def _shap_work(self, msg):
        """Do shap work
        """
        self._logger.info("Shapley loops...")
        parameters = parse_params_from_stringv2(msg[2])
        packet_size = len(msg[2])
        self._logger.debug(f"Packet size of params={packet_size}")

        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(10, activation="sigmoid")
        ])

        optimizer = tf.keras.optimizers.Adam(0.001)

        loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        test_loss_avg = tf.keras.metrics.Mean()
        test_acc_avg = tf.keras.metrics.SparseCategoricalAccuracy()
        
        # Update local model using global parameters
        model.set_weights(parameters)

        @tf.function
        def train_loop(x, y):

            # Calculate gradients
            with tf.GradientTape() as t:
                # training=training is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                predictions = model(x, training=True)
                loss = loss_func(y, predictions)

            grads = t.gradient(loss, model.trainable_variables)

            # Optimize the model
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg(loss)

            # # Compare predicted label to actual
            epoch_accuracy.update_state(y, predictions)

        # Do some work
        tic = time.time()
        for _ in np.arange(self.config.shap.n_iterations):
            start = time.time()
            for x, y in self.train_dataset:
                train_loop(x, y)
            end = time.time()
            elapsed = end - start
            self._logger.debug(f"Gradient calc elapsed time={elapsed:7.4f}")

            epoch_loss = epoch_loss_avg.result()
            epoch_train_acc = epoch_accuracy.result()

            self._logger.info(
                f"iteration={self.iter}, train_loss={epoch_loss:7.4f}, "
                f"delta={self.model_watchdog.divergence}"
            )

            epoch_loss_avg.reset_states()
            epoch_accuracy.reset_states()
            test_loss_avg.reset_states()
            test_acc_avg.reset_states()

        self._logger.info("blocked for %.3f s", (time.time() - tic))

        data = [
            params_response_to_stringv2(
                model.trainable_weights,
                epoch_loss
            )
        ]

        self._logger.debug(f"Sending shap work back to server")
        multipart = [b"SHAP", self.identity.encode()]
        multipart.extend(data)
        self.push.send_multipart(multipart)
        self.model_watchdog.reset()

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
                parameters = parse_params_from_stringv2(msg[2])
                packet_size = len(msg[2])
                self._logger.debug(f"Packet size of params={packet_size}")
                
                # Update local model using global parameters
                self.model.set_weights(parameters)
            else:
                self._logger.debug("Delta < threshold - no need to update params")

            update_ref_model = msg[3]
            update_ref_model = parse_numpy_from_string(update_ref_model, dtype=bool, shape=())
            self._logger.debug(f"Update ref model={update_ref_model}")
            if update_ref_model:
                self.model_watchdog.update_ref_model(parameters)

            # Do some work
            tic = time.time()
            epoch_loss = self._training_loop()
            self._logger.info("blocked for %.3f s", (time.time() - tic))
            if self.config.comms.mode == 3:
                # Only do this for dynamic averaging technique
                if self.model_watchdog.divergence <= self.config.distribute.delta_threshold:
                    self.violated = False
                # If exceeds threshold then will communicate
                else:
                    self._logger.debug(
                        f"Sending work, delta >= threshold="
                        f"{self.config.distribute.delta_threshold}"
                    )
                    self.violated = True

            data = [
                params_response_to_stringv2(
                    self.model.trainable_weights,
                    epoch_loss
                )
            ]

            send_work = (self.iter - 1) % self.comm_interval == 0
            send_work = send_work or (self.iter >= self.comm_every_iter)
            send_work = send_work and self.violated # If work is violated, we send the work
            if self.config.comms.mode == 3:
                send_work = send_work or (self.iter == self.max_iter)

            # send_work = send_work or (self.iter <= self.comm_every_iter)
            self._logger.debug(
                f"Send work={send_work}, iter={self.iter}, "
                f"comm_interval={self.comm_interval}"
            )
            if send_work:
                multipart = [b"WORK", self.identity.encode()]
                multipart.extend(data)
                self.push.send_multipart(multipart)
            elif not send_work and self.config.comms.mode == 3:
                self._logger.debug(f"Skipping sending work...")
                multipart = [b"SKIP", self.identity.encode()]
                data = [params_skip_response_to_string(self.model_watchdog.divergence)]
                multipart.extend(data)
                self.push.send_multipart(multipart)

        if cmd == b"SHAP":
            self._shap_work(msg)

        if cmd == b"EXIT":
            self._logger.info("Ending session")
            self.kill()

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

    def kill(self):
        """Kills sockets
        """
        self._logger.info("Cleaning up")
        self.loop.stop()
