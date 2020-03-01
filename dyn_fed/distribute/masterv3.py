"""Master with heartbeater
"""
import logging
import signal
import time
import socket
import os
import json
from typing import Optional, Tuple, Callable

import gevent
import numpy as np
import zmq.green as zmq

import tensorflow as tf

from dyn_fed.data.utils import next_batch, next_batch_unbalanced
from dyn_fed.distribute import WatchDog
from dyn_fed.distribute.heartbeater import Heartbeater
from dyn_fed.distribute.states import (COMPLETE, MAP, MAP_PARAMS,
                                                 START)
from dyn_fed.metrics import accuracy_scorev2
from dyn_fed.proto.utils import (params_to_stringv2,
                                 parse_params_response_from_stringv2,
                                 parse_setup_response_from_string,
                                 setup_to_string,
                                 comms_setup_to_string)
from dyn_fed.tools import TFLogger

# pylint: disable=no-member

class MasterV3():
    """Master class
    """
    def __init__(self, model, optimizer, strategy, period=1000):
        # Sockets
        self.heart_pub_socket = None
        self.heart_ctrl_socket = None
        self.pub_socket = None
        self.ctrl_socket = None
        self.pull_socket = None
        # Polling
        self.poller = None

        # Model variables
        self.strategy = strategy
        self.model = model # Keras model
        self.optimizer = optimizer # Keras optimizer
        self.config = self.strategy.config

        # Counter
        self.iter = 0
        self.n_iterations = int(
            np.ceil(
                self.config.model.n_iterations /
                self.config.comms.interval
                )
        )

        self.train_dataset = None
        self.test_dataset = None
        self.test_loss = None
        self.train_accuracy = None
        self.test_accuracy = None
        self.loss_func = None

        self._calculated_byte_size = False
        self._n_mbs = 0.0
        self.svd_time = 0

        # Environment variables
        self.state = START
        self.heartbeater = Heartbeater(self.strategy.n_workers, period)
        self.watch_dog = WatchDog()

        self._logger = logging.getLogger(f"dfl.distribute.{self.__class__.__name__}")
        self._tf_logger = None
        # Get ipaddress for workers to connect to
        self._save_ip()

    def _save_ip(self):
        """Save IP address to shared folder
        """
        self.hostname = socket.gethostname()
        self.ip_address = socket.gethostbyname(self.hostname)

        self._logger.info(f"Master on ip={self.ip_address}")

        ip_filename = "ip_config.json"
        if "SLURM_JOBID" in os.environ:
            slurm_job_id = os.environ["SLURM_JOBID"]
            ip_filename = f"ip_config_{slurm_job_id}.json"

        ip_config = {"ipAddress" : self.ip_address}
        config_folder = self.strategy.config['executor']['config_folder']
        with open(os.path.join(config_folder, ip_filename), "w") as f:
            json.dump(ip_config, f)

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

    def _track_samples(self, heart, i, x_batch):
        """Track samples
        """
        if self.state == START:
            lower_bound = x_batch.shape[0] * i
            upper_bound = lower_bound + x_batch.shape[0]
            global_idxs = np.arange(lower_bound, upper_bound)
            local_idxs = np.arange(x_batch.shape[0])
            idx_mapping = dict(zip(global_idxs, local_idxs))
            self.watch_dog.states[heart].mapping = idx_mapping

    def _map(self):
        """Map data to workers
        """
        if self.state == MAP:
            self._logger.info("Sending work to workers")
            # First map data

            self._logger.debug(f"State={self.state}")

            X, y = self.train_dataset
            if X.ndim > 2:
                X = X.reshape(X.shape[0], -1)
            n_samples = X.shape[0]
            if self.config.data.unbalanced:
                hearts = len(self.heartbeater.hearts)
                batch_gen = next_batch_unbalanced(
                    X,
                    y,
                    hearts,
                    shuffle=self.config.model.shuffle
                )
            else:
                batch_size = int(np.ceil(n_samples / len(self.heartbeater.hearts)))
                batch_gen = next_batch(
                    X,
                    y,
                    batch_size,
                    shuffle=self.config.model.shuffle,
                    overlap=0.0
                )

            self._logger.debug(f"Workerstates={self.watch_dog.states}")

            for i, heart in enumerate(self.heartbeater.hearts):
                x_batch, y_batch = next(batch_gen)
                x_batch = x_batch
                y_batch = y_batch
                self._logger.debug(f"X.shape={x_batch.shape}, y.shape={y_batch.shape}")

                self._track_samples(heart, i, x_batch)
                msg = [setup_to_string(x_batch, y_batch, n_samples, self.state)]
                multipart = [heart, b"WORK_DATA"]
                multipart.extend(msg)
                self.ctrl_socket.send_multipart(multipart)

            self.state = MAP_PARAMS

            # Keep track of iterations for each worker
            for worker in self.watch_dog.states:
                worker.comm_iterations = self.n_iterations

            if self.config.comms.mode == 1 or \
                self.config.distribute.aggregate_mode == 3:
                self._calculate_dynamic_comms()

        if self.state == MAP_PARAMS:
            # Determine if worker needs to communicate
            if self.config.comms.mode == 2:
                self._logger.debug("Sending communication info")
                self._send_comm_info()

            # Map params
            msg = [params_to_stringv2(self.model.trainable_weights)]
            
            multipart = [b"", b"WORK_PARAMS"]
            multipart.extend(msg)

            self._logger.info("Sending params")
            self.pub_socket.send_multipart(multipart)

    def _check_metrics(self):
        """Checks metrics on training and validation dataset

        Returns:
            train_acc (float): Training accuracy
            test_acc (float): Test accuracy
        """
        @tf.function
        def model_validate(features, labels):
            predictions = self.model(features, training=False)
            v_loss = self.loss_func(labels, predictions)

            self.test_loss(v_loss)
            self.test_accuracy(labels, predictions)

        for x, y in self.test_dataset:
            model_validate(x, y)

        train_acc = self.train_accuracy.result()
        test_acc = self.test_accuracy.result()
        test_loss = self.test_loss.result()

        self.test_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_accuracy.reset_states()
        

        return train_acc, test_acc, test_loss

    
    def _reduce(self,
                weights,
                errors,
                model,
                samples,
                n_samples,
                workers_received,
                mode=0,
                epsilon=1e-8):
        """Aggregate received parameters

        Args:
            d_Wbs (list): List of parameters received
            errors (list): List of losses for each worker
            model (dfl.Model): Model being used
            samples (list): List containing number of samples for each worker
            n_samples (int): Total number of samples
            mode (int): Aggregation mode (default: 0)

        Returns:
            parameters (list): List of numpy matrices for each layer
            epoch_loss (float): Aggregated epoch loss
        """
        # pylint: disable=too-many-arguments, too-many-locals
        # Iterate through workers and weight parameters by corresponding epoch loss
        parameters = [
            np.zeros_like(w.numpy()) for w in model.trainable_weights
        ]
        sum_es = np.sum(np.exp(errors))
        sum_svds = np.sum(
            np.exp([self.watch_dog.states[worker].svd_idx for worker in workers_received])
        )
        epoch_loss = np.mean(errors)
        for j in np.arange(len(errors)):
            weight = 1.0 / len(errors)  # Default aggregation is the average across workers
            # Weight by loss calculated by worker - worker with highest loss has greater weight
            if mode == 1:
                pass # TODO: Add number of samples for each worker to heartbeat version
                # n_samples_worker = samples[j]
                # weight = n_samples_worker / n_samples
            elif mode == 2:
                weight = np.exp(errors[j]) / sum_es
            elif mode == 3:
                worker = workers_received[j]
                weight = np.exp(self.watch_dog.states[worker].svd_idx) / sum_svds
            self._logger.debug(f"worker={j}, weight={weight}, loss={errors[j]}")
            for k in np.arange(len(model.trainable_weights)):
                parameters[k] += (
                    weights[j][k] * weight
                    if weight > 0
                    else weights[j][k] * epsilon
                ) # For parameters

        return parameters, epoch_loss

    def _gather(self, worker, content, errors, weights):
        """Gather parameters from worker
        """
        parameters, loss = \
            parse_params_response_from_stringv2(content)

        self._logger.debug(
            f"Received work from {worker}"
        )

        # Update previous and current loss
        self.watch_dog.states[worker].prev_loss = \
            self.watch_dog.states[worker].current_loss

        self.watch_dog.states[worker].current_loss = loss

        # Collect loss and parameters for indivual worker
        errors.append(loss)
        weights.append(parameters)

        return errors, weights

    def _check_responses(self, n_responses):
        """Check if expected responses need to changed based on heartbeats

        Args:
            n_responses: No. of expected responses

        Returns:
            n_responses: Updated no. of responses
        """
        if n_responses > len(self.heartbeater.hearts):
            self._logger.info(
                f"Changed no of hearts from {n_responses} to {self.heartbeater.hearts}"
                )
            n_responses = len(self.heartbeater.hearts)
        return n_responses


    def _poll(self, i, errors, weights, workers_received):
        """Poll for events from worker
        """
        events = dict(self.poller.poll())
        if (self.pull_socket in events) and \
            (events.get(self.pull_socket) == zmq.POLLIN):
            msg = self.pull_socket.recv_multipart()
            cmd = msg[0]
            worker = msg[1]
            content = msg[2]
            if cmd == b"WORK":
                errors, weights = self._gather(
                    worker,
                    content,
                    errors,
                    weights
                )

                i += 1
                workers_received.append(worker)

        return i, errors, weights, workers_received

    def _expected_responses(self):
        """Returns expected no. of responses
        """
        n_responses = len(self.heartbeater.hearts)

        if self.config.comms.mode == 1 or \
            self.config.comms.mode == 2:
            comm_intervals = np.array(
                [worker.comm_interval for worker in self.watch_dog.states]
            )
            comm_every_iter = np.array(
                [worker.comm_every_iter for worker in self.watch_dog.states]
            )

            identities = np.array(
                [worker.identity for worker in self.watch_dog.states]
            )

            self._logger.debug(f"Comm intervals={comm_intervals}")

            who_comms = np.mod(self.iter, comm_intervals)
            who_comms = set(np.argwhere(who_comms == 0).flatten())
            every_iter = set(np.argwhere(self.iter >= comm_every_iter).flatten())
            # every_iter = set(np.argwhere(self.model.iter <= comm_every_iter).flatten())
            both = who_comms.union(every_iter)
            identities = identities[list(both)]
            n_responses = len(both)
            self._logger.debug(
                f"i={self.iter}, who={who_comms}, every_iter={every_iter}, "
                f"b={both}, responses={n_responses}, \nidentities={identities}, "
                f"intervals for responses={comm_intervals[list(who_comms)]}"
            )

        return n_responses

    def _recv(self):
        """Reduce params from workers
        """
        if self.state == MAP_PARAMS:
            self._logger.info("Recv work")
            # Recv work
            i = 0
            errors = []
            weights = []
            workers_received = []
            
            n_responses = self._expected_responses()

            self._logger.debug(f"Expected responses={n_responses}")

            while i < n_responses:
                # Need to sleep gevent to be able to have heartbeat thread
                gevent.sleep(0.00000001)

                # Poll messages from workers and collect them
                i, errors, weights, workers_received = \
                    self._poll(i, errors, weights, workers_received)

                # Update no. of expected responses to end the while loop
                n_responses = self._check_responses(n_responses)

            # Determine dynamic communication scheme
            if (self.config.comms.mode == 2) and (self.iter != 0):
                if workers_received:
                    self._calculate_dynamic_comms_loss(workers_received)

            # Keep track of communication rounds for each worker
            for worker in workers_received:
                self.watch_dog.states[worker].comm_rounds += 1

            # Aggregate parameters
            parameters, epoch_loss = self._reduce(
                weights=weights,
                errors=errors,
                model=self.model,
                samples=None,
                n_samples=None,
                workers_received=workers_received,
                mode=self.config.distribute.aggregate_mode
            )

            # Update model with these parameters
            if parameters:
                self.model.set_weights(parameters)
                # delta = self._update_model(parameters)

            # Check metrics
            train_acc, test_acc, test_loss = self._check_metrics()

            self._logger.info(
                f"iteration = {self.iter}, "
                f"Loss = {epoch_loss:7.4f}, test_loss={test_loss:7.4f}, "
                f"train acc={train_acc*100:7.4f}%, test acc={test_acc*100:7.4f}%"
            )

            self.iter += 1

    def setup(self, train_dataset: Tuple, test_dataset: Optional[Tuple]=None):
        """Setup master with train and test data and train steps
        """
        self.train_dataset = train_dataset # Numpy tuple
        self.test_dataset = test_dataset # Numpy tuple

        X_test, y_test = self.test_dataset
        self.test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        global_batch_size = self.config.model.batch_size * self.strategy.n_workers
        self.test_dataset = self.test_dataset.batch(global_batch_size)

        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy'
        )
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy'
        )

    def heart_loop(self):
        """Heart loop
        """
        self._logger.info("Starting heart beater")
        while self.state != COMPLETE:
            # Send beat
            self.state, newhearts, heartfailures = \
                self.heartbeater.beat(
                    self.heart_pub_socket,
                    self.state,
                    self.strategy.n_workers
                )
            if newhearts:
                list(map(self.watch_dog.add_worker, newhearts))
            if heartfailures:
                list(map(self.watch_dog.pop, heartfailures))
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

    def train_loop(self):
        """Machine learning training loop
        """
        try:
            # Setup data and send to clients
            start = time.time()
            self._logger.debug(f"epochs={self.n_iterations}")
            while self.iter < self.n_iterations:
                # Need to have a small sleep to enable gevent threading
                gevent.sleep(0.00000001)
                # self._logger.info(f'Iteration={self.iter}')

                # Send data or params
                self._map()

                # Aggregate params
                self._recv()

            # self._n_mbs = self._calculate_packet_size()

            self.done()
            end = time.time()
            elapsed = end - start
            elapsed -= self.svd_time
            self._logger.info(
                "Time taken for %d iterations is %7.6fs",
                self.n_iterations,
                elapsed
            )
            self._logger.info(f'Total packet size communicated={self._n_mbs:.3f}MB')
        except KeyboardInterrupt:
            self._logger.info("Keyboard quit")
            self.done()
        except zmq.ZMQError:
            self._logger.info("ZMQError")
            self.done()
        except Exception as e:
            self._logger.exception(e)
            self.done()
        finally:
            self._logger.info("Exiting peacefully")
            # self.kill()

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
