"""Master with heartbeater
"""
import logging
import signal
import time
import socket
import os
import json

import gevent
import numpy as np
import zmq.green as zmq

from fault_tolerant_ml.data.utils import next_batch
from fault_tolerant_ml.distribute import WatchDog
from fault_tolerant_ml.distribute.heartbeater import Heartbeater
from fault_tolerant_ml.distribute.states import (COMPLETE, MAP, MAP_PARAMS,
                                                 START)
from fault_tolerant_ml.metrics import accuracy_scorev2
from fault_tolerant_ml.proto.utils import (params_to_string,
                                           parse_params_response_from_string,
                                           parse_setup_response_from_string,
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
        self.n_iterations = int(
            np.ceil(self.model.max_iter / self.strategy.comm_period)
        )
        self.X = None
        self.y = None
        self.X_valid = None
        self.y_valid = None

        # Get ipaddress for workers to connect to

        self.heartbeater = Heartbeater(self.n_workers, period)
        self.watch_dog = WatchDog()

        self._logger = logging.getLogger(f"ftml.distribute.{self.__class__.__name__}")

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
        with open(os.path.join(self.strategy.config_folder, ip_filename), "w") as f:
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

    def _poll_svd(self, i):
        """Poll for the SVD index from workers
        """
        # Poll messages from workers and collect them
        events = dict(self.poller.poll())
        if (self.pull_socket in events) and \
            (events.get(self.pull_socket) == zmq.POLLIN):
            # Receive svd info
            msg = self.pull_socket.recv_multipart()
            cmd = msg[0]
            if cmd == b"SVD":
                worker = msg[1]
                content = msg[2]
                svd_idx = parse_setup_response_from_string(content)
                self._logger.info(
                    f"SVD_idx for worker {worker} is {svd_idx}"
                )
                self.watch_dog.states[worker].svd_idx = svd_idx
                i += 1
        return i

    def _calculate_dynamic_comms(self):
        """Calculate dynamic comm period
        """
        i = 0
        n_responses = len(self.heartbeater.hearts)
        while i < n_responses:
            # Need to sleep gevent to be able to have heartbeat thread
            gevent.sleep(0.00000001)

            i = self._poll_svd(i)

            # Update no. of expected responses to end the while loop
            n_responses = self._check_responses(n_responses)

        # Normalize svd idx
        svds = np.array([state.svd_idx for state in self.watch_dog.states])
        max_svd = np.max(svds)
        normalized_svds = svds / max_svd
        comm_iterations = np.floor(normalized_svds * self.n_iterations).astype(int)

        for i, worker in enumerate(self.watch_dog.states):
            worker.comm_iterations = comm_iterations[i]

        self._logger.debug(f"Comm iterations={comm_iterations}")


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

            self._logger.debug(f"Workerstates={self.watch_dog.states}")

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

            self._calculate_dynamic_comms()

        if self.state == MAP_PARAMS:
            # Map params
            msg = [params_to_string(self.model.layers)]
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
        y_pred = self.model.forward(self.X_valid)
        y_train_pred = self.model.forward(self.X)
        
        train_acc = accuracy_scorev2(self.y.data, y_train_pred.data)
        test_acc = accuracy_scorev2(self.y_valid.data, y_pred.data)

        return train_acc, test_acc

    def _update_model(self, parameters):
        """Update model given new parameters

        Args:
            parameters (list): List of numpy matrices for each layer
        """
        deltas = []
        for i in np.arange(self.model.n_layers):
            deltas.append(
                np.max(
                    np.abs(
                        self.model.layers[i].W.data - parameters[i][0]
                    )
                )
            )
            self.model.layers[i].W.data = parameters[i][0]
            self.model.layers[i].b.data = parameters[i][1]

        delta = np.max(deltas)
        return delta

    def _reduce(self,
                d_Wbs,
                errors,
                model,
                samples,
                n_samples,
                mode=0,
                epsilon=1e-8):
        """Aggregate received parameters

        Args:
            d_Wbs (list): List of parameters received
            errors (list): List of losses for each worker
            model (ftml.Model): Model being used
            samples (list): List containing number of samples for each worker
            n_samples (int): Total number of samples
            mode (int): Aggregation mode (default: 0)

        Returns:
            parameters (list): List of numpy matrices for each layer
            epoch_loss (float): Aggregated epoch loss
        """
        # pylint: disable=too-many-arguments, too-many-locals
        # Iterate through workers and weight parameters by corresponding epoch loss
        parameters = [[np.zeros_like(l.W.data), np.zeros_like(l.b.data)] for l in model.layers]
        sum_es = np.sum(np.exp(errors))
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
            self._logger.debug(f"worker={j}, weight={weight}, loss={errors[j]}")
            for k in np.arange(model.n_layers):
                parameters[k][0] += (
                    d_Wbs[j][k][0] * weight
                    if weight > 0
                    else d_Wbs[j][k][0] * epsilon
                ) # For W parameter
                parameters[k][1] += (
                    d_Wbs[j][k][1] * weight
                    if weight > 0
                    else d_Wbs[j][k][1] * epsilon
                ) # For b parameter

        return parameters, epoch_loss

    def _gather(self, worker, content, errors, d_Wbs):
        """Gather parameters from worker
        """
        parameters, mr, loss = \
            parse_params_response_from_string(content)

        self._logger.info(
            f"Received work from {worker}, mr.shape={mr.shape}"
        )

        # Collect loss and parameters
        errors.append(loss)
        d_Wbs.append(parameters)

        return errors, d_Wbs

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

    def _poll(self, i, errors, d_Wbs):
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
                errors, d_Wbs = self._gather(
                    worker,
                    content,
                    errors,
                    d_Wbs
                )

                i += 1

        return i, errors, d_Wbs

    def _recv(self):
        """Reduce params from workers
        """
        if self.state == MAP_PARAMS:
            self._logger.info("Recv work")
            # Recv work
            i = 0
            errors = []
            d_Wbs = []
            n_responses = len(self.heartbeater.hearts)
            while i < n_responses:
                # Need to sleep gevent to be able to have heartbeat thread
                gevent.sleep(0.00000001)

                # Poll messages from workers and collect them
                i, errors, d_Wbs = self._poll(i, errors, d_Wbs)

                # Update no. of expected responses to end the while loop
                n_responses = self._check_responses(n_responses)

            # Aggregate parameters
            parameters, epoch_loss = self._reduce(
                d_Wbs,
                errors,
                self.model,
                None,
                None,
                self.strategy.aggregate_mode
            )

            # Update model with these parameters
            delta = self._update_model(parameters)

            # Check metrics
            train_acc, test_acc = self._check_metrics()

            self._logger.info(
                f"iteration = {self.model.iter}, delta = {delta:7.4f}, "
                f"Loss = {epoch_loss:7.4f}, train acc={train_acc*100:7.4f}%, "
                f"test acc={test_acc*100:7.4f}%"
            )

            self.model.iter += 1


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
            self.state, newhearts, heartfailures = \
                self.heartbeater.beat(self.heart_pub_socket, self.state)
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
            start = time.time()
            while self.model.iter < self.n_iterations:
                # Need to have a small sleep to enable gevent threading
                gevent.sleep(0.00000001)
                
                # Send data or params
                self._map()

                # Aggregate params
                self._recv()

            self.done()
            end = time.time()
            elapsed = end - start
            self._logger.info(
                "Time taken for %d iterations is %7.6fs",
                self.n_iterations,
                elapsed
            )
        except KeyboardInterrupt:
            self._logger.info("Keyboard quit")
            self.done()
        except zmq.ZMQError:
            self._logger.info("ZMQError")
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
