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
from dyn_fed.proto.utils import (params_to_string,
                                 parse_params_response_from_string,
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
        print(f"epochs type={type(self.n_iterations)}")
        self.train_dataset = None
        self.test_dataset = None
        self.train_step = None
        self.test_step = None

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

            # self.state = MAP_PARAMS
            self.iter += 1

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
            msg = [params_to_string(self.model.layers)]
            
            multipart = [b"", b"WORK_PARAMS"]
            multipart.extend(msg)

            self._logger.info("Sending params")
            self.pub_socket.send_multipart(multipart)

    def setup(self, train_dataset: Tuple, train_step: Callable,
              test_dataset: Optional[Tuple]=None, test_step: Optional[Callable]=None):
        """Setup master with train and test data and train steps
        """
        self.train_dataset = train_dataset # Numpy tuple
        self.test_dataset = test_dataset # Numpy tuple
        self.train_step = train_step # Callable
        self.test_step = test_step

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
                # self._recv()

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
