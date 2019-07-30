"""Contains all master logic for fault tolerant ml. 

Any master devices should run the master logic.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import zmq.green as zmq
import numpy as np
import logging
import click
import gevent
import signal
import os
import socket
import json

# Local
from fault_tolerant_ml.utils import zhelpers
from fault_tolerant_ml.utils import setup_logger
from fault_tolerant_ml.metrics import accuracy_scorev2
from fault_tolerant_ml.utils.maths import linspace_quantization
from fault_tolerant_ml.tools import TFLogger
from fault_tolerant_ml.distribute import WatchDog
from fault_tolerant_ml.distribute import Distributor
from fault_tolerant_ml.distribute.wrappers import ftml_train, ftml_train_collect, ftml_trainv2
from fault_tolerant_ml.distribute.states import *
from fault_tolerant_ml.operators import Tensor

class Master(object):
    """Master class for distributed machine learning system
    """
    def __init__(self, model, verbose):
        
        # ZMQ variables
        self.ctrl_socket = None
        self.publisher = None
        self.context   = zmq.Context()

        self.mapping = {}

        self.watch_dog = WatchDog()
        self.distributor = Distributor()

        # Distributed environ variables
        self.state = START
        self.model = model
        self.strategy = self.model.strategy
        self.delay_change = False

        # Get ipaddress for workers to connect to
        self.hostname = socket.gethostname()
        self.ip_address = socket.gethostbyname(self.hostname)

        # Model variables
        self.n_iterations = int(np.ceil(self.model.max_iter / self.strategy.comm_period))
        self.optimizer = self.model.optimizer

        # Tracking variables
        self.times = []
        self._tf_logger = None
        if "TFDIR" in os.environ:
            logdir = os.path.join(os.environ["TFDIR"], f"tf/{self.model.encode_name}/master")
            self._tf_logger = TFLogger(logdir)

        # Setup logger
        self.logger = logging.getLogger(f"ftml.distribute.{self.__class__.__name__}")

        data_dir = self.strategy.shared_folder
        self.logger.info(f"Master on ip={self.ip_address}")

        ip_filename = "ip_config.json"
        if "SLURM_JOBID" in os.environ:
            slurm_job_id = os.environ["SLURM_JOBID"]
            ip_filename = f"ip_config_{slurm_job_id}.json"

        ip_config = {"ipAddress" : self.ip_address}
        with open(os.path.join(data_dir, ip_filename), "w") as f:
            json.dump(ip_config, f)

    @ftml_train_collect
    def _train_iteration(self, events):
        W_p = self.model.W.copy()
        # Receive updated parameters from workers
        # d_W, epoch_loss = self.gather(events, timeout=10)
        params = {
            "watch_dog": self.watch_dog,
            "strategy": self.strategy,
            "state": self.state,
            "n_samples": self.data.n_samples,
            "timeout": 10,
            "quantize": self.strategy.quantize,
            # "W": self.model.W
            "W": self.model.W.data
        }
        parameters, epoch_loss = self.distributor.collect(
            events=events, 
            socket=self.pull_socket,
            params=params
        )

        if self.strategy.send_gradients:
            # Update the global parameters with weighted error
            self.model.W = \
            self.optimizer.minimize(
                X=self.X_train, 
                y=None, 
                y_pred=None, 
                W=self.model.W, 
                precomputed_gradients=parameters
            )
        else:
            self.logger.info(f"parameters.dtype={parameters.dtype}")
            self.model.W.data = parameters
            self.logger.info(f"type(self.model.W.data)={self.model.W.data.dtype}")

        y_pred = self.model.predict(self.data.X_test)
        y_train_pred = self.model.predict(self.data.X_train)
        # train_acc = accuracy_scorev2(self.data.y_train, y_train_pred)
        # test_acc = accuracy_scorev2(self.data.y_test, y_pred)
        train_acc = accuracy_scorev2(self.data.y_train.data, y_train_pred.data)
        test_acc = accuracy_scorev2(self.data.y_test.data, y_pred.data)

        if self._tf_logger is not None:
            self._tf_logger.histogram("W-master", self.model.W.data, self.model.iter, bins=self.n_iterations)
            self._tf_logger.scalar("loss-master", epoch_loss, self.model.iter)
            self._tf_logger.scalar("train-accuracy-master", train_acc, self.model.iter)
            self._tf_logger.scalar("test-accuracy-master", test_acc, self.model.iter)
            grad_l2_norm = np.linalg.norm(parameters)
            self._tf_logger.scalar("gradnorm-master", grad_l2_norm, self.model.iter)

        # delta = np.max(np.abs(W_p - self.model.W))
        delta = np.max(np.abs(W_p.data - self.model.W.data))

        # self.logger.info(f"iteration = {self.strategy.model.iter}, delta = {delta:7.4f}, Loss = {epoch_loss:7.4f}")
        self.logger.info(f"iteration = {self.model.iter}, delta = {delta:7.4f}, Loss = {epoch_loss:7.4f}, train acc={train_acc*100:7.4f}%, test acc={test_acc*100:7.4f}%")

        return delta
        
    @ftml_trainv2
    def _train(self, events):
        """Function that kicks off distribution strategy

        Args:
            events (dict): Dictionary of zmq.events to know what and when messages are received
        """
        # Map tasks
        self.map()

        # Gather and apply gradients
        self._train_iteration(events)

    @ftml_train
    def _training_loop(self):
        """Not being used at the moment
        """
        delta = 1.0
        start = time.time()

        while self.model.iter < self.n_iterations:

            # Poll events
            events = dict(self.poller.poll())

            # If we have more than 1 worker
            if len(self.watch_dog.states) > 0:
                # Map tasks
                self.map()

                # Gather and apply gradients
                self._train_iteration(events)

            else:
                if (self.pull_socket in events) and (events.get(self.pull_socket) == zmq.POLLIN):
                    # Don't use the results if they've already been counted
                    command = self.pull_socket.recv(flags=zmq.SNDMORE)

                    if command == b"CONNECT":
                        self.register_workers()

        # Tell workers to exit
        self.done()
        self.state = COMPLETE
        end = time.time()
        self.logger.info("Time taken for %d iterations is %7.6fs" % (self.n_iterations, end-start))

    def connect(self):
        """Connects to necessary sockets
        """
        # Prepare our context and publisher
        self.publisher = self.context.socket(zmq.PUB) # pylint: disable=no-member
        self.publisher.bind("tcp://*:5563")

        self.pull_socket = self.context.socket(zmq.PULL) # pylint: disable=no-member
        self.pull_socket.bind("tcp://*:5562")

        self.ctrl_socket = self.context.socket(zmq.ROUTER) # pylint: disable=no-member
        self.ctrl_socket.setsockopt_string(zmq.IDENTITY, 'MASTER') # pylint: disable=no-member
        self.ctrl_socket.bind("tcp://*:5565")
    
    def setup_poller(self):
        poller = zmq.Poller()
        poller.register(self.pull_socket, zmq.POLLIN | zmq.POLLERR)
        poller.register(self.ctrl_socket, zmq.POLLIN | zmq.POLLERR)
        poller.register(self.publisher, zmq.POLLOUT | zmq.POLLERR)

        return poller

    def send_heartbeat(self):
        for worker in self.watch_dog.states:
            self.logger.debug('PING')
            worker.state = False
            self.ctrl_socket.send_multipart([worker.identity, b'HEARTBEAT'])

    def heartbeat_loop(self):
        while self.state != COMPLETE:
            self.send_heartbeat()
            gevent.sleep(0.5)

    def register_workers(self, worker_id=None):
        """Registers workers in a round robin fashion
        """
        if not worker_id:
            worker_id = self.pull_socket.recv()

        self.watch_dog.add_worker(worker_id)

    def detect_workers(self):
        """Detects workers by polling whoever has sent through the CONNECT command along with their worker ids
        """
        timeout = self.strategy.worker_timeout # 10 second time out
        start = time.time()

        while True:
            events = dict(self.poller.poll())

            if (self.pull_socket in events) and (events.get(self.pull_socket) == zmq.POLLIN):
                command = self.pull_socket.recv(flags=zmq.SNDMORE)

                if command == b"CONNECT":
                    self.register_workers()
                    # start = time.time()
            else:
        
                end = time.time()
                # if round(end - start, 2) % 1 == 0:
                #     self.logger.debug(end-start)
                if end-start > timeout:
                    self.logger.info(f"{timeout} second timeout - no more workers found")
                    break

        self.logger.info(f"Signed up all {len(self.watch_dog.states)} workers")
        self.logger.debug(f"Signed up all workers = {self.watch_dog.states}")

    def set_params(self):
        """Prepares parameters to be sent by the distributor
        """
        params = {}
        params["state"] = self.state
        params["scenario"] = self.strategy.scenario
        params["remap"] = self.strategy.remap
        params["quantize"] = self.strategy.quantize

        if self.state == DIST_PARAMS:
            params["delay_change"] = self.delay_change
        else:
            params["n_alive"] = self.watch_dog.n_alive
            params["n_workers"] = self.strategy.n_workers
            params["n_samples"] = self.data.n_samples
            params["n_features"] = self.data.n_features
            params["n_classes"] = self.data.n_classes
            params["n_most_rep"] = self.optimizer.n_most_rep
            params["send_gradients"] = self.strategy.send_gradients
            params["comm_period"] = self.strategy.comm_period
            params["mapping"] = self.mapping
        return params

    def map(self):
        """Starts new task depending on the state of the system.

        Possible states range from mapping of data, remapping of data (if worker dies or another worker is added),
        or distributing parameters.
        """
        if self.state == START:

            self.state = MAP

        if self.state == MAP:
            data = (self.X_train, self.y_train)
            params = self.set_params()

            self.distributor.map(
                socket=self.ctrl_socket, 
                data=data, 
                workers=self.watch_dog.states, 
                params=params, 
                gen_func=self.data.next_batch
            )
            self.state = DIST_PARAMS

            # # Plot class balances
            # if "FIGDIR" in os.environ:

            #     import pandas as pd
            #     from fault_tolerant_ml.viz.target import ClassBalance

            #     figdir = os.environ["FIGDIR"]

            #     worker_ids = list(self.watch_dog.states.keys())
            #     fname = os.path.join(figdir, f"class-balance.png")
            #     class_bal = [v[1] for (k, v) in self.distributor.labels_per_worker.items()]
            #     class_names = self.data.class_names

            #     class_balance = ClassBalance(labels=worker_ids, legend=class_names, fname=fname, stacked=True, percentage=True)
            #     class_balance.fit(y=class_bal)
            #     class_balance.poof()

        if self.state == REMAP:

            self.logger.debug(f"Redistributing data")
            if self.strategy.remap == 1:
                
                # Remap only data for workers that went down in previous iteration
                # Get indices for dead workers
                if self.mapping:
                    dead_worker = [w for w in self.watch_dog.states if not w.mr_idxs_used and not w.state]
                    if dead_worker:
                        dead_worker = dead_worker[0]
                    else:
                        return

                    remap_idxs = np.hstack([[w.mapping.get(i) for i in w.most_representative] for w in self.watch_dog.states if not w.mr_idxs_used and not w.state])
                    worker_ids_down = [w.identity for w in self.watch_dog.states if not w.mr_idxs_used and not w.state]
                    self.logger.debug(f"remapping idxs={remap_idxs}, worker_ids={worker_ids_down}")
                    self.logger.debug(f"Dead worker={len(dead_worker.mapping.keys())}")
                    
                    self.logger.debug(f"Remap idxs={remap_idxs.shape}")
                else:
                    remap_idxs = np.hstack([w.most_representative for w in self.watch_dog.states if not w.mr_idxs_used and not w.state])

                n_samples = remap_idxs.shape[0]
                new_range = np.arange(n_samples)

                self.logger.debug(f"N samples = {n_samples}")

                self.mapping = dict(zip(new_range, remap_idxs))

                self.logger.debug(f"Mapping={self.mapping}")

                X_train, y_train = self.data.X_train[remap_idxs], self.data.y_train[remap_idxs]

                for w in self.watch_dog.states:
                    if not w.mr_idxs_used:
                        w.mr_idxs_used = True

                data =(X_train, y_train)
                params = self.set_params()
                params["n_samples"] = n_samples

            else:
                # Stack all indices from current dataset that we will use to remap
                global_idxs = np.hstack([w.idxs for w in self.watch_dog.states if (not w.mr_idxs_used) and (not w.idxs is None)])
                new_range = np.arange(global_idxs.shape[0])
                self.logger.debug(f"new data idxs shape={global_idxs.shape}")

                self.logger.debug(f"Min={global_idxs.min()}, Max={global_idxs.max()}")
                
                # If we have had a failure before we need to just keep track of the global indices
                if self.mapping:
                    # The new dataset will be smaller than the original dataset. We still would like to only 
                    # use indices of the original dataset to simplify things. This recalculates those indices
                    global_idxs = np.array([self.mapping.get(i) for i in global_idxs])

                self.logger.debug(f"Min={global_idxs.min()}, Max={global_idxs.max()}")
                self.logger.debug("Updating mapping")
                self.mapping = dict(zip(new_range, global_idxs))

                self.X_train, self.y_train = self.data.update_xy(global_idxs)

                for w in self.watch_dog.states:
                    if not w.mr_idxs_used:
                        w.mr_idxs_used = True

                data = (self.X_train, self.y_train)
                params = self.set_params()

            self.distributor.map(
                    socket=self.ctrl_socket, 
                    data=data, 
                    workers=self.watch_dog.states, 
                    params=params, 
                    gen_func=self.data.next_batch
                )

            self.state = DIST_PARAMS

        if self.state == DIST_PARAMS:
            # self.send_heartbeat()
            self.times.append(time.time())

            data = self.model.W.data if self.strategy.quantize != 1 else linspace_quantization(self.model.W.data, interval=200)
            workers = None
            params = self.set_params()

            self.distributor.map(
                socket=self.publisher, 
                data=data, 
                workers=workers, 
                params=params
            )

            self.state = REDUCE

    def print_metrics(self):
        """Print metrics relating to communication times
        """
        # Print avg loop iteration time
        diff = np.diff(self.times)
        self.logger.debug(f"Times={diff.mean():7.6f}s")

    def plot_metrics(self):

        if "FIGDIR" in os.environ:

            import pandas as pd
            from fault_tolerant_ml.viz.target import ClassBalance

            figdir = os.path.join(os.environ["FIGDIR"], self.model.encode_name)
            if not os.path.exists(figdir):
                os.mkdir(figdir)

            try:
                self.logger.debug("Saving class balances distribution plot...")
                worker_ids = [s.identity.decode() for s in self.watch_dog.states if s.state]
                fname = os.path.join(figdir, f"mnist-class-balance.png")
                class_bal = [v[1] for (k, v) in self.distributor.labels_per_worker.items() if k.identity.decode() in worker_ids]
                class_names = self.data.class_names

                class_balance = ClassBalance(labels=worker_ids, legend=class_names, fname=fname, stacked=True, percentage=True)
                class_balance.fit(y=class_bal)
                class_balance.poof()

                fig = class_balance.fig

                if self._tf_logger is not None:
                    self._tf_logger.images("class-bal-master", [fig], self.model.iter)
            except Exception as e:
                self.logger.exception(e)

    def main_loop(self):
        """Main loop for training.

        First detects workers who are willing to do work. Then distributes the data accordingly. Then we perform 
        gradient descent iteratively. We parallelize the gradient calculation and calculate a weighted average
        gradient matrix. This weighted average is calculated as the number of samples that a worker received as a 
        fraction of the total number of samples in the entire dataset.
        """
        # For reproducibility
        np.random.seed(42)
        
        self.logger.info(f"Initialized dummy data of size {self.data}")

        # self.model.W = \
        # np.random.randn(self.data.n_features, self.data.n_classes).astype(self.data.X_train.dtype) * 0.01
        self.model.W = \
        Tensor(np.random.randn(self.data.n_features, self.data.n_classes).astype(self.data.X_train.dtype) * 0.01, is_param=True)
        self.logger.debug(f"Init W={self.model.W}")
        
        self.poller = self.setup_poller()

        # self.training_loop()
        self._train() # pylint: disable=no-value-for-parameter
        # self.train_iter()

        self.print_metrics()
        
    def train(self, data):
        """Starts work of master. First connects to workers and then performs machine learning training
        """
        self.data = data
        self.X_train, self.y_train = self.data.X_train, self.data.y_train
        gevent.signal(signal.SIGQUIT, gevent.kill)

        main_loop = gevent.spawn(self.main_loop)
        # heartbeat_loop = gevent.spawn(self.heartbeat_loop)
        
        gevent.joinall([
            main_loop, 
            # heartbeat_loop,
        ])

        self.kill()

    def kill(self):
        """Kills sockets
        """
        self.poller.unregister(self.pull_socket)
        self.poller.unregister(self.publisher)
        self.pull_socket.close()
        self.publisher.close()
        self.ctrl_socket.close()
        # self.context.term()

    def done(self):
        """Sends exit signal to workers
        """
        time.sleep(1)
        self.publisher.send_multipart([b"", b"EXIT"])