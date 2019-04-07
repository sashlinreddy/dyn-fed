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
# from dotenv import find_dotenv, load_dotenv

# Local
from fault_tolerant_ml.utils import zhelpers
from fault_tolerant_ml.utils import setup_logger
from fault_tolerant_ml.data import DummyData, OccupancyData
from fault_tolerant_ml.ml.optimizer import ParallelSGDOptimizer
from fault_tolerant_ml.ml import hypotheses
from fault_tolerant_ml.ml.metrics import test_hypothesis, accuracy
from fault_tolerant_ml.tools import TFLogger
from fault_tolerant_ml.distribute import WatchDog
from fault_tolerant_ml.distribute.distributor import Distributor
from fault_tolerant_ml.distribute.states import *

class Master(object):
    """Master class for distributed machine learning system
    """
    def __init__(self, n_iterations, learning_rate, verbose, scenario, n_most_representative,
    delay, switch_delta):
        
        # ZMQ variables
        self.ctrl_socket = None
        self.publisher = None
        self.context   = zmq.Context()

        self.mapping = {}

        self.watch_dog = WatchDog()
        self.distributor = Distributor()

        # Distributed environ variables
        self.state = START
        self.comm_period = delay
        self.switch_delta = switch_delta
        self.delay_change = False
        self.scenario = scenario
        self.n_most_representative = n_most_representative

        # Model variables
        self.n_iterations = int(np.ceil(n_iterations / self.comm_period))
        self.learning_rate = learning_rate
        self.hypothesis = hypotheses.log_hypothesis
        self.optimizer = ParallelSGDOptimizer(learning_rate=self.learning_rate)

        # Tracking variables
        self.times = []
        self.tf_logger = None
        if "LOGDIR" in os.environ:
            logdir = os.path.join(os.environ["LOGDIR"], "tf/master")
            self.tf_logger = TFLogger(logdir)

        # Setup logger
        self.logger = setup_logger(level=verbose)

    @staticmethod
    def get_quantized_params(theta):
        """Quantizes parameters

        Arguments:
            theta (np.ndarray): Parameter matrix to be quantized

        Returns:
            msg (np.ndarray): Structured numpy array that is quantized

        """
        min_theta_val = theta.min() + 1e-8
        max_theta_val = theta.max() + 1e-8
        interval = 8
        bins = np.linspace(min_theta_val, max_theta_val, interval)
        theta_bins = np.digitize(theta, bins).astype(np.int8)

        struct_field_names = ["min_val", "max_val", "interval", "bins"]
        struct_field_types = [np.float32, np.float32, np.int32, 'b']
        struct_field_shapes = [1, 1, 1, (theta.shape)]

        msg = np.zeros(1, dtype=(list(zip(struct_field_names, struct_field_types, struct_field_shapes))))
        msg[0] = (min_theta_val, max_theta_val, interval, theta_bins)

        return msg

    def connect(self):
        """Connects to necessary sockets
        """
        # Prepare our context and publisher
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind("tcp://*:5563")

        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind("tcp://*:5562")

        self.ctrl_socket = self.context.socket(zmq.ROUTER)
        self.ctrl_socket.setsockopt_string(zmq.IDENTITY, 'MASTER')
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

    def set_params(self):
        """Prepares parameters to be sent by the distributor
        """
        params = {}
        params["state"] = self.state
        params["scenario"] = self.scenario

        if self.state == DIST_PARAMS:
            params["delay_change"] = self.delay_change
        else:
            params["n_alive"] = self.watch_dog.n_alive
            params["n_samples"] = self.data.n_samples
            params["n_features"] = self.data.n_features
            params["n_classes"] = self.data.n_classes
            params["n_most_representative"] = self.n_most_representative
            params["learning_rate"] = self.learning_rate
            params["comm_period"] = self.comm_period
            params["mapping"] = self.mapping
        return params

    def register_workers(self, worker_id=None):
        """Registers workers in a round robin fashion
        """
        if not worker_id:
            worker_id = self.pull_socket.recv()

        self.watch_dog.add_worker(worker_id)

    def start_next_task(self):
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
                gen_func=OccupancyData.next_batch
            )
            self.state = DIST_PARAMS

        if self.state == REMAP:

            self.logger.debug(f"Redistributing data")
            if self.scenario == 2:
                
                # Remap only data for workers that went down in previous iteration
                # Get indices for dead workers
                if self.mapping:
                    dead_worker = [w for w in self.watch_dog.states if not w.mr_idxs_used and not w.state][0]
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

                self.distributor.map(
                    socket=self.ctrl_socket, 
                    data=data, 
                    workers=self.watch_dog.states, 
                    params=params, 
                    gen_func=OccupancyData.next_batch
                )

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
                    gen_func=OccupancyData.next_batch
                )
            self.state = DIST_PARAMS

        if self.state == DIST_PARAMS:
            # self.send_heartbeat()
            self.times.append(time.time())

            data = self.theta if self.scenario != 1 else Master.get_quantized_params(self.theta)
            workers = None
            params = self.set_params()

            self.distributor.map(
                socket=self.publisher, 
                data=data, 
                workers=workers, 
                params=params,
                gen_func=OccupancyData.next_batch
            )

            self.state = REDUCE

    def get_gradients(self, events):
        """Receives gradients from workers

        Args:
            events (dict): Dictionary of events from our poller

        Returns:
            d_theta (numpy.ndarray): Our gradient matrix that is aggregated with a weighting according to the number    of samples each worker has
            epoch_loss (float): The loss for this epoch aggregated from each worker, also weighted according to the     work each worker did
        """
        d_theta = np.zeros(self.theta.shape)
        epoch_loss = 0.0

        self.logger.debug(f"Receiving gradients")
        n_alive_workers = self.watch_dog.n_alive
        self.logger.debug(f"Alive workers={n_alive_workers}")

        i = 0
        timeout = 1 # We give 1 seconds to poll worker if state changed since poll event
        running_time = 0
        n_connected = 0

        workers_received = set()

        while True:

            # Stop condition
            if i >= n_alive_workers:
                break

            # Timer to calculate running time for an iteration. We can then calculate the running time for 
            # an iteration so that if a state changes since a poll event, we can break if the running time 
            # exceeds the timeout
            start_i = time.time()

            if (self.pull_socket in events) and (events.get(self.pull_socket) == zmq.POLLIN):
                try:
                    msg = self.pull_socket.recv_multipart(zmq.NOBLOCK)
                except zmq.ZMQError as e:
                    if e.errno == zmq.EAGAIN:
                        # state changed since poll event
                        running_time += time.time() - start_i
                        if running_time > timeout:
                            self.logger.debug(f"Running time exceeded timeout={running_time}")
                            active_workers = set(self.watch_dog.active_workers)
                            # Get workers that we did not receive work from
                            diff = active_workers - workers_received
                            for w in diff:
                                # Set dead workers state to false
                                self.watch_dog.states[w].state = False
                                if self.scenario != 2:                                    
                                    self.watch_dog.states[w].idxs = self.watch_dog.states[w].most_representative
                            
                            self.state = REMAP
                            break

                        continue

                self.logger.debug(f"Alive workers={n_alive_workers}")
                if i == 0:
                    worker, d_theta_temp, epoch_loss_temp, mr = msg
                else:
                    # Receive multipart including command message
                    cmd = msg[0]
                    if cmd == b"WORK":
                        worker, d_theta_temp, epoch_loss_temp, mr = msg[1:]
                    elif cmd == b"CONNECT":
                        self.register_workers(msg[1])
                        n_connected += 1
                        i += 1
                        continue

                # Calculate weighting
                samples_for_worker = self.watch_dog.states[worker].n_samples
                beta = (samples_for_worker / self.data.n_samples)

                # Decode gradient matrix
                d_theta_temp = np.frombuffer(d_theta_temp, dtype=np.float64)
                d_theta_temp = d_theta_temp.reshape(self.theta.shape)

                # Store most representative points
                mr = np.frombuffer(mr, dtype=np.int)
                # Determine current index - we will map this back to the global index if worker dies
                if self.scenario == 2:
                    self.watch_dog.states[worker].most_representative = self.watch_dog.states[worker].lower_bound + mr
                    self.logger.debug(f"Min mr={np.min(self.watch_dog.states[worker].most_representative)}, Max mr={np.max(self.watch_dog.states[worker].most_representative)}")
                else:
                    self.watch_dog.states[worker].most_representative = np.min(self.watch_dog.states[worker].idxs) + mr
                    

                # Decode loss
                epoch_loss_temp = float(epoch_loss_temp.decode())

                # Weight parameters and loss
                d_theta += beta * d_theta_temp              
                epoch_loss += beta * epoch_loss_temp

                workers_received.add(worker)

                i += 1
                running_time = 0

        # Average parameters
        # d_theta /= len(self.workers)
        # epoch_loss /= len(self.workers)
        # self.logger.debug(f"Len worker={len(self.workers)}, i-1={i-1}")
        assert i > 0
        assert i > 0
        i -= n_connected
        d_theta /= i
        epoch_loss /= i

        self.logger.debug("Calculated gradients")
        
        return d_theta, epoch_loss

    def detect_workers(self):
        """Detects workers by polling whoever has sent through the CONNECT command along with their worker ids
        """
        # timeout = 10 # 10 second time out
        # start = time.time()

        while True:
            events = dict(self.poller.poll())

            if (self.pull_socket in events) and (events.get(self.pull_socket) == zmq.POLLIN):
                command = self.pull_socket.recv(flags=zmq.SNDMORE)

                if command == b"CONNECT":
                    self.register_workers()
                    # start = time.time()
            else:
                break
        
            # end = time.time()
            # if round(end - start, 2) % 1 == 0:
            #     self.logger.debug(end-start)
            # if end-start > timeout:
            #     break

        self.logger.debug(f"Signed up all workers = {self.watch_dog.states}")

    def main_loop(self):
        """Main loop for training.

        First detects workers who are willing to do work. Then distributes the data accordingly. Then we perform 
        gradient descent iteratively. We parallelize the gradient calculation and calculate a weighted average
        gradient matrix. This weighted average is calculated as the number of samples that a worker received as a 
        fraction of the total number of samples in the entire dataset.
        """
        # For reproducibility
        np.random.seed(42)

        self.data = OccupancyData(filepath="/c/Users/nb304836/Documents/git-repos/large_scale_ml/data/occupancy_data/datatraining.txt", n_stacks=100)
        self.data.transform()
        self.X_train, self.y_train = self.data.X_train, self.data.y_train
        
        self.logger.info(f"Initialized dummy data of size {self.data}")

        self.theta = np.random.randn(self.data.n_features, self.data.n_classes).astype(np.float32)
        self.logger.debug(f"Init theta={self.theta}")
        
        self.poller = self.setup_poller()
        
        try:

            # Detect all workers by polling by whoevers sending their worker ids
            self.detect_workers()
            if not self.watch_dog.states:
                self.logger.info("No workers found")
                raise KeyboardInterrupt

            completed = False
            i = 0
            delta = 1.0
            start = time.time()

            while not completed:

                events = dict(self.poller.poll())

                if len(self.watch_dog.states) > 0:
                    if (self.publisher in events) and (events.get(self.publisher) == zmq.POLLOUT):
                        self.start_next_task()

                    # Check heartbeat
                    if (self.ctrl_socket in events) and (events.get(self.ctrl_socket) == zmq.POLLIN):
                        address, msg = self.ctrl_socket.recv_multipart()
                        self.watch_dog.states[address].state = True
                        self.logger.debug(f"Address={address.decode()}, Msg={msg.decode()}")

                    if (self.pull_socket in events) and (events.get(self.pull_socket) == zmq.POLLIN):
                        # Don't use the results if they've already been counted
                        command = self.pull_socket.recv(flags=zmq.SNDMORE)

                        if command == b"CONNECT":
                            self.register_workers()
                            self.state = MAP

                        elif command == b"WORK":
                            theta_p = self.theta.copy()
                            # Receive updated parameters from workers
                            d_theta, epoch_loss = self.get_gradients(events)

                            # Update the global parameters with weighted error
                            self.theta = self.optimizer.minimize(X=None, y=None, y_pred=None, theta=self.theta, d_theta=d_theta)

                            if self.tf_logger is not None:
                                self.tf_logger.histogram("theta-master", self.theta, i, bins=self.n_iterations)
                                self.tf_logger.scalar("epoch-master", epoch_loss, i)

                            delta = np.max(np.abs(theta_p - self.theta))

                            if self.state != REMAP:
                                self.state = DIST_PARAMS
                            self.logger.info(f"iteration = {i}, delta = {delta:7.4f}, Loss = {epoch_loss:7.4f}")
                            i += 1
                            if delta < self.switch_delta and self.comm_period > 1 and not self.delay_change:
                                self.delay_change = True
                                self.n_iterations = i + (self.n_iterations - i) * self.comm_period
                                self.logger.debug(f"Iterations now = {self.n_iterations}")
                else:
                    if (self.pull_socket in events) and (events.get(self.pull_socket) == zmq.POLLIN):
                        # Don't use the results if they've already been counted
                        command = self.pull_socket.recv(flags=zmq.SNDMORE)

                        if command == b"CONNECT":
                            self.register_workers()

                if i >= self.n_iterations:
                    completed = True

            # Tell workers to exit
            self.done()
            self.state = COMPLETE
            end = time.time()
            self.logger.info("Time taken for %d iterations is %7.6fs" % (self.n_iterations, end-start))

            diff = np.diff(self.times)
            self.logger.debug(f"Times={diff.mean():7.6f}s")

            # Print confusion matrix
            confusion_matrix = test_hypothesis(self.data.X_test, self.data.y_test, self.theta)
            self.logger.info(f"Confusion matrix=\n{confusion_matrix}")

            # Accuracy
            acc = accuracy(self.data.X_test, self.data.y_test, self.theta, self.hypothesis)
            self.logger.info(f"Accuracy={acc * 100:7.4f}%")
            
        except KeyboardInterrupt as e:
            pass
        except zmq.ZMQError as zmq_err:
            self.logger.error(zmq_err)
            self.done()
        finally:
            self.logger.info("Exiting peacefully. Cleaning up...")
            # self.kill()

    def start(self):
        """Starts work of master. First connects to workers and then performs machine learning training
        """
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

@click.command()
@click.option('--n_iterations', '-i', default=400, type=int)
@click.option('--learning_rate', '-lr', default=0.1, type=float)
@click.option('--verbose', '-v', default=10, type=int)
@click.option('--scenario', '-s', default=1, type=int)
@click.option('--n_most_representative', '-nmr', default=100, type=int)
@click.option('--delay', '-d', default=1, type=int)
@click.option('--switch_delta', '-sd', default=0.0074, type=float)
def run(n_iterations, learning_rate, verbose, scenario, n_most_representative, delay, 
switch_delta):
    """Controller function which creates the master and starts off the training

    Args:
        n_iterations (int): No. of iterations we perform for training
        learning_rate (float): The rate at which we want our model to learn
        verbose (int): The logger level as an integer. See more in the logging file for different options
        scenario (int): The scenario we would like to run
    """

    # # load_dotenv(find_dotenv())

    if "LOGDIR" in os.environ:
        from fault_tolerant_ml.lib.io.file_io import flush_dir
        ignore_dir = [os.path.join(os.environ["LOGDIR"], "tf/")]
        # ignore_dir = []
        flush_dir(os.environ["LOGDIR"], ignore_dir=ignore_dir)

    master = Master(
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        verbose=verbose,
        scenario=scenario,
        n_most_representative=n_most_representative,
        delay=delay,
        switch_delta=switch_delta
    )
    master.connect()
    time.sleep(1)
    master.start()

if __name__ == "__main__":
    run()