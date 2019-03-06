import time
import zmq.green as zmq
import numpy as np
import logging
import click
import gevent
import signal
import binascii

# Local
from fault_tolerant_ml.utils import zhelpers
from fault_tolerant_ml.utils import setup_logger
from fault_tolerant_ml.data import DummyData, OccupancyData
from fault_tolerant_ml.ml import hypotheses
from fault_tolerant_ml.ml.metrics import test_hypothesis, accuracy
from fault_tolerant_ml.core import WorkerState, WorkerStates

START       = 0
MAP         = 1
REMAP       = 2
DIST_PARAMS = 3
REDUCE      = 4 
COMPLETE    = 5

class Master(object):
    """Master class for distributed machine learning system
    """
    def __init__(self, n_iterations, learning_rate, verbose, scenario):
        
        # ZMQ variables
        self.ctrl_socket = None
        self.publisher = None
        self.context   = zmq.Context()

        self.ws = WorkerStates()
        self.workers = set()
        self.worker_states = {}
        self.mapping = {}
        self.state = START

        # Model variables
        self.n_iterations = n_iterations
        self.alpha = learning_rate
        self.hypothesis = hypotheses.log_hypothesis

        self.scenario = scenario
        self.times = []

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

        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.setsockopt_string(zmq.IDENTITY, 'MASTER')
        self.router_socket.bind("tcp://*:5564")

        self.ctrl_socket = self.context.socket(zmq.ROUTER)
        self.ctrl_socket.setsockopt_string(zmq.IDENTITY, 'MASTER')
        self.ctrl_socket.bind("tcp://*:5565")
    
    def setup_poller(self):
        poller = zmq.Poller()
        poller.register(self.pull_socket, zmq.POLLIN | zmq.POLLERR)
        poller.register(self.ctrl_socket, zmq.POLLIN | zmq.POLLERR)
        poller.register(self.router_socket, zmq.POLLIN | zmq.POLLERR)
        poller.register(self.publisher, zmq.POLLOUT | zmq.POLLERR)

        return poller

    def send_heartbeat(self):
        for worker in self.workers:
            self.logger.debug('PING')
            self.worker_states[worker] = False
            self.ctrl_socket.send_multipart([worker, b'HEARTBEAT'])

    def heartbeat_loop(self):
        while self.state != COMPLETE:
            self.send_heartbeat()
            gevent.sleep(0.5)

    def distribute_data(self):
        """Distributes the data to the workers
        """
        # Distribute data/data indices to work on
        self.logger.debug("Distributing data")
        batch_size = int(np.ceil(self.data.n_samples / self.ws.n_alive()))
        batch_gen = self.data.next_batch(self.X_train, self.y_train, batch_size)

        # Encode to bytes
        n_samples = str(self.data.n_samples).encode()
        n_features = str(self.data.n_features).encode()
        n_classes = str(self.data.n_classes).encode()
        scenario = str(self.scenario).encode()

        # Iterate through workers and send
        i = 0
        for worker in self.ws:

            if worker.state:
                worker.mr_idxs_used = False
                # Get next batch to send
                X_batch, y_batch = next(batch_gen)
                self.logger.debug(f"X.shape={X_batch.shape}, y.shape={y_batch.shape}")
                batch_data = np.hstack((X_batch, y_batch))

                # Encode data
                dtype = batch_data.dtype.str.encode()
                shape = str(batch_data.shape).encode()
                msg = batch_data.tostring()

                # Keep track of samples per worker
                worker.n_samples = X_batch.shape[0]
                lower_bound = X_batch.shape[0] * i
                upper_bound = lower_bound + X_batch.shape[0]
                worker.idxs = np.arange(lower_bound, upper_bound)
                if worker.most_representative is None:
                    worker.most_representative = np.zeros((10,))

                self.ctrl_socket.send_multipart([worker.identity, b"WORK", batch_data, dtype, shape])
                self.ctrl_socket.send_multipart([worker.identity, b"WORK", n_samples, n_features, n_classes, scenario])
                i += 1

        self.logger.debug(f"Worker ranges={[(np.min(w.idxs), np.max(w.idxs)) for w in self.ws]}")

    def register_workers(self):
        """Registers workers in a round robin fashion
        """
        worker_id = self.pull_socket.recv()

        if worker_id not in self.ws:
            self.logger.info(f"Worker Registered: {worker_id}")
            self.ws.add_worker(worker_id)
        else:
            self.logger.debug("Worker asking for work again?")

    def start_next_task(self):
        """Starts new task depending on the state of the system.

        Possible states range from mapping of data, remapping of data (if worker dies or another worker is added),
        or distributing parameters.
        """
        if self.state == START:

            self.state = MAP

        if self.state == MAP:
            self.distribute_data()
            self.state = DIST_PARAMS

        if self.state == REMAP:

            self.logger.debug(f"Redistributing data")
            # Stack all indices from current dataset that we will use to remap
            global_idxs = np.hstack([w.idxs for w in self.ws if not w.mr_idxs_used])
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

            for w in self.ws:
                if not w.mr_idxs_used:
                    w.mr_idxs_used = True

            self.distribute_data()
            self.state = DIST_PARAMS

        if self.state == DIST_PARAMS:
            # self.send_heartbeat()
            self.logger.debug("Distributing parameters")
            self.times.append(time.time())
            if self.scenario == 0:
                
                # Get message send ready
                msg = self.theta.tostring()
                dtype = self.theta.dtype.str.encode()
                shape = str(self.theta.shape).encode()

                self.publisher.send_multipart([b"", b"WORK", msg, dtype, shape])
            elif self.scenario == 1:
                
                # Quantize parameters
                theta_q = Master.get_quantized_params(self.theta)
                # Get message send ready
                msg = theta_q.tostring()

                self.publisher.send_multipart([b"", b"WORK", msg])

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
        n_alive_workers = self.ws.n_alive()
        self.logger.debug(f"Alive workers={n_alive_workers}")

        i = 0
        timeout = 1 # We give 1 seconds to poll worker if state changed since poll event
        running_time = 0

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
                    self.logger.debug("Got msg in the bag")
                except zmq.ZMQError as e:
                    if e.errno == zmq.EAGAIN:
                        # state changed since poll event
                        running_time += time.time() - start_i
                        if running_time > timeout:
                            self.logger.debug(f"Running time exceeded timeout={running_time}")
                            active_workers = set([w.identity for w in self.ws if w.state])
                            # Get workers that we did not receive work from
                            diff = active_workers - workers_received
                            for w in diff:
                                # Set dead workers state to false and replace their worker idxs with their
                                # most representative samples
                                self.ws[w].state = False
                                self.ws[w].idxs = self.ws[w].most_representative
                            
                            self.state = REMAP
                            break

                        continue

                self.logger.debug(f"Alive workers={n_alive_workers}")
                if i == 0:
                    worker, d_theta_temp, epoch_loss_temp, mr = msg
                else:
                    # Receive multipart including command message
                    cmd, worker, d_theta_temp, epoch_loss_temp, mr = msg

                # Calculate weighting
                samples_for_worker = self.ws[worker].n_samples
                beta = (samples_for_worker / self.data.n_samples)

                # Decode gradient matrix
                d_theta_temp = np.frombuffer(d_theta_temp, dtype=np.float64)
                d_theta_temp = d_theta_temp.reshape(self.theta.shape)

                # Store most representative points
                mr = np.frombuffer(mr, dtype=np.int)
                # Determine current index - we will map this back to the global index if worker dies
                self.ws[worker].most_representative = np.min(self.ws[worker].idxs) + mr

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
        d_theta /= i
        epoch_loss /= i

        self.logger.debug("Calculated gradients")
        
        return d_theta, epoch_loss

    def detect_workers(self):
        """Detects workers by polling whoever has sent through the CONNECT command along with their worker ids
        """
        while True:
            events = dict(self.poller.poll())

            if (self.pull_socket in events) and (events.get(self.pull_socket) == zmq.POLLIN):
                command = self.pull_socket.recv(flags=zmq.SNDMORE)

                if command == b"CONNECT":
                    self.register_workers()
            else:
                break

        self.logger.debug(f"Signed up all workers = {self.ws}")

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

            completed = False
            i = 0
            delta = 1.0
            start = time.time()

            while not completed:

                events = dict(self.poller.poll())

                if len(self.ws) > 0:
                    if (self.publisher in events) and (events.get(self.publisher) == zmq.POLLOUT):
                        self.start_next_task()

                    # Check heartbeat
                    if (self.ctrl_socket in events) and (events.get(self.ctrl_socket) == zmq.POLLIN):
                        address, msg = self.ctrl_socket.recv_multipart()
                        self.worker_states[address] = True
                        self.logger.debug(f"Address={address.decode()}, Msg={msg.decode()}")

                    if (self.pull_socket in events) and (events.get(self.pull_socket) == zmq.POLLIN):
                        # Don't use the results if they've already been counted
                        command = self.pull_socket.recv(flags=zmq.SNDMORE)

                        if command == b"CONNECT":
                            self.register_workers()

                        elif command == b"WORK":
                            theta_p = self.theta.copy()
                            # Receive updated parameters from workers
                            d_theta, epoch_loss = self.get_gradients(events)

                            # Update the global parameters with weighted error
                            for k in np.arange(self.data.n_classes):
                                self.theta[:, k] = self.theta[:, k] - self.alpha * d_theta[:, k]

                            delta = np.max(np.abs(theta_p - self.theta))

                            if self.state != REMAP:
                                self.state = DIST_PARAMS
                            self.logger.debug(f"iteration = {i}, delta = {delta:7.4f}, Loss = {epoch_loss:7.4f}")
                            i += 1
                else:
                    if (self.pull_socket in events) and (events.get(self.pull_socket) == zmq.POLLIN):
                        # Don't use the results if they've already been counted
                        command = self.pull_socket.recv(flags=zmq.SNDMORE)

                        if command == b"CONNECT":
                            self.register_workers()

                if i > self.n_iterations:
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
        self.poller.unregister(self.router_socket)
        self.poller.unregister(self.publisher)
        self.pull_socket.close()
        self.publisher.close()
        self.router_socket.close()
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
@click.option('--scenario', '-s', default=0, type=int)
def run(n_iterations, learning_rate, verbose, scenario):
    """Controller function which creates the master and starts off the training

    Args:
        n_iterations (int): No. of iterations we perform for training
        learning_rate (float): The rate at which we want our model to learn
        verbose (int): The logger level as an integer. See more in the logging file for different options
        scenario (int): The scenario we would like to run
    """
    master = Master(
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        verbose=verbose,
        scenario=scenario
    )
    master.connect()
    time.sleep(1)
    master.start()

if __name__ == "__main__":
    run()