import time
import zmq
import numpy as np
import logging
import click

# Local
from fault_tolerant_ml.utils import zhelpers
from fault_tolerant_ml.utils import setup_logger
from fault_tolerant_ml.data import DummyData, OccupancyData
from fault_tolerant_ml.ml import hypotheses
from fault_tolerant_ml.ml.metrics import test_hypothesis, accuracy

START       = 0
MAP         = 1
DIST_PARAMS = 2
REDUCE      = 3 

class Master(object):
    """Master class for distributed machine learning system
    """
    def __init__(self, n_iterations, learning_rate, verbose, scenario):
        
        # ZMQ variables
        self.ctrl_socket = None
        self.publisher = None
        self.receiver = None

        self.workers = set()
        self.state = START

        # Model variables
        self.n_iterations = n_iterations
        self.alpha = learning_rate
        self.hypothesis = hypotheses.log_hypothesis

        self.samples_per_worker = {}
        self.most_representative = {}
        self.worker_idxs = {}
        self.scenario = scenario
        self.times = []

        # Setup logger
        # self.logger = logging.getLogger("masters")
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
        self.context   = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind("tcp://*:5563")

        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind("tcp://*:5562")

        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.bind("tcp://*:5564")

        self.ctrl_socket = self.context.socket(zmq.ROUTER)
        self.ctrl_socket.setsockopt_string(zmq.IDENTITY, 'MASTER')
        self.ctrl_socket.bind("tcp://*:5565")

    def distribute_data(self):
        """Distributes the data to the workers
        """
        # Distribute data/data indices to work on
        self.logger.debug("Distributing data")
        batch_size = int(np.ceil(self.data.n_samples / len(self.workers)))
        batch_gen = self.data.next_batch(self.data.X_train, self.data.y_train, batch_size)

        # Encode to bytes
        n_samples = str(self.data.n_samples).encode()
        n_features = str(self.data.n_features).encode()
        n_classes = str(self.data.n_classes).encode()
        scenario = str(self.scenario).encode()

        # Iterate through workers and send
        for i, worker in enumerate(self.workers):

            # Get next batch to send
            X_batch, y_batch = next(batch_gen)
            self.logger.debug(f"X.shape={X_batch.shape}, y.shape={y_batch.shape}")
            batch_data = np.hstack((X_batch, y_batch))

            # Encode data
            dtype = batch_data.dtype.str.encode()
            shape = str(batch_data.shape).encode()
            msg = batch_data.tostring()

            # Keep track of samples per worker
            self.samples_per_worker[worker] = X_batch.shape[0]
            lower_bound = X_batch.shape[0] * i
            upper_bound = lower_bound + X_batch.shape[0]
            self.worker_idxs[worker] = np.arange(lower_bound, upper_bound)
            self.most_representative[worker] = np.zeros((100,))

            self.ctrl_socket.send_multipart([worker, batch_data, dtype, shape])
            self.ctrl_socket.send_multipart([worker, n_samples, n_features, n_classes, scenario])

        self.logger.debug(f"Worker ranges={[(np.min(idxs), np.max(idxs)) for idxs in self.worker_idxs.values()]}")

    def register_workers(self):
        """Registers workers in a round robin fashion
        """
        worker_id = self.pull_socket.recv()

        if worker_id not in self.workers:
            self.logger.info(f"Worker Registered: {worker_id}")
            self.workers.add(worker_id)
            # self.ctrl_socket.send_multipart([worker_id, ])
        else:
            self.logger.debug("Worker asking for work again?")

    def start_next_task(self):
        """Starts new task depending on the state of the system
        """
        if self.state == START:

            self.state = MAP

        if self.state == MAP:
            self.distribute_data()
            self.state = DIST_PARAMS

        if self.state == DIST_PARAMS:
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

    def get_gradients(self):
        """Receives gradients from workers
        """
        d_theta = np.zeros(self.theta.shape)
        epoch_loss = 0.0

        for i in np.arange(len(self.workers)):

            # samples_for_worker = self.samples_per_worker[worker]
            # beta = (samples_for_worker / self.data.n_samples)
            # beta = 1.0

            # Since we received the command then we only receive the gradients, and epoch loss
            # for the first worker that we are receiving information from. 
            # TODO: Need to correctly weight each worker with the amount of work they have done. 
            # Cannot use a push pull socket
            if i == 0:
                worker, d_theta, epoch_loss, mr = self.pull_socket.recv_multipart()
                samples_for_worker = self.samples_per_worker[worker]
                beta = (samples_for_worker / self.data.n_samples)
                d_theta = np.frombuffer(d_theta, dtype=np.float64)
                d_theta = d_theta.reshape(self.theta.shape)
                d_theta = d_theta.copy()
                self.most_representative[worker] = np.frombuffer(mr, dtype=np.int)

                epoch_loss = float(epoch_loss.decode())

                d_theta += beta * d_theta               
                epoch_loss += beta * epoch_loss
            else:

                # Receive multipart including command message
                cmd, worker, d_theta_temp, loss, mr = self.pull_socket.recv_multipart()
                samples_for_worker = self.samples_per_worker[worker]
                beta = (samples_for_worker / self.data.n_samples)
                d_theta_temp = np.frombuffer(d_theta_temp, dtype=np.float64)
                d_theta_temp = d_theta_temp.reshape(self.theta.shape)
                self.most_representative[worker] = np.frombuffer(mr, dtype=np.int)

                loss = float(loss.decode())
                
                d_theta += beta * d_theta_temp               
                epoch_loss += beta * loss

        # Average parameters
        d_theta /= len(self.workers)
        epoch_loss /= len(self.workers)
        
        return d_theta, epoch_loss

    def start(self):
        """Starts work of master. First connects to workers and then performs machine learning training
        """

        # For reproducibility
        np.random.seed(42)

        self.data = OccupancyData(filepath="/c/Users/nb304836/Documents/git-repos/large_scale_ml/data/occupancy_data/datatraining.txt", n_stacks=100)
        self.data.transform()
        
        self.logger.info(f"Initialized dummy data of size {self.data}")

        self.theta = np.random.randn(self.data.n_features, self.data.n_classes).astype(np.float32)
        self.logger.debug(f"Init theta={self.theta}")
        
        poller = zmq.Poller()

        poller.register(self.pull_socket, zmq.POLLIN)
        # poller.register(self.ctrl_socket, zmq.POLLIN)
        poller.register(self.push_socket, zmq.POLLOUT)
        poller.register(self.publisher, zmq.POLLOUT)
        
        try:

            # Detect all workers by polling by whoevers sending their worker ids
            while True:
                events = dict(poller.poll())

                self.logger.debug(f"events={events}")
                if events.get(self.pull_socket) == zmq.POLLIN:
                    command = self.pull_socket.recv(flags=zmq.SNDMORE)

                    if command == b"CONNECT":
                        self.register_workers()
                else:
                    break

            self.logger.debug("Signed up all workers")

            completed = False
            i = 0
            delta = 1.0
            start = time.time()

            while not completed:

                events = dict(poller.poll())

                if len(self.workers) > 0:
                    if events.get(self.push_socket) == zmq.POLLOUT:
                        self.start_next_task()

                    if events.get(self.pull_socket) == zmq.POLLIN:
                        # Don't use the results if they've already been counted
                        command = self. pull_socket.recv(flags=zmq.SNDMORE)

                        if command == b"CONNECT":
                            self.register_workers()

                        elif command == b"WORK":
                            
                            theta_p = self.theta.copy()
                            # Receive updated parameters from workers
                            d_theta, epoch_loss = self.get_gradients()

                            # Update the global parameters with weighted error
                            for k in np.arange(self.data.n_classes):
                                self.theta[:, k] = self.theta[:, k] - self.alpha * d_theta[:, k]

                            delta = np.max(np.abs(theta_p - self.theta))

                            self.state = DIST_PARAMS
                            self.logger.debug(f"iteration = {i}, delta = {delta:7.4f}, Loss = {epoch_loss:7.4f}")
                            i += 1
                else:
                    if events.get(self.pull_socket) == zmq.POLLIN:
                        # Don't use the results if they've already been counted
                        command = self.pull_socket.recv(flags=zmq.SNDMORE)

                        if command == b"CONNECT":
                            self.register_workers()

                if i > self.n_iterations:
                    completed = True

            # Tell workers to exit
            self.done()
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
            poller.unregister(self.pull_socket)
            poller.unregister(self.push_socket)
            self.kill()

    def kill(self):
        """Kills sockets
        """
        self.publisher.close()
        self.push_socket.close()
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