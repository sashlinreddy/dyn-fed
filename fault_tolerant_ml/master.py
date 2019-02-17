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

    def __init__(self, n_iterations, learning_rate, verbose):
        
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

        # Setup logger
        # self.logger = logging.getLogger("masters")
        self.logger = setup_logger(level=verbose)

    def connect(self):
        
        # Prepare our context and publisher
        self.context   = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind("tcp://*:5563")

        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind("tcp://*:5562")

        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.bind("tcp://*:5564")

        self.ctrl_socket = self.context.socket(zmq.ROUTER)
        self.ctrl_socket.bind("tcp://*:5565")

    def distribute_data(self):
        # Distribute data/data indices to work on

        # batch_size = int(np.ceil(data.n_samples / n_workers))
        # for X_batch, y_batch in data.next_batch(batch_size):

        #     data = X_batch.tostring()
        #     self.logger.info("Sending data")
        #     self.push_socket.send(data)

        self.logger.debug("Distributing data")
        batch_size = int(np.ceil(self.data.n_samples / len(self.workers)))
        batch_gen = self.data.next_batch(self.data.X_train, self.data.y_train, batch_size)

        ns_enc = str(self.data.n_samples).encode()
        samp_feat_d = dict(
            n_samples=self.data.n_samples,
            n_features=self.data.n_features,
            n_classes=self.data.n_classes
        )

        for worker in self.workers:

            X_batch, y_batch = next(batch_gen)
            self.logger.debug(f"X.shape={X_batch.shape}, y.shape={y_batch.shape}")
            batch_data = np.hstack((X_batch, y_batch))
            msg = batch_data.tostring()
            self.samples_per_worker[worker] = X_batch.shape[0]
            
            self.ctrl_socket.send(worker, zmq.SNDMORE)
            zhelpers.send_array(self.ctrl_socket, batch_data)
            self.ctrl_socket.send(worker, zmq.SNDMORE)
            self.ctrl_socket.send_json(samp_feat_d)
            # self.ctrl_socket.send_multipart([worker, msg])

    def register_workers(self):
        
        worker_id = self.pull_socket.recv()

        if worker_id not in self.workers:
            self.logger.info(f"Worker Registered: {worker_id}")
            self.workers.add(worker_id)
            # self.ctrl_socket.send_multipart([worker_id, ])
        else:
            self.logger.debug("Worker asking for work again?")

    def start_next_task(self):

        if self.state == START:

            self.state = MAP

        if self.state == MAP:
            self.distribute_data()
            self.state = DIST_PARAMS

        if self.state == DIST_PARAMS:
            # msg = self.theta.tostring()
            # self.logger.debug("Distributing parameters")
            # self.logger.debug(f"Distributing params, theta = {self.theta}")
            # self.publisher.send_multipart([b"", msg])
            zhelpers.send_array(self.publisher, self.theta)
            self.state = REDUCE

    def get_gradients(self):

        d_theta = np.zeros(self.theta.shape)
        epoch_loss = 0.0

        for i, worker in enumerate(self.workers):

            samples_for_worker = self.samples_per_worker[worker]
            # beta = (samples_for_worker / self.data.n_samples)
            beta = 1.0

            # Since we received the command then we only receive the gradients, and epoch loss
            # for the first worker that we are receiving information from. 
            # TODO: Need to correctly weight each worker with the amount of work they have done. 
            # Cannot use a push pull socket
            if i == 0:
                d_theta, epoch_loss = self.pull_socket.recv_multipart()
                d_theta = np.frombuffer(d_theta, dtype=np.float64)
                d_theta = d_theta.reshape(self.theta.shape)
                d_theta = d_theta.copy()

                epoch_loss = float(epoch_loss.decode())

                d_theta += beta * d_theta               
                epoch_loss += beta * epoch_loss
            else:

                # Receive multipart including command message
                cmd, d_theta_temp, loss = self.pull_socket.recv_multipart()
                d_theta_temp = np.frombuffer(d_theta_temp, dtype=np.float64)
                d_theta_temp = d_theta_temp.reshape(self.theta.shape)

                loss = float(loss.decode())
                
                d_theta += beta * d_theta_temp               
                epoch_loss += beta * loss

        # Average parameters
        d_theta /= len(self.workers)
        epoch_loss /= len(self.workers)
        
        return d_theta, epoch_loss

    def start(self):

        # n_samples = 100
        # self.data = DummyData(n_samples=n_samples, n_features=10, n_classes=1)
        # self.data.transform()

        # For reproducibility
        np.random.seed(42)

        self.data = OccupancyData(filepath="/c/Users/nb304836/Documents/git-repos/large_scale_ml/data/occupancy_data/datatraining.txt", n_stacks=100)
        self.data.transform()
        
        self.logger.info(f"Initialized dummy data of size {self.data}")

        self.theta = np.random.randn(self.data.n_features, self.data.n_classes)
        self.logger.debug(f"Init theta={self.theta}")
        
        poller = zmq.Poller()

        poller.register(self.pull_socket, zmq.POLLIN)
        poller.register(self.push_socket, zmq.POLLOUT)
        poller.register(self.publisher, zmq.POLLOUT)
        
        try:

            n_start_subs = 5

            # TODO: Detect workers properly without hardcoded number of workers
            for i in range(n_start_subs):
                # Don't use the results if they've already been counted
                command = self.pull_socket.recv(flags=zmq.SNDMORE)

                if command == b"CONNECT":
                    self.register_workers()

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
                        command = self.pull_socket.recv(flags=zmq.SNDMORE)

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
        self.publisher.close()
        self.push_socket.close()
        self.ctrl_socket.close()
        # self.context.term()

    def done(self):
        time.sleep(1)
        # self.publisher.send_multipart([b"B", b"EXIT"])
        msg = {"EXIT" : 1}
        self.publisher.send_json(msg)


@click.command()
@click.option('--n_iterations', '-i', default=400, type=int)
@click.option('--learning_rate', '-lr', default=0.1, type=float)
@click.option('--verbose', '-v', default=10, type=int)
def run(n_iterations, learning_rate, verbose):
    master = Master(
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        verbose=verbose
    )
    master.connect()
    time.sleep(1)
    master.start()

if __name__ == "__main__":
    run()