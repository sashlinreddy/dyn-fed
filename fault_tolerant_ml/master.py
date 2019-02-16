import time
import zmq
import numpy as np
import logging
from fault_tolerant_ml.utils import setup_logger
from fault_tolerant_ml.data import DummyData

START       = 0
MAP         = 1
DIST_PARAMS = 2
REDUCE      = 3 

class Master(object):

    def __init__(self):
        
        self.ctrl_socket = None
        self.publisher = None
        self.receiver = None

        self.workers = set()
        self.state = START

        # Setup logger
        self.logger = logging.getLogger("masters")

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
        batch_size = int(np.ceil(self.dummy_data.n_samples / len(self.workers)))
        batch_gen = self.dummy_data.next_batch(batch_size)
        for worker in self.workers:

            X_batch, y_batch = next(batch_gen)
            msg = X_batch.tostring()
            self.ctrl_socket.send_multipart([worker, msg])

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
            msg = self.theta.tostring()
            self.logger.debug("Distributing parameters")
            self.publisher.send_multipart([b"", msg])
            self.state = REDUCE

    def get_gradients(self):

        d_theta = self.pull_socket.recv()
        d_theta = np.frombuffer(d_theta, dtype=np.float64)
        d_theta = d_theta.reshape(self.theta.shape)
        d_theta = d_theta.copy()

        for j in range(len(self.workers) - 1):

            cmd, d_theta_temp = self.pull_socket.recv_multipart()
            d_theta_temp = np.frombuffer(d_theta_temp, dtype=np.float64)
            d_theta_temp = d_theta_temp.reshape(self.theta.shape)
            d_theta += d_theta_temp

        # Average parameters
        d_theta /= len(self.workers)
        
        return d_theta

    def start(self):
        n_samples = 100
        n_workers = 3

        self.dummy_data = DummyData(n_samples=n_samples, n_features=10, n_classes=1)
        self.dummy_data.transform()
        self.logger.info(f"Initialized dummy data of size {self.dummy_data}")

        self.theta = np.random.randn(self.dummy_data.n_features, self.dummy_data.n_classes)
        
        poller = zmq.Poller()

        poller.register(self.pull_socket, zmq.POLLIN)
        poller.register(self.push_socket, zmq.POLLOUT)
        poller.register(self.publisher, zmq.POLLOUT)
        
        try:

            n_start_subs = 3

            # TODO: Detect workers
            for i in range(n_start_subs):
                # Don't use the results if they've already been counted
                command = self.pull_socket.recv(flags=zmq.SNDMORE)

                if command == b"CONNECT":
                    self.register_workers()

            completed = False
            i = 0
            n_iterations = 10

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
                            
                            # Receive updated parameters from workers
                            d_theta = self.get_gradients()
                            i += 1
                            self.state = DIST_PARAMS
                            self.logger.debug("Update parameters")
                else:
                    if events.get(self.pull_socket) == zmq.POLLIN:
                        # Don't use the results if they've already been counted
                        command = self.pull_socket.recv(flags=zmq.SNDMORE)

                        if command == b"CONNECT":
                            self.register_workers()

                if i > n_iterations:
                    completed = True

            # Tell workers to exit
            self.done()
            
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
        self.publisher.send_multipart([b"B", b"EXIT"])

if __name__ == "__main__":
    logger = setup_logger()
    master = Master()
    master.connect()
    time.sleep(1)
    master.start()