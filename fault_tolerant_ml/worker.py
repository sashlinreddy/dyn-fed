import zmq.green as zmq
import time
import numpy as np
import uuid
import logging
import click

# Local
from fault_tolerant_ml.utils import setup_logger
from fault_tolerant_ml.utils import zhelpers
from fault_tolerant_ml.ml import hypotheses, loss_fns

class Worker(object):

    def __init__(self, verbose):

        self.worker_id = str(uuid.uuid4())
        self.subscriber = None
        self.starter = None
        # self.logger = logging.getLogger("masters")
        self.logger = setup_logger(filename=f'log-{self.worker_id}.log', level=verbose)
        self.have_work = False
        self.hypothesis = hypotheses.log_hypothesis
        self.gradient = loss_fns.cross_entropy_gradient

    def connect(self):
        # Prepare our context and publisher
        self.context    = zmq.Context()
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect("tcp://localhost:5563")
        self.subscriber.setsockopt(zmq.SUBSCRIBE, b"")

        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.connect("tcp://localhost:5562")

        self.router_socket = self.context.socket(zmq.DEALER)
        self.router_socket.setsockopt_string(zmq.IDENTITY, self.worker_id)
        self.router_socket.connect("tcp://localhost:5564")

        self.ctrl_socket = self.context.socket(zmq.DEALER)
        self.ctrl_socket.setsockopt_string(zmq.IDENTITY, self.worker_id)
        self.ctrl_socket.connect("tcp://localhost:5565")

    def setup_poller(self):
        poller = zmq.Poller()
        poller.register(self.subscriber, zmq.POLLIN | zmq.POLLERR)
        poller.register(self.router_socket, zmq.POLLIN | zmq.POLLERR)
        poller.register(self.ctrl_socket, zmq.POLLIN | zmq.POLLERR)

        return poller

    def do_work(self, X, y, theta):

        self.logger.info('Doing work')
        # Get predictions
        h = self.hypothesis(X, theta)
        # Calculate error/residuals
        e = (h - y)

        # Calculate log loss
        log_loss = -y * np.log(h) - (1 - y) * np.log(1 - h)

        # TODO: Calculate most representative data points
        # We regard data points that have a high loss to be most representative
        most_representative = np.argsort(-log_loss.flatten())[0:100]

        # Calculate processor loss - this is aggregated
        batch_loss = np.mean(log_loss)

        n_features, n_classes = theta.shape
        d_theta = np.zeros((n_features, n_classes))

        # To be moved to optimizer
        for k in np.arange(n_classes):
            d_theta[:, k] = self.gradient(X, e[:, np.newaxis, k])

        return d_theta, batch_loss, most_representative

    def start(self):

        poller = self.setup_poller()
        # poller.register(self.push_socket, zmq.POLLOUT)

        self.logger.info('Started Worker %s' % self.worker_id)

        try:
            start = time.time()
            i = 0
            self.scenario = 0
            n_samples = 0
            n_features = 0
            n_classes = 0

            # self.starter.send_multipart([b"READY", self.worker_id.encode()])
            # self.ctrl_socket.send(b"READY")

            while True:

                if self.have_work:

                    events = dict(poller.poll())

                    if (self.ctrl_socket in events) and (events.get(self.ctrl_socket) == zmq.POLLIN):
                        command = self.ctrl_socket.recv()
                        self.logger.debug(f"Command={command}")
                        if command == b"HEARTBEAT":
                            self.ctrl_socket.send(b"PONG")
                            self.logger.debug("PONG")

                    if (self.subscriber in events) and (events.get(self.subscriber) == zmq.POLLIN):
                        # Read envelope with address
                        self.logger.debug("Receiving contents")
                        contents = self.subscriber.recv_multipart()
                        address = contents[0]
                        cmd = contents[1]
                        msg = contents[2:]
                        packet_size = np.sum([len(m) for m in contents])

                        self.logger.debug(f"Packet size={packet_size} bytes")

                        if cmd == b"EXIT":
                            self.logger.info("Received EXIT command")
                            break
                        else:

                            if self.scenario == 0:

                                # Receive parameter matrix on the subscriber socket
                                data, dtype, shape = msg
                                shape = shape.decode()

                                # Reconstruct numpy array
                                buf = memoryview(data)
                                theta = np.frombuffer(buf, dtype=dtype)
                                theta = theta.reshape(eval(shape))
                                self.logger.info(f"theta.shape{theta.shape}")
                                
                                theta = theta.copy()
                            elif self.scenario == 1:

                                # Receive numpy struct array
                                # msg = self.subscriber.recv(flags=flags, copy=True, track=False)
                                buf = memoryview(msg[0])

                                # Reconstruct theta matrix from min, max, no. of intervals and which bins
                                # each parameter value falls in
                                struct_field_names = ["min_val", "max_val", "interval", "bins"]
                                struct_field_types = [np.float32, np.float32, np.int32, 'b']
                                struct_field_shapes = [1, 1, 1, ((n_features, n_classes))]
                                dtype=(list(zip(struct_field_names, struct_field_types, struct_field_shapes)))
                                                                
                                data = np.frombuffer(buf, dtype=dtype)
                                min_theta_val, max_theta_val, interval, theta_bins = data[0]

                                # Generate lineared space vector
                                bins = np.linspace(min_theta_val, max_theta_val, interval)
                                theta = bins[theta_bins].reshape(n_features, n_classes)                             

                            # Each worker does work and we get the resulting gradients
                            d_theta, batch_loss, most_representative = self.do_work(self.X, self.y, theta)
                            self.logger.debug(f"iteration = {i}, Loss = {batch_loss:7.4f}")
                            i += 1

                            # Get messages ready to send by converting them to bytes format. We do not
                            # need to send the shape since the gradients have the same shape as theta which
                            # the master already owns
                            msg = d_theta.tostring()
                            loss = str(batch_loss).encode()
                            mr = most_representative.tostring()

                            # self.push_socket.send(b"WORK")
                            # self.router_socket.send_multipart([b"MASTER", b"WORK"])
                            # self.logger.debug("Sent work command")
                            # self.router_socket.send_multipart([b"WORK", msg, loss, mr], zmq.NOBLOCK)
                            self.push_socket.send_multipart([b"WORK", self.worker_id.encode(), msg, loss, mr])

                            self.logger.debug("Sent work back to master")
                else:

                    self.logger.info("Connecting to server")
                    self.push_socket.send_multipart([b"CONNECT", self.worker_id.encode()])
                    self.logger.debug("Connected")

                    # Receive X, y
                    data, dtype, shape = self.ctrl_socket.recv_multipart()
                    shape = shape.decode()
                    data = np.frombuffer(data, dtype=dtype)
                    data = data.reshape(eval(shape))

                    # Receive shape of X, y so we can reshape
                    n_samples, n_features, n_classes, scenario = self.ctrl_socket.recv_multipart()
                    n_samples = int(n_samples.decode())
                    n_features = int(n_features.decode())
                    n_classes = int(n_classes.decode())
                    self.scenario = int(scenario.decode())

                    self.X, self.y = data[:, :n_features], data[:, -n_classes:]

                    # Check if we need to add a new axis if the dimension of y is not 2d
                    if len(self.y.shape) < 2:
                        self.y = self.y[:, np.newaxis]
                    self.logger.info(f"Received data, X.shape={self.X.shape}, y.shape={self.y.shape}")
                    self.have_work = True
                    # poller.register(self.ctrl_socket, zmq.POLLIN)

            end = time.time()

            self.logger.info("Time taken for %d iterations is %7.6fs" % (i, end-start))
        except KeyboardInterrupt as e:
            self.logger.info("Keyboard quit")
        except zmq.ZMQError:
            self.logger.info("ZMQError")
        finally:
            poller.unregister(self.router_socket)
            poller.unregister(self.subscriber)
            self.kill()

    def kill(self):
        self.subscriber.close()
        self.router_socket.close()
        self.ctrl_socket.close()
        # self.context.term()

@click.command()
@click.option('--verbose', '-v', default=10, type=int)
def run(verbose):
    worker = Worker(
        verbose=verbose
    )
    worker.connect()
    time.sleep(1)
    worker.start()

if __name__ == "__main__":
    run()