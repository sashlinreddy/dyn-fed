import zmq
import time
import numpy as np
import uuid
import logging

# Local
from fault_tolerant_ml.utils import setup_logger
from fault_tolerant_ml.utils import zhelpers
from fault_tolerant_ml.ml import hypotheses, loss_fns

class Worker(object):

    def __init__(self):

        self.worker_id = str(uuid.uuid4())
        self.subscriber = None
        self.starter = None
        self.logger = logging.getLogger("masters")
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

        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.connect("tcp://localhost:5564")

        self.ctrl_socket = self.context.socket(zmq.ROUTER)
        self.ctrl_socket.setsockopt_string(zmq.IDENTITY, self.worker_id)
        self.ctrl_socket.connect("tcp://localhost:5565")

    def do_work(self, X, y, theta):

        self.logger.info('Doing work')
        # Get predictions
        h = self.hypothesis(X, theta)
        # Calculate error/residuals
        e = (h - y)

        # Calculate log loss
        log_loss = -y * np.log(h) - (1 - y) * np.log(1 - h)

        # TODO: Calculate most representative data points

        # Calculate processor loss - this is aggregated
        batch_loss = np.mean(log_loss)

        n_features, n_classes = theta.shape
        d_theta = np.zeros((n_features, n_classes))

        # To be moved to optimizer
        for k in np.arange(n_classes):
            d_theta[:, k] = self.gradient(X, e[:, np.newaxis, k])

        return d_theta

    def start(self):

        poller = zmq.Poller()
        poller.register(self.subscriber, zmq.POLLIN)
        poller.register(self.pull_socket, zmq.POLLIN)
        # poller.register(self.push_socket, zmq.POLLOUT)

        self.logger.info('Started Worker %s' % self.worker_id)

        try:
            start = time.time()
            i = 0

            # self.starter.send_multipart([b"READY", self.worker_id.encode()])
            # self.ctrl_socket.send(b"READY")

            while True:

                if self.have_work:

                    events = dict(poller.poll())

                    if events.get(self.subscriber) == zmq.POLLIN:
                        # Read envelope with address
                        # [address, contents] = self.subscriber.recv_multipart()
                        flags = 0
                        contents = self.subscriber.recv_json(flags=flags)
                        # finished = contents == b"EXIT"
                        # print(contents == b"EXIT")
                        if "EXIT" in contents:
                            self.logger.info("Received EXIT command")
                            break
                        else:
                            msg = self.subscriber.recv(flags=flags, copy=True, track=False)
                            buf = memoryview(msg)
                            theta = np.frombuffer(buf, dtype=contents['dtype'])
                            theta = theta.reshape(contents['shape'])
                            # theta = np.frombuffer(contents, dtype=np.float64)
                            self.logger.info(f"theta.shape{theta.shape}")
                            
                            theta = theta.copy()

                            # theta /= 2
                            d_theta = self.do_work(self.X, self.y, theta)
                            i += 1
                            msg = d_theta.tostring()

                            # self.push_socket.send(msg)
                            self.push_socket.send_multipart([b"WORK", msg])
                            self.logger.info("Sent result back to master")
                            # self.logger.info("[%s] %s" % (address, contents))
                else:

                    self.logger.info("Connecting to server")
                    self.push_socket.send_multipart([b"CONNECT", self.worker_id.encode()])

                    worker_id = self.ctrl_socket.recv()
                    data = zhelpers.recv_array(self.ctrl_socket)

                    worker_id = self.ctrl_socket.recv()
                    samp_feat_d = self.ctrl_socket.recv_json()
                    n_samples = samp_feat_d["n_samples"]
                    n_features = samp_feat_d["n_features"]
                    n_classes = samp_feat_d["n_classes"]
                    self.X, self.y = data[:, :n_features], data[:, :-n_classes]
                    # worker_id, data = self.ctrl_socket.recv_multipart()
                    # np_data = np.frombuffer(data, dtype=np.float64)
                    self.logger.info(f"Received data, shape={data.shape}")
                    self.have_work = True

            end = time.time()

            self.logger.info("Time taken for %d iterations is %7.6fs" % (i, end-start))
        except KeyboardInterrupt as e:
            self.logger.info("Keyboard quit")
        except zmq.ZMQError:
            self.logger.info("ZMQError")
        finally:
            poller.unregister(self.pull_socket)
            poller.unregister(self.subscriber)
            self.kill()

    def kill(self):
        self.subscriber.close()
        self.pull_socket.close()
        self.ctrl_socket.close()
        # self.context.term()

if __name__ == "__main__":
    logger = setup_logger()
    worker = Worker()
    worker.connect()
    time.sleep(1)
    worker.start()