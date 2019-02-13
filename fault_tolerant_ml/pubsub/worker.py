import zmq
import time
import numpy as np
import uuid
import logging
from fault_tolerant_ml.utils import setup_logger

class Worker(object):

    def __init__(self):

        self.worker_id = str(uuid.uuid4())
        self.subscriber = None
        self.starter = None
        self.logger = logging.getLogger("masters")

    def connect(self):
        # Prepare our context and publisher
        self.context    = zmq.Context()
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect("tcp://localhost:5563")
        self.subscriber.setsockopt(zmq.SUBSCRIBE, b"")

        self.starter = self.context.socket(zmq.REQ)
        self.starter.connect("tcp://localhost:5564")

        self.ctrl_socket = self.context.socket(zmq.DEALER)
        self.ctrl_socket.setsockopt_string(zmq.IDENTITY, self.worker_id)
        self.ctrl_socket.connect("tcp://localhost:5565")

        self.poller = zmq.Poller()
        self.poller.register(self.subscriber, zmq.POLLIN)

    def start(self):

        self.logger.info('Started Worker %s' % self.worker_id)

        try:
            start = time.time()
            i = 0

            # self.starter.send_multipart([b"READY", self.worker_id.encode()])
            self.ctrl_socket.send(b"READY")

            while True:

                socks = dict(self.poller.poll())

                if self.subscriber in socks:
                    # Read envelope with address
                    [address, contents] = self.subscriber.recv_multipart()
                    # finished = contents == b"EXIT"
                    # print(contents == b"EXIT")
                    if contents == b"EXIT":
                        break
                    else:
                        np_contents = np.frombuffer(contents, dtype=np.int32)
                        # self.logger.info("[%s] %s" % (address, np_contents.shape))
                i += 1

            end = time.time()

            self.logger.info("Time taken for %d iterations is %7.6fs" % (i, end-start))
        except KeyboardInterrupt as e:
            pass
        finally:
            self.kill()

    def kill(self):
        self.subscriber.close()
        self.starter.close()
        self.ctrl_socket.close()
        self.context.term()

if __name__ == "__main__":
    logger = setup_logger()
    worker = Worker()
    worker.connect()
    worker.start()