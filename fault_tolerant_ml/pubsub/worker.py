import zmq
import time
import numpy as np
import uuid
import logging
from fault_tolerant_ml.utils import setup_logger

class Worker(object):

    def __init__(self):

        self.subscriber = None
        self.starter = None
        self.logger = logging.getLogger("masters")

    def connect(self):
        # Prepare our context and publisher
        self.context    = zmq.Context()
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect("tcp://localhost:5563")
        self.subscriber.setsockopt(zmq.SUBSCRIBE, b"B")
        self.starter = self.context.socket(zmq.REQ)
        self.starter.connect("tcp://localhost:5564")        

    def start(self):
        try:
            start = time.time()
            i = 0

            self.starter.send(b"READY")

            while True:
                # Read envelope with address
                [address, contents] = self.subscriber.recv_multipart()
                # finished = contents == b"EXIT"
                # print(contents == b"EXIT")
                if contents == b"EXIT":
                    break
                else:
                    np_contents = np.frombuffer(contents, dtype=np.int32)
                    self.logger.info("[%s] %s" % (address, np_contents.shape))
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
        self.context.term()

def main():
    """ main method """

    # Prepare our context and publisher
    context    = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://localhost:5563")
    subscriber.setsockopt(zmq.SUBSCRIBE, b"B")
    starter = context.socket(zmq.REQ)
    starter.connect("tcp://localhost:5564")

    worker_id = str(uuid.uuid4())

    try:
        start = time.time()
        i = 0

        starter.send(b"READY")

        while True:
            # Read envelope with address
            [address, contents] = subscriber.recv_multipart()
            # finished = contents == b"EXIT"
            # print(contents == b"EXIT")
            if contents == b"EXIT":
                break
            else:
                np_contents = np.frombuffer(contents, dtype=np.int32)
                print("[%s] %s" % (address, np_contents.shape))
            i += 1

        end = time.time()

        print("Time taken for %d iterations is %7.6fs" % (i, end-start))
    except KeyboardInterrupt as e:
        pass
    finally:
        # We never get here but clean up anyhow
        print("Exiting peacefully. Cleaning up...")
        subscriber.close()
        starter.close()
        context.term()


if __name__ == "__main__":
    # main()
    logger = setup_logger()
    worker = Worker()
    worker.connect()
    worker.start()