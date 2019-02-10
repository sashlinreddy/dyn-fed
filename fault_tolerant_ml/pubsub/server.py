import time
import zmq
import numpy as np
import logging
from fault_tolerant_ml.utils import setup_logger

class Master(object):

    def __init__(self):
        
        self.ctrl_socket = None
        self.publisher = None
        self.receiver = None

        self.workers = set()

        # Setup logger
        self.logger = logging.getLogger("masters")

    def connect(self):
        
        # Prepare our context and publisher
        self.context   = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind("tcp://*:5563")

        self.receiver = self.context.socket(zmq.REP)
        self.receiver.bind("tcp://*:5564")    

    def start(self):
        arr_size = 100
        x = np.random.randint(0, 10, size=(arr_size,), dtype=np.int32)
        self.logger.info(f"Initialized array of size {arr_size}")
        # print(f"x={x}")
        msg = x.tostring()
        n_iterations = 1000
        i = 0
        
        time.sleep(1)
        start = self.receiver.recv()
        self.logger.info(start.decode())

        try:
            while i < n_iterations:
                # Write two messages, each with an envelope and content
                self.publisher.send_multipart([b"A", msg])
                self.publisher.send_multipart([b"B", msg])
                i += 1
                # time.sleep(0.25)

            time.sleep(1)
            self.publisher.send_multipart([b"B", b"EXIT"])
        except KeyboardInterrupt as e:
            pass
        finally:
            # We never get here but clean up anyhow
            self.logger.info("Exiting peacefully. Cleaning up...")
            self.kill()

    def kill(self):
        self.publisher.close()
        self.receiver.close()
        self.context.term()

    def done(self):
        pass

def main():
    """main method"""

    # Prepare our context and publisher
    context   = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5563")

    receiver = context.socket(zmq.REP)
    receiver.bind("tcp://*:5564")

    workers = set()

    arr_size = 100
    x = np.random.randint(0, 10, size=(arr_size,), dtype=np.int32)
    print(f"Initialized array of size {arr_size}")
    # print(f"x={x}")
    msg = x.tostring()
    n_iterations = 1000
    i = 0
    
    time.sleep(1)
    start = receiver.recv()
    print(start.decode())

    try:
        while i < n_iterations:
            # Write two messages, each with an envelope and content
            publisher.send_multipart([b"A", msg])
            publisher.send_multipart([b"B", msg])
            i += 1
            # time.sleep(0.25)

        time.sleep(1)
        publisher.send_multipart([b"B", b"EXIT"])
    except KeyboardInterrupt as e:
        pass
    finally:
        # We never get here but clean up anyhow
        print("Exiting peacefully. Cleaning up...")
        publisher.close()
        receiver.close()
        context.term()


if __name__ == "__main__":
    # main()
    logger = setup_logger()
    master = Master()
    master.connect()
    master.start()