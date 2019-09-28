"""Client example using zmqstreams and heartbeating. To be used with server.py
"""
import logging
import time
import uuid

import numpy as np
import zmq
from zmq import devices
from zmq.eventloop import zmqstream
from tornado import ioloop

logging.basicConfig(level="INFO")
logger = logging.getLogger()

# pylint: disable=no-member

class Client():
    """Client class
    """
    def __init__(self, worker_id, io_loop, sub, push, ctrl):
        self.io_loop = io_loop
        self.sub = sub
        self.push = push
        self.ctrl = ctrl
        self.worker_id = worker_id

        logger.info("Setting up")
        self.ctrl.on_recv(self.recv_work)
        # self.ctrl.on_recv(self.handle_dealer)

    def handle_dealer(self, msg):
        """Handle dealer streams
        """
        command = msg[0]
        logger.info(f"Command={command}")

        if command == b"CONNECTED":
            logger.info("WORKER RECEIVED CONNECT")

    def recv_work(self, msg):
        """Receive work
        """
        logger.info("Recv work")
        cmd = msg[0]
        if cmd == b"WORK_DATA":
            buf = memoryview(msg[1])
            arr = np.frombuffer(buf, dtype=np.float)
            logger.info(f"Data.shape={arr.shape}")
            # self.ctrl.stop_on_recv()
            # After receiving data we can recv params
            self.sub.on_recv(self.recv_params)

    def recv_params(self, msg):
        """Recv params
        """
        cmd = msg[0]
        if cmd == b"WORK_PARAMS":
            buf = memoryview(msg[1])
            arr = np.frombuffer(buf, dtype=np.float).copy()
            logger.info(f"Params.shape={arr.shape}")

            # Do some work
            arr = arr * 2
            A = np.random.random((2**11, 2**11))
            tic = time.time()
            np.dot(A, A.transpose())
            logger.info("blocked for %.3f s"%(time.time() - tic))

            self.push.send_multipart(
                [b"WORK", self.worker_id.encode(), arr.tostring()]
            )


if __name__ == "__main__":

    time.sleep(1)
    worker_id = str(uuid.uuid4())

    try:

        loop = ioloop.IOLoop()
        context = zmq.Context()

        dev = devices.ThreadDevice(zmq.FORWARDER, zmq.SUB, zmq.DEALER)
        dev.setsockopt_in(zmq.SUBSCRIBE, b"")
        dev.setsockopt_out(zmq.IDENTITY, worker_id.encode())
        dev.connect_in('tcp://127.0.0.1:5564')
        dev.connect_out('tcp://127.0.0.1:5565')
        dev.start()

        subscriber = context.socket(zmq.SUB) # pylint: disable=no-member
        subscriber.connect("tcp://localhost:5563")
        # subscriber.connect(f"tcp://{master_ip_address}:5563")
        subscriber.setsockopt(zmq.SUBSCRIBE, b"") # pylint: disable=no-member

        push_socket = context.socket(zmq.PUSH) # pylint: disable=no-member
        push_socket.connect("tcp://localhost:5562")
        # push_socket.connect(f"tcp://{master_ip_address}:5562")

        ctrl_socket = context.socket(zmq.DEALER) # pylint: disable=no-member
        ctrl_socket.setsockopt_string(zmq.IDENTITY, worker_id) # pylint: disable=no-member
        ctrl_socket.connect("tcp://localhost:5566")
        # ctrl_socket.connect(f"tcp://{master_ip_address}:5565")

        sub_stream = zmqstream.ZMQStream(subscriber, loop)
        ctrl_stream = zmqstream.ZMQStream(ctrl_socket, loop)

        # sub_stream.on_recv_stream(handle_dealer)
        # push_socket.send_multipart([b"CONNECT", worker_id.encode()])

        client = Client(worker_id, loop, sub_stream, push_socket, ctrl_stream)

        # wait for connections
        time.sleep(1)

        # A = np.random.random((2**11, 2**11))
        # print("starting blocking loop")
        # while True:
        #     tic = time.time()
        #     np.dot(A, A.transpose())
        #     print("blocked for %.3f s"%(time.time() - tic))

        loop.start()

    except KeyboardInterrupt:
        logger.info("Keyboard quit")
    except zmq.ZMQError:
        logger.info("ZMQError")
    finally:
        # subscriber.close()
        # ctrl_socket.close()
        loop.stop()
