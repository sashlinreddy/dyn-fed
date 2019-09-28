"""Master with heartbeater
"""
import logging
import time
import threading
import signal

import numpy as np
import zmq.green as zmq
import gevent

# import tornado

from fault_tolerant_ml.distribute.heartbeater import Heartbeater
from fault_tolerant_ml.distribute.states import MAP, DIST_PARAMS, REDUCE, START

logging.basicConfig(level="INFO")
logger = logging.getLogger('ftml')

# pylint: disable=no-member

class Master():
    """Master class
    """
    def __init__(self, period=1000):
        # Sockets
        self.heart_pub_socket = None
        self.heart_ctrl_socket = None
        self.pub_socket = None
        self.ctrl_socket = None
        self.pull_socket = None

        # Polling
        self.poller = None
        self.period = period
        self.state = START

        # Model variables
        self.n_iterations = 5
        self.iteration = 0
        self.expected_workers = 3

        self.heartbeater = Heartbeater(self.expected_workers, period)
        # Heartbeater variables
        # self.hearts = set()
        # self.responses = set()
        # self.lifetime = 0
        # self.tic = time.time()

    def connect(self):
        """Connect sockets
        """
        context = zmq.Context()

        # Heart sockets
        self.heart_pub_socket = context.socket(zmq.PUB)
        self.heart_pub_socket.bind("tcp://*:5564")

        self.heart_ctrl_socket = context.socket(zmq.ROUTER)
        self.heart_ctrl_socket.setsockopt_string(zmq.IDENTITY, 'MASTER')
        self.heart_ctrl_socket.bind("tcp://*:5565")

        # Normal sockets
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:5563")

        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind("tcp://*:5562")

        self.ctrl_socket = context.socket(zmq.ROUTER)
        self.ctrl_socket.setsockopt_string(zmq.IDENTITY, 'MASTER')
        self.ctrl_socket.bind("tcp://*:5566")

        self.poller = self.setup_poller()

    def setup_poller(self):
        """Setup poller
        """
        poller = zmq.Poller()
        poller.register(self.pull_socket, zmq.POLLIN | zmq.POLLERR)
        poller.register(self.ctrl_socket, zmq.POLLIN | zmq.POLLERR)
        poller.register(self.pub_socket, zmq.POLLOUT | zmq.POLLERR)

        poller.register(self.heart_pub_socket, zmq.POLLOUT | zmq.POLLERR)
        poller.register(self.heart_ctrl_socket, zmq.POLLIN | zmq.POLLERR)

        return poller

    def start(self):
        """Start server
        """
        self.connect()

        gevent.signal(signal.SIGQUIT, gevent.kill)

        heart_loop = gevent.spawn(self.heart_loop)
        server_loop = gevent.spawn(self.send_work)
        
        gevent.joinall([
            heart_loop,
            server_loop
        ])

    def heart_loop(self):
        """Heart
        """
        logger.info("Starting heart beater")
        while True:
            # Send beat
            self.state = self.heartbeater.beat(self.heart_pub_socket, self.state)
            # Receive responses
            gevent.sleep(1)
            events = dict(self.poller.poll())
            while (self.heart_ctrl_socket in events) and \
                (events.get(self.heart_ctrl_socket) == zmq.POLLIN):
                events = dict(self.poller.poll())
                if (self.heart_ctrl_socket in events) and \
                    (events.get(self.heart_ctrl_socket) == zmq.POLLIN):
                    msg = self.heart_ctrl_socket.recv_multipart()
                    self.heartbeater.handle_pong(msg)

    def kill(self):
        """Kills sockets
        """
        self.poller.unregister(self.pull_socket)
        self.poller.unregister(self.pub_socket)
        self.poller.unregister(self.ctrl_socket)
        self.poller.unregister(self.heart_pub_socket)
        self.poller.unregister(self.heart_ctrl_socket)

        self.pull_socket.close()
        self.pub_socket.close()
        self.ctrl_socket.close()
        self.heart_ctrl_socket.close()
        self.heart_pub_socket.close()
                
    def send_work(self):
        """Send work to workers
        """
           
        while True:
            gevent.sleep(0.00000001)
            if self.state == MAP:
                logger.info("Sending work to workers")
                # First map data
                A = np.random.randn(10, 1)
                logger.info(f"Hearts={self.heartbeater.hearts}")
                for heart in self.heartbeater.hearts:
                    logger.info(f"Work to {heart}")
                    msg = A.tostring()
                    self.ctrl_socket.send_multipart([heart, b"WORK_DATA", msg])

                self.state = DIST_PARAMS

            if self.state == DIST_PARAMS:

                logger.info(f"Iteration={self.iteration}")
                # Map params
                B = np.random.randn(5, 1)
                msg = B.tostring()
                logger.info("Sending params")
                self.pub_socket.send_multipart([b"WORK_PARAMS", msg])

                logger.info("Recv work")
                # Recv work
                i = 0
                hearts = len(self.heartbeater.hearts)
                while i < hearts:
                    gevent.sleep(0.000001)
                    events = dict(self.poller.poll())
                    if (self.pull_socket in events) and \
                        (events.get(self.pull_socket) == zmq.POLLIN):
                        msg = self.pull_socket.recv_multipart()
                        cmd = msg[0]
                        worker = msg[1]
                        content = msg[2]
                        if cmd == b"WORK":
                            buf = memoryview(content)
                            arr = np.frombuffer(buf, dtype=np.float).copy()
                            logger.info(
                                f"Received work from {worker}, content.shape={arr.shape}"
                            )

                        i += 1

                    # logger.info(
                    #     f"hearts={hearts}, "
                    #     f"self.heartbeater.hearts={len(self.heartbeater.hearts)}, "
                    #     f"i={i}"
                    # )
                    if hearts > len(self.heartbeater.hearts):
                        logger.info(
                            f"Changed no of hearts from {hearts} to {self.heartbeater.hearts}"
                            )
                        hearts = len(self.heartbeater.hearts)

                self.iteration += 1

if __name__ == "__main__":

    try:

        server = Master()

        server.start()

    except KeyboardInterrupt:
        logger.info("Keyboard quit")
    except zmq.ZMQError:
        logger.info("ZMQError")
    finally:
        server.kill()
        # subscriber.close()
        # ctrl_socket.close()
        # server.loop.stop()
