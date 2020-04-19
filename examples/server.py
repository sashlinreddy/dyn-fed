"""Example using zmstreams and heartbeating
"""
import logging
import time
import signal

import numpy as np
import zmq.green as zmq
import gevent

from tornado import ioloop

from dyn_fed.distribute.states import MAP, MAP_PARAMS, START

logging.basicConfig(level="INFO")
logger = logging.getLogger()

# pylint: disable=no-member

class Server():
    """Server class
    """
    def __init__(self, period=1000):
        # Sockets
        self.heart_pub_socket = None
        self.heart_ctrl_socket = None
        self.pub_socket = None
        self.ctrl_socket = None

        # Streams
        self.loop = None
        self.pub = None
        self.pull = None
        self.ctrl = None
        self.heart_pub = None
        self.heart_ctrl = None
        self.poller = None
        self.period = period

        self.n_iterations = 5
        self.iteration = 0
        self.state = START
        self.caller = None

        # Heartbeater variables
        self.hearts = set()
        self.responses = set()
        self.expected_workers = 4
        self.at_start = True
        self.begin = False

        # self.heart_ctrl.on_recv(self.handle_pong)
        
        self.lifetime = 0
        self.tic = time.time()

        # t = threading.Thread(target=self.heart)
        # t.daemon = True
        # t.start()

        # self.pull.on_recv(self.detect_workers)

    def heart_connect(self):
        """Connect heart in separate thread
        """
        

    def connect(self):
        """Connect sockets
        """
        self.loop = ioloop.IOLoop()

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

        self.pull = context.socket(zmq.PULL)
        self.pull.bind("tcp://*:5562")

        self.ctrl_socket = context.socket(zmq.ROUTER)
        self.ctrl_socket.setsockopt_string(zmq.IDENTITY, 'MASTER')
        self.ctrl_socket.bind("tcp://*:5566")

        self.poller = self.setup_poller()

        # Heart streams
        # self.heart_pub = zmqstream.ZMQStream(self.heart_pub_socket, self.loop)
        # self.heart_ctrl = zmqstream.ZMQStream(self.heart_ctrl_socket, self.loop)

        # self.pub = zmqstream.ZMQStream(self.pub_socket, self.loop)
        # # pull_stream = zmqstream.ZMQStream(pull_socket, loop)
        # self.ctrl = zmqstream.ZMQStream(self.ctrl_socket, self.loop)

    def setup_poller(self):
        """Setup poller
        """
        poller = zmq.Poller()
        poller.register(self.pull, zmq.POLLIN | zmq.POLLERR)
        poller.register(self.ctrl_socket, zmq.POLLIN | zmq.POLLERR)
        poller.register(self.pub_socket, zmq.POLLOUT | zmq.POLLERR)

        poller.register(self.heart_pub_socket, zmq.POLLOUT | zmq.POLLERR)
        poller.register(self.heart_ctrl_socket, zmq.POLLIN | zmq.POLLERR)

        return poller

    def start(self):
        """Start server
        """
        self.connect()
        # self.heart_ctrl.on_recv(self.handle_pong)
        # self.heart()
        # self.loop.start()

        gevent.signal(signal.SIGQUIT, gevent.kill)

        heart_loop = gevent.spawn(self.heart)
        server_loop = gevent.spawn(self.send_work)
        
        gevent.joinall([
            heart_loop,
            server_loop
        ])

    def heart(self):
        """Heart
        """
        logger.info("Starting heart beater")
        while True:
            # Send beat
            self.beat()
            # Receive
            gevent.sleep(1)
            events = dict(self.poller.poll())
            while (self.heart_ctrl_socket in events) and \
                (events.get(self.heart_ctrl_socket) == zmq.POLLIN):
                events = dict(self.poller.poll())
                if (self.heart_ctrl_socket in events) and \
                    (events.get(self.heart_ctrl_socket) == zmq.POLLIN):
                    # Handle pong
                    msg = self.heart_ctrl_socket.recv_multipart()
                    self.handle_pong(msg)
    
    def beat(self):
        """Handles heartbeat
        """
        toc = time.time()
        self.lifetime += toc-self.tic
        self.tic = toc
        print(self.lifetime)
        # self.message = str(self.lifetime)
        logger.info(f"Responses={self.responses}")
        goodhearts = self.hearts.intersection(self.responses)
        heartfailures = self.hearts.difference(goodhearts)
        newhearts = self.responses.difference(goodhearts)
        # print(newhearts, goodhearts, heartfailures)
        list(map(self.handle_new_heart, newhearts))
        list(map(self.handle_heart_failure, heartfailures))

        if (self.state == START) and (len(self.hearts) >= self.expected_workers):
            self.state = MAP

        # If we have 
        self.responses = set()
        logger.info("%i beating hearts: %s", len(self.hearts), self.hearts)
        if self.state == START:
            logger.info("Sending connect")
            self.heart_pub_socket.send_multipart([b"CONNECT", str(self.lifetime).encode()])
        else:
            logger.info("Normal heartbeat")
            self.heart_pub_socket.send(str(self.lifetime).encode())
        # if self.at_start:
        #     if len(self.hearts) == self.expected_workers:
        #         self.begin = True

    def handle_pong(self, msg):
        "if heart is beating"
        if msg[1].decode() == "CONNECT":
            self.responses.add(msg[0])
        elif msg[1].decode() == str(self.lifetime):
            self.responses.add(msg[0])
        else:
            logger.info("got bad heartbeat (possibly old?): %s", msg[1])

    def handle_new_heart(self, heart):
        """Handle new heart
        """
        logger.info("yay, got new heart %s!", heart)
        self.hearts.add(heart)

    def handle_heart_failure(self, heart):
        """Handle heart failure
        """
        logger.info("Heart %s failed :(", heart)
        self.hearts.remove(heart)

    def init_map(self):
        """Initial data map
        """
        A = np.random.randn(10, 1)
        msg = A.tostring()
        self.pub.send_multipart([b"WORK", msg])

    def kill(self):
        """Kills sockets
        """
        self.poller.unregister(self.pull)
        self.poller.unregister(self.pub_socket)
        self.poller.unregister(self.ctrl_socket)
        self.poller.unregister(self.heart_pub_socket)
        self.poller.unregister(self.heart_ctrl_socket)

        self.pull.close()
        self.pub_socket.close()
        self.ctrl_socket.close()
        self.heart_ctrl_socket.close()
        self.heart_pub_socket.close()
        
    def send_work(self):
        """Send work to clients
        """
        # while True:
        #     print("Hello")
        #     gevent.sleep(0.3)

        # self.pub
        # print("Sending work")
        # if self.iteration == self.n_iterations:
        #     self.pub.stop_on_send()
        # else:
        #     if self.state == MAP:
        #         A = np.random.randn(10, 1)
        #         msg = A.tostring()
        #         self.pub.send_multipart([b"WORK", msg])
        #         self.state = REDUCE
        #         self.iteration += 1

        while True:
            gevent.sleep(0.000000001)

            if self.state == MAP:
                logger.info("Sending work to clients")
                # First map data
                A = np.random.randn(10, 1)
                logger.info(f"Hearts={self.hearts}")
                for heart in self.hearts:
                    logger.info(f"Work to {heart}")
                    msg = A.tostring()
                    self.ctrl_socket.send_multipart([heart, b"WORK_DATA", msg])

                self.state = MAP_PARAMS

            if self.state == MAP_PARAMS:
                while True:

                    gevent.sleep(0.00000001)
                    logger.info(f"Iteration={self.iteration}")
                    # Map params
                    B = np.random.randn(5, 1)
                    msg = B.tostring()
                    logger.info("Sending params")
                    self.pub_socket.send_multipart([b"WORK_PARAMS", msg])

                    logger.info("Recv work")
                    # Recv work
                    i = 0
                    hearts = len(self.hearts)
                    while i < hearts:
                        gevent.sleep(0.0000001)
                        events = dict(self.poller.poll())
                        if (self.pull in events) and (events.get(self.pull) == zmq.POLLIN):
                            msg = self.pull.recv_multipart()
                            logger.info(msg)
                            cmd = msg[0]
                            client = msg[1]
                            content = msg[2]
                            if cmd == b"WORK":
                                buf = memoryview(content)
                                arr = np.frombuffer(buf, dtype=np.float).copy()
                                logger.info(
                                    f"Received work from {client}, content.shape={arr.shape}"
                                )

                            i += 1

                        if hearts != len(self.hearts):
                            hearts = len(self.hearts)

                    self.iteration += 1

if __name__ == "__main__":

    try:

        server = Server()

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
