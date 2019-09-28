"""Class handling heartbeater
"""
import time
import logging

from fault_tolerant_ml.distribute.states import START, MAP

logger = logging.getLogger('ftml')

class Heartbeater():
    """Heartbeater class
    """
    def __init__(self, n_workers, period):
        self.period = period
        self.n_workers = n_workers
        self.lifetime = 0

        self.hearts = set()
        self.responses = set()
        self.lifetime = 0
        self.tic = time.time()

    def beat(self, socket, state):
        """Handle single heartbeat
        """
        toc = time.time()
        self.lifetime += toc-self.tic
        self.tic = toc
        logger.info(self.lifetime)
        # self.message = str(self.lifetime)
        # logger.info(f"Responses={self.responses}")
        goodhearts = self.hearts.intersection(self.responses)
        heartfailures = self.hearts.difference(goodhearts)
        newhearts = self.responses.difference(goodhearts)
        # print(newhearts, goodhearts, heartfailures)
        list(map(self.handle_new_heart, newhearts))
        list(map(self.handle_heart_failure, heartfailures))

        # If there are new hearts we need to map data to everyone again
        if newhearts:
            state = MAP

        # If we have 
        self.responses = set()
        logger.info("%i beating hearts: %s", len(self.hearts), self.hearts)
        if state == START:
            logger.info("Sending connect")
            socket.send_multipart([b"CONNECT", str(self.lifetime).encode()])
        else:
            logger.info("Normal heartbeat")
            socket.send(str(self.lifetime).encode())

        return state

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
