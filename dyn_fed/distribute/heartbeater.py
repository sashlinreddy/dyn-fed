"""Class handling heartbeater
"""
import time
import logging

from dyn_fed.distribute.states import START, MAP

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

        self._logger = logging.getLogger(f"dfl.distribute.{self.__class__.__name__}")

    def beat(self, socket, state, n_workers):
        """Handle single heartbeat
        """
        toc = time.time()
        self.lifetime += toc-self.tic
        self.tic = toc
        self._logger.info(self.lifetime)
        # self.message = str(self.lifetime)
        # self._logger.info(f"Responses={self.responses}")
        goodhearts = self.hearts.intersection(self.responses)
        heartfailures = self.hearts.difference(goodhearts)
        newhearts = self.responses.difference(goodhearts)
        # print(newhearts, goodhearts, heartfailures)
        list(map(self.handle_new_heart, newhearts))
        list(map(self.handle_heart_failure, heartfailures))

        # If there are new hearts we need to map data to everyone again
        if newhearts and state != START:
            state = MAP
        if (len(self.hearts) >= n_workers) and (state == START):
            state = MAP

        # If we have 
        self.responses = set()
        self._logger.info("%i beating hearts: %s, state=%s", len(self.hearts), self.hearts, state)
        if state == START:
            self._logger.info("Sending connect")
            socket.send_multipart([b"CONNECT", str(self.lifetime).encode()])
        else:
            self._logger.info("Normal heartbeat")
            socket.send(str(self.lifetime).encode())

        return state, newhearts, heartfailures

    def handle_pong(self, msg):
        "if heart is beating"
        if msg[1].decode() == "CONNECT":
            self.responses.add(msg[0])
        elif msg[1].decode() == str(self.lifetime):
            self.responses.add(msg[0])
        else:
            self._logger.info("got bad heartbeat (possibly old?): %s", msg[1])

    def handle_new_heart(self, heart):
        """Handle new heart
        """
        self._logger.info("yay, got new heart %s!", heart)
        self.hearts.add(heart)

    def handle_heart_failure(self, heart):
        """Handle heart failure
        """
        self._logger.info("Heart %s failed :(", heart)
        self.hearts.remove(heart)
