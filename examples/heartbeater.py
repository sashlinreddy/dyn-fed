#!/usr/bin/env python
"""

For use with heart.py

A basic heartbeater using PUB and ROUTER sockets. pings are sent out on the PUB, and hearts
are tracked based on their DEALER identities.

You can start many hearts with heart.py, and the heartbeater will monitor all of them, and 
notice when they stop responding.

Authors
-------
* MinRK
"""
from __future__ import print_function

import time
import zmq
from zmq.eventloop import zmqstream
from tornado import ioloop

# pylint: disable=no-member

class HeartBeater():
    """A basic HeartBeater class
    pingstream: a PUB stream
    pongstream: an ROUTER stream"""

    def __init__(self, io_loop, pingstream, pongstream, period=1000):
        self.loop = io_loop
        self.period = period

        self.pingstream = pingstream
        self.pongstream = pongstream
        self.pongstream.on_recv(self.handle_pong)

        self.hearts = set()
        self.responses = set()
        self.lifetime = 0
        self.tic = time.time()

        self.caller = ioloop.PeriodicCallback(self.beat, period)
        self.caller.start()

    def beat(self):
        """Handles heartbeat
        """
        toc = time.time()
        self.lifetime += toc-self.tic
        self.tic = toc
        print(self.lifetime)
        # self.message = str(self.lifetime)
        goodhearts = self.hearts.intersection(self.responses)
        # print(f"Goodhearts={goodhearts}, response={self.responses}")
        heartfailures = self.hearts.difference(goodhearts)
        newhearts = self.responses.difference(goodhearts)
        # print(f"new={newhearts}, good={goodhearts}, fail={heartfailures}")
        list(map(self.handle_new_heart, newhearts))
        # print(f"Hearts={self.hearts}")
        list(map(self.handle_heart_failure, heartfailures))
        self.responses = set()
        print("%i beating hearts: %s"%(len(self.hearts), self.hearts))
        self.pingstream.send(str(self.lifetime).encode())

    def handle_new_heart(self, heart):
        """Handles new hearts
        """
        print("yay, got new heart %s!"%heart)
        self.hearts.add(heart)

    def handle_heart_failure(self, heart):
        """Handle heart failures
        """
        print("Heart %s failed :("%heart)
        self.hearts.remove(heart)

    def handle_pong(self, msg):
        "if heart is beating"
        if msg[1].decode() == str(self.lifetime):
            self.responses.add(msg[0])
        else:
            print("got bad heartbeat (possibly old?): %s"%msg[1])

# sub.setsockopt(zmq.SUBSCRIBE)


if __name__ == '__main__':
    loop = ioloop.IOLoop()
    context = zmq.Context()
    pub = context.socket(zmq.PUB)
    pub.bind('tcp://127.0.0.1:5555')
    router = context.socket(zmq.ROUTER)
    router.bind('tcp://127.0.0.1:5556')

    time.sleep(1)

    outstream = zmqstream.ZMQStream(pub, loop)
    instream = zmqstream.ZMQStream(router, loop)

    hb = HeartBeater(loop, outstream, instream)

    loop.start()
