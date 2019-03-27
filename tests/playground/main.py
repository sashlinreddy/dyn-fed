import sys
import zmq
from multiprocessing import Process
import multiprocessing as mp
import numpy as np
import time
from random import choice
import zhelpers

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

class Base(Process):
    """
    Inherit from Process and
    holds the zmq address.
    """
    def __init__(self):
        super().__init__()
        # self.address = address

class ClientTask(Base):
    """ClientTask"""
    def get_rank(self):
        # Returns relative PID of a pool process
        return mp.current_process()._identity[0]

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        print(f"Client {self.get_rank()}")
        # identity = 'worker-%d' % self.get_rank()
        # socket.setsockopt_string(zmq.IDENTITY, identity)
        zhelpers.set_id(socket)
        socket.connect('tcp://localhost:5570')
        # print('Client %s started' % (identity))
        poll = zmq.Poller()
        poll.register(socket, zmq.POLLIN)
        reqs = 0
        while True:
            for i in range(5):
                sockets = dict(poll.poll(1000))
                if socket in sockets:
                    if sockets[socket] == zmq.POLLIN:
                        msg = socket.recv()
                        print('Client %s received: %s\n' % (identity, msg))
                        del msg
            reqs = reqs + 1
            print('Req #%d sent..' % (reqs))
            socket.send_string('request #%d' % (reqs))
        socket.close()
        context.term()

class ServerTask(Base):
    """ServerTask"""
    def run(self):
        context = zmq.Context()
        frontend = context.socket(zmq.ROUTER)
        frontend.bind('tcp://*:5570')
        backend = context.socket(zmq.DEALER)
        backend.bind('inproc://backend')
        workers = []
        for i in range(5):
            worker = ServerWorker()
            worker.start()
            workers.append(worker)
        poll = zmq.Poller()
        poll.register(frontend, zmq.POLLIN)
        poll.register(backend, zmq.POLLIN)
        while True:
            sockets = dict(poll.poll())
            if frontend in sockets:
                if sockets[frontend] == zmq.POLLIN:
                    _id = frontend.recv()
                    msg = frontend.recv()
                    print('Server received %s id %s\n' % (msg, _id))
                    backend.send(_id, zmq.SNDMORE)
                    backend.send(msg)
            if backend in sockets:
                if sockets[backend] == zmq.POLLIN:
                    _id = backend.recv()
                    msg = backend.recv()
                    print('Sending to frontend %s id %s\n' % (msg, _id))
                    frontend.send(_id, zmq.SNDMORE)
                    frontend.send(msg)
        frontend.close()
        backend.close()
        context.term()

class ServerWorker(Base):
    """ServerWorker"""
    def run(self):
        context = zmq.Context()
        worker = context.socket(zmq.DEALER)
        worker.connect('inproc://backend')
        print('Worker started')
        while True:
            _id = worker.recv()
            msg = worker.recv()
            print('Worker received %s from %s' % (msg, _id))
            replies = choice(range(5))
            for i in range(replies):
                time.sleep(1/choice(range(1,10)))
                worker.send(_id, zmq.SNDMORE)
                worker.send(msg)
        del msg
        worker.close()

def main():
    """main function"""
    server = ServerTask()
    server.start()
    for i in np.arange(3):
        client = ClientTask()
        client.start()
        client.join()
        
    server.join()

if __name__ == "__main__":
    main()