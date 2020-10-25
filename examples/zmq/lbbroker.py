"""LB Broker example
"""
from __future__ import print_function

import multiprocessing

import zmq


NBR_CLIENTS = 10
NBR_WORKERS = 3


def client_task(ident):
    """Basic request-reply client using REQ socket."""
    socket = zmq.Context().socket(zmq.REQ) # pylint: disable=no-member
    socket.identity = u"Client-{}".format(ident).encode("ascii")
    socket.connect("ipc://frontend.ipc")

    # Send request, get reply
    socket.send(b"HELLO")
    reply = socket.recv()
    print("{}: {}".format(socket.identity.decode("ascii"),
                          reply.decode("ascii")))


def worker_task(ident):
    """Worker task, using a REQ socket to do load-balancing."""
    socket = zmq.Context().socket(zmq.REQ) # pylint: disable=no-member
    socket.identity = u"Worker-{}".format(ident).encode("ascii")
    socket.connect("ipc://backend.ipc")

    # Tell broker we're ready for work
    socket.send(b"READY")

    while True:
        address, _, request = socket.recv_multipart() # pylint: disable=unbalanced-tuple-unpacking
        print("{}: {}".format(socket.identity.decode("ascii"),
                              request.decode("ascii")))
        socket.send_multipart([address, b"", b"OK"])


def main():
    """Load balancer main loop."""
    # Prepare context and sockets
    context = zmq.Context.instance()
    frontend = context.socket(zmq.ROUTER) # pylint: disable=no-member
    frontend.bind("ipc://frontend.ipc")
    backend = context.socket(zmq.ROUTER) # pylint: disable=no-member
    backend.bind("ipc://backend.ipc")

    # Start background tasks
    def start(task, *args):
        process = multiprocessing.Process(target=task, args=args)
        process.daemon = True
        process.start()
    for i in range(NBR_CLIENTS):
        start(client_task, i)
    for i in range(NBR_WORKERS):
        start(worker_task, i)

    # Initialize main loop state
    count = NBR_CLIENTS
    workers = []
    poller = zmq.Poller()
    # Only poll for requests from backend until workers are available
    poller.register(backend, zmq.POLLIN)

    while True:
        sockets = dict(poller.poll())

        if backend in sockets:
            # Handle client activity on the backend
            request = backend.recv_multipart()
            client, _, client = request[:3]
            if not workers:
                # Poll for clients now that a client is available
                poller.register(frontend, zmq.POLLIN)
            workers.append(client)
            if client != b"READY" and len(request) > 3:
                # If client reply, send rest back to frontend
                _, reply = request[3:]
                frontend.send_multipart([client, b"", reply])
                count -= 1
                if not count:
                    break

        if frontend in sockets:
            # Get next client request, route to last-used client
            client, _, request = frontend.recv_multipart()
            client = workers.pop(0)
            backend.send_multipart([client, b"", client, b"", request])
            if not workers:
                # Don't poll clients if no workers are available
                poller.unregister(frontend)

    # Clean up
    backend.close()
    frontend.close()
    context.term()


if __name__ == "__main__":
    main()
