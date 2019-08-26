"""Multiprocessor example with zmq
"""
# from threading import Thread
from multiprocessing import Process
import numpy as np

import zmq

from fault_tolerant_ml.utils import zhelpers

NBR_WORKERS = 5


def worker_thread(context=None):
    """Worker thread
    """
    # context = context or zmq.Context.instance()
    context = context or zmq.Context()
    worker = context.socket(zmq.REQ) # pylint: disable=no-member

    # We use a string identity for ease here
    zhelpers.set_id(worker)
    worker.connect("tcp://localhost:5671")

    total = 0
    while True:
        # Tell the router we're ready for work
        worker.send(b"ready")

        # Get workload from router, until finished
        workload = worker.recv()
        finished = workload == b"END"
        if finished:
            print("Processed: %d tasks" % total)
            break
        else:
            np_contents = np.frombuffer(workload, dtype=np.int32).copy()
            np_contents *= 2
            print(np_contents.shape)
        total += 1

        # Do some random work
        # time.sleep(0.1 * random.random())

for _ in range(NBR_WORKERS):
    Process(target=worker_thread).start()
    