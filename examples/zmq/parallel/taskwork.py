import sys
import time
import zmq
import multiprocessing as mp
import numpy as np

def worker():
    context = zmq.Context()

    # Socket to receive messages on
    receiver = context.socket(zmq.PULL)
    # receiver = context.socket(zmq.SUB)
    receiver.connect("tcp://localhost:5557")
    # receiver.setsockopt(zmq.SUBSCRIBE, b"")

    # Socket to send messages to
    sender = context.socket(zmq.PUSH)
    sender.connect("tcp://localhost:5558")

    # Socket for control input
    controller = context.socket(zmq.SUB)
    controller.connect("tcp://localhost:5559")
    controller.setsockopt(zmq.SUBSCRIBE, b"")

    # Process messages from receiver and controller
    poller = zmq.Poller()
    poller.register(receiver, zmq.POLLIN)
    poller.register(controller, zmq.POLLIN)
    # Process messages from both sockets
    while True:
        socks = dict(poller.poll())

        if socks.get(receiver) == zmq.POLLIN:
            message = receiver.recv_string()
            # message = receiver.recv()
            # [address, message] = receiver.recv_multipart()

            # Process task
            workload = int(message)  # Workload in msecs

            # Do the work
            time.sleep(workload / 1000.0)
            # np_data = np.frombuffer(message, dtype=np.float64)
            # print(f"Received data, shape={np_data.shape}")

            # Send results to sink
            message = str(workload)
            sender.send_string(message)
            # sender.send(message)

            # Simple progress indicator for the viewer
            sys.stdout.write(".")
            sys.stdout.flush()

            # worker_id = mp.current_process()._identity
            # sys.stdout.write(f"worker_id={worker_id}")
            # sys.stdout.flush()

        # Any waiting controller command acts as 'KILL'
        if socks.get(controller) == zmq.POLLIN:
            worker_id = mp.current_process()._identity
            sys.stdout.write(f"Killing worker ={worker_id[0]}")
            sys.stdout.flush()
            break

    # Finished
    receiver.close()
    sender.close()
    controller.close()
    context.term()

def main():

    print("Starting workers")
    n_workers = 1
    # Launch pool of worker threads
    for i in range(n_workers):
        process = mp.Process(target=worker)
        # thread.daemon = True
        process.start()

if __name__ == '__main__':
    main()