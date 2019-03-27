import zmq
import multiprocessing as mp

def worker_routine(context=None):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect("tcp://localhost:5560")
    # subscriber.setsockopt(zmq.SUBSCRIBE, b"B")

    try:
        while True:
            message = socket.recv()
            print("Received request: %s" % message)
            socket.send(b"World")
    except KeyboardInterrupt as e:
        print("Ending session")
    finally:
        socket.close()
        context.term()

def main():
    n_workers = 3
    print(f"Starting {n_workers} workers")
    # Launch pool of worker threads
    for i in range(n_workers):
        process = mp.Process(target=worker_routine)
        # thread.daemon = True
        process.start()

if __name__ == '__main__':
    main()