import zmq
from dyn_fed.data import DummyData

N_TASKS = 40
NBR_WORKERS = 5
context = zmq.Context()
client = context.socket(zmq.ROUTER)
client.bind("tcp://*:5671")

dummy_data = DummyData(filepath="", n_samples=100, lower_bound=0, upper_bound=10)
print(f"Dummy data.shape={dummy_data.X.shape}")
# msg = b'This is the workload'
msg = dummy_data.to_string()

for _ in range(N_TASKS):
    # LRU worker is next waiting in the queue
    address, empty, ready = client.recv_multipart()

    client.send_multipart([
        address,
        b'',
        msg,
    ])

# Now ask mama to shut down and report their results
for _ in range(NBR_WORKERS):
    address, empty, ready = client.recv_multipart()
    client.send_multipart([
        address,
        b'',
        b'END',
    ])
