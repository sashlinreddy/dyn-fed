import zmq
import random
import time
from dyn_fed.data import DummyData

try:
    raw_input
except NameError:
    # Python 3
    raw_input = input

context = zmq.Context()

# Socket to send messages on
sender = context.socket(zmq.PUSH)
# sender = context.socket(zmq.PUB)
sender.bind("tcp://*:5557")

# Socket with direct access to the sink: used to syncronize start of batch
sink = context.socket(zmq.PUSH)
sink.connect("tcp://localhost:5558")

print("Press Enter when the workers are ready: ")
_ = raw_input()
print("Sending tasks to workers...")

# The first message is "0" and signals start of batch
sink.send(b'0')

# Initialize random number generator
random.seed()

# dummy_data = DummyData(n_samples=100, n_features=10, n_classes=1)
# X, y = dummy_data.transform()

# Send 100 tasks
total_msec = 0
for task_nbr in range(100):

    # Random workload from 1 to 100 msecs
    workload = random.randint(1, 100)
    total_msec += workload

    # sender.send_string(u'%i' % workload)
    msg = str(workload).encode()
    sender.send(msg)

# n_workers = 2
# batch_size = 100 // n_workers

# for X_batch, y_batch in dummy_data.next_batch(batch_size):

#     data = X_batch.tostring()
#     print("Sending data")
#     sender.send(data)

print("Total expected cost: %s msec" % total_msec)

# Give 0MQ time to deliver
time.sleep(1)