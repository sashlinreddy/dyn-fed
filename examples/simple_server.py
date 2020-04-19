"""Simple server example to connect to android
"""

from __future__ import print_function

import socket

import time
import zmq

# pylint: disable=no-member
def main():
    """Run server
    """

if __name__ == "__main__":
    try:
        hostname = socket.gethostname()
        print(f"Hostname={hostname}")
        _, _, ip_address = socket.gethostbyname_ex(hostname)
        ip_address = ip_address[-1]

        context = zmq.Context()
        rep = context.socket(zmq.ROUTER)
        rep.setsockopt_string(zmq.IDENTITY, 'MASTER')
        rep.bind("tcp://*:5560")

        print(f"Master connected on {ip_address}")

        time.sleep(1)

        while True:
            # time.sleep(1)
            print("Router: Waiting for recv")
            # msg = rep.recv_multipart()
            msg: bytes = rep.recv_multipart()

            client, m = msg
            m = m.decode()
            # print(f"Msg={msg}")
            # m = m.decode()
            print(f"Request received={m} from {client}")

            print("Sending back to client")
            reply = "World ".encode()
            # rep.se
            rep.send_multipart([client, reply])

    except KeyboardInterrupt:
        print("Keyboard interrupt")
    except zmq.ZMQError as e:
        print(e)
        print("ZMQError")
    finally:
        rep.close()
