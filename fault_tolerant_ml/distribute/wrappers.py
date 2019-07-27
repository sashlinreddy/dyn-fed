import time
import zmq.green as zmq
from fault_tolerant_ml.distribute.states import *
class ftml_wrapper(object):

    def __init__(self, decorated):
        self.decorated = decorated

    def __get__(self, instance, owner):
        self.cls = owner
        self.obj = instance

        return self.__call__

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class ftml_train(ftml_wrapper):

        def __call__(self, *args, **kwargs):

            try:
                # Detect all workers by polling by whoevers sending their worker ids
                self.obj.detect_workers()
                if not self.obj.watch_dog.states:
                    self.obj.logger.info("No workers found")
                    raise KeyboardInterrupt

                self.decorated(self.obj)
        
            except KeyboardInterrupt as e:
                pass
            except zmq.ZMQError as zmq_err:
                self.obj.logger.error(zmq_err)
                self.obj.done()
            except Exception as e:
                self.obj.logger.exception(e)
            finally:
                self.obj.logger.info("Exiting peacefully. Cleaning up...")

class ftml_trainv2(ftml_wrapper):

    def __call__(self, *args, **kwargs):

        try:
            # Detect all workers by polling by whoevers sending their worker ids
            self.obj.detect_workers()
            if not self.obj.watch_dog.states:
                self.obj.logger.info("No workers found")
                raise KeyboardInterrupt
            # self.obj.model.iter = 0
            # i = 0
            delta = 1.0
            start = time.time()

            # while i < self.n_iterations:
            while self.obj.model.iter < self.obj.n_iterations:

                events = dict(self.obj.poller.poll())

                if len(self.obj.watch_dog.states) > 0:
                    self.decorated(self.obj, events)
                else:
                    if (self.obj.pull_socket in events) and (events.get(self.obj.pull_socket) == zmq.POLLIN):
                        # Don't use the results if they've already been counted
                        command = self.obj.pull_socket.recv(flags=zmq.SNDMORE)

                        if command == b"CONNECT":
                            self.obj.register_workers()

            # Tell workers to exit
            self.obj.done()
            self.obj.state = COMPLETE
            end = time.time()
            self.obj.logger.info("Time taken for %d iterations is %7.6fs" % (self.obj.n_iterations, end-start))

        except KeyboardInterrupt as e:
            pass
        except zmq.ZMQError as zmq_err:
            self.obj.logger.error(zmq_err)
            self.obj.done()
        except Exception as e:
            self.obj.logger.exception(e)
        finally:
            self.obj.logger.info("Exiting peacefully. Cleaning up...")

class ftml_train_collect(ftml_wrapper):

    def __call__(self, *args, **kwargs):
        events = args[0]
        # Check heartbeat
        if (self.obj.ctrl_socket in events) and (events.get(self.obj.ctrl_socket) == zmq.POLLIN):
            address, msg = self.obj.ctrl_socket.recv_multipart()
            self.obj.watch_dog.states[address].state = True
            self.obj.logger.debug(f"Address={address.decode()}, Msg={msg.decode()}")

        if (self.obj.pull_socket in events) and (events.get(self.obj.pull_socket) == zmq.POLLIN):
            # Don't use the results if they've already been counted
            command = self.obj.pull_socket.recv(flags=zmq.SNDMORE)

            if command == b"CONNECT":
                self.obj.register_workers()
                self.obj.state = MAP

            elif command == b"WORK":
                delta = self.decorated(self.obj, events)

                if self.obj.state != REMAP:
                    self.obj.state = DIST_PARAMS
                
                self.obj.model.iter += 1
                if delta < self.obj.strategy.delta_switch and self.obj.strategy.comm_period > 1 and not self.obj.delay_change:
                    self.obj.delay_change = True
                    self.obj.n_iterations = self.obj.model.iter + (self.obj.n_iterations - self.obj.model.iter) * self.obj.strategy.comm_period
                    self.obj.logger.debug(f"Iterations now = {self.obj.n_iterations}")
