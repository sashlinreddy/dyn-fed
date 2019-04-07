import zmq.green as zmq

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

    def __init__(self):
        pass

    def funcname(self, parameterlist):
        pass
                
