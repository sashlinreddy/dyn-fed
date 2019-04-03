import logging 

class Collector(object):
    
    def __init__(self):
        
        self._logger = logging.getLogger("ftml")

    def collect(self, events):
        """Receives gradients from workers

        Args:
            events (dict): Dictionary of events from our poller

        Returns:
            d_theta (numpy.ndarray): Our gradient matrix that is aggregated with a weighting according to the number    of samples each worker has
            epoch_loss (float): The loss for this epoch aggregated from each worker, also weighted according to the     work each worker did
        """
        d_theta = np.zeros(self.theta.shape)
        epoch_loss = 0.0

        self._logger.debug(f"Receiving gradients")
        n_alive_workers = self.watch_dog.n_alive
        self._logger.debug(f"Alive workers={n_alive_workers}")

        i = 0
        timeout = 1 # We give 1 seconds to poll worker if state changed since poll event
        running_time = 0
        n_connected = 0

        workers_received = set()

        while True:

            # Stop condition
            if i >= n_alive_workers:
                break

            # Timer to calculate running time for an iteration. We can then calculate the running time for 
            # an iteration so that if a state changes since a poll event, we can break if the running time 
            # exceeds the timeout
            start_i = time.time()

            if (self.pull_socket in events) and (events.get(self.pull_socket) == zmq.POLLIN):
                try:
                    msg = self.pull_socket.recv_multipart(zmq.NOBLOCK)
                except zmq.ZMQError as e:
                    if e.errno == zmq.EAGAIN:
                        # state changed since poll event
                        running_time += time.time() - start_i
                        if running_time > timeout:
                            self._logger.debug(f"Running time exceeded timeout={running_time}")
                            active_workers = set(self.watch_dog.active_workers)
                            # Get workers that we did not receive work from
                            diff = active_workers - workers_received
                            for w in diff:
                                # Set dead workers state to false
                                self.watch_dog.states[w].state = False
                                if self.scenario != 2:                                    
                                    self.watch_dog.states[w].idxs = self.watch_dog.states[w].most_representative
                            
                            self.state = REMAP
                            break

                        continue

                self._logger.debug(f"Alive workers={n_alive_workers}")
                if i == 0:
                    worker, d_theta_temp, epoch_loss_temp, mr = msg
                else:
                    # Receive multipart including command message
                    cmd = msg[0]
                    if cmd == b"WORK":
                        worker, d_theta_temp, epoch_loss_temp, mr = msg[1:]
                    elif cmd == b"CONNECT":
                        self.register_workers(msg[1])
                        n_connected += 1
                        i += 1
                        continue

                # Calculate weighting
                samples_for_worker = self.watch_dog.states[worker].n_samples
                beta = (samples_for_worker / self.data.n_samples)

                # Decode gradient matrix
                d_theta_temp = np.frombuffer(d_theta_temp, dtype=np.float64)
                d_theta_temp = d_theta_temp.reshape(self.theta.shape)

                # Store most representative points
                mr = np.frombuffer(mr, dtype=np.int)
                # Determine current index - we will map this back to the global index if worker dies
                if self.scenario == 2:
                    self.watch_dog.states[worker].most_representative = self.watch_dog.states[worker].lower_bound + mr
                    self._logger.debug(f"Min mr={np.min(self.watch_dog.states[worker].most_representative)}, Max mr={np.max(self.watch_dog.states[worker].most_representative)}")
                else:
                    self.watch_dog.states[worker].most_representative = np.min(self.watch_dog.states[worker].idxs) + mr
                    

                # Decode loss
                epoch_loss_temp = float(epoch_loss_temp.decode())

                # Weight parameters and loss
                d_theta += beta * d_theta_temp              
                epoch_loss += beta * epoch_loss_temp

                workers_received.add(worker)

                i += 1
                running_time = 0

        # Average parameters
        # d_theta /= len(self.workers)
        # epoch_loss /= len(self.workers)
        # self._logger.debug(f"Len worker={len(self.workers)}, i-1={i-1}")
        assert i > 0
        assert i > 0
        i -= n_connected
        d_theta /= i
        epoch_loss /= i

        self._logger.debug("Calculated gradients")
        
        return d_theta, epoch_loss