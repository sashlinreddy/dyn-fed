"""Coordinator for distributed training
"""
from __future__ import absolute_import, print_function

import logging
import time

import numpy as np
import zmq.green as zmq

from dyn_fed.distribute.states import MAP, MAP_PARAMS, REMAP
from dyn_fed.proto.utils import (parse_params_response_from_string,
                                           setup_to_string)


class Coordinator():
    """Responsible for distributing data
    """
    
    def __init__(self):
        
        self._logger = logging.getLogger(f"dfl.distribute.{self.__class__.__name__}")
        self.labels_per_worker = {}
        self.state = 0

    def _encode(self, params, variables):
        
        return [str(params[i]).encode() for i in variables]

    def _check_timeout(self, running_time, timeout, watch_dog, strategy, workers_received):
        wait = True
        if running_time > timeout:
            self._logger.debug(f"Running time exceeded timeout={running_time}")
            active_workers = set(watch_dog.active_workers)
            # Get workers that we did not receive work from
            diff = active_workers - workers_received
            for w in diff:
                # Set dead workers state to false
                watch_dog.states[w].state = False
                if strategy.remap != 1:                                    
                    watch_dog.states[w].idxs = watch_dog.states[w].most_representative
            
            self.state = REMAP
            wait = False

        return wait

    def bcast(self, socket, data, subscribe_msg=b""):
        """Broadcasts to anyone listening on the socket
        """

        multipart = [b"", subscribe_msg]
        multipart.extend(data)

        socket.send_multipart(multipart)

    def send(self, socket, worker, data, tag=b""):
        """Send to specific rank/worker
        """
        multipart = [worker, tag]
        multipart.extend(data)
        socket.send_multipart(multipart)

    def aggregate(self,
                  d_Wbs,
                  errors,
                  model,
                  samples,
                  n_samples,
                  mode=0,
                  epsilon=1e-8):
        """Aggregate received parameters

        Args:
            d_Wbs (list): List of parameters received
            errors (list): List of losses for each worker
            model (dfl.Model): Model being used
            samples (list): List containing number of samples for each worker
            n_samples (int): Total number of samples
            mode (int): Aggregation mode (default: 0)

        Returns:
            parameters (list): List of numpy matrices for each layer
        """
        # pylint: disable=too-many-arguments, too-many-locals
        # Iterate through workers and weight parameters by corresponding epoch loss
        parameters = [[np.zeros_like(l.W.data), np.zeros_like(l.b.data)] for l in model.layers]
        sum_es = np.sum(np.exp(errors))
        epoch_loss = np.mean(errors)
        for j in np.arange(len(errors)):
            weight = 1.0 / len(errors)  # Default aggregation is the average across workers
            # Weight by loss calculated by worker - worker with highest loss has greater weight
            if mode == 1:
                n_samples_worker = samples[j]
                weight = n_samples_worker / n_samples
            elif mode == 2:
                weight = np.exp(errors[j]) / sum_es
            self._logger.debug(f"worker={j}, weight={weight}, loss={errors[j]}")
            for k in np.arange(model.n_layers):
                parameters[k][0] += (
                    d_Wbs[j][k][0] * weight
                    if weight > 0
                    else d_Wbs[j][k][0] * epsilon
                ) # For W parameter
                parameters[k][1] += (
                    d_Wbs[j][k][1] * weight
                    if weight > 0
                    else d_Wbs[j][k][1] * epsilon
                ) # For b parameter

        return parameters, epoch_loss

    def collect(self, events, socket, params, aggregate_mode=0):
        """Receives gradients from workers

        Args:
            events (dict): Dictionary of events from our poller

        Returns:
            d_W (numpy.ndarray): Our gradient matrix that is aggregated
            with a weighting according to the number of samples each 
            worker has
            epoch_loss (float): The loss for this epoch aggregated from each
            worker, also weighted according to the work each worker did
        """
        self._logger.debug(f"Collecting gradients")
        watch_dog = params["watch_dog"]
        self.state: int = params["state"]
        W: np.ndarray = params["W"]
        epoch_loss: int = 0.0

        self._logger.debug(f"Alive workers={watch_dog.n_alive}")

        i = 0
        running_time = 0
        n_connected = 0

        workers_received = set()

        errors = []
        d_Wbs = []
        samples = []
        parameters = (
            [
                [np.zeros_like(l.W.data), np.zeros_like(l.b.data)]
                for l in params["model"].layers
            ]
        )

        while i < watch_dog.n_alive:

            # Timer to calculate running time for an iteration. We can then
            # calculate the running time for an iteration so that if a state
            # changes since a poll event, we can break if the running time 
            # exceeds the timeout
            start_i = time.time()

            if (socket in events) and (events.get(socket) == zmq.POLLIN):
                try:
                    msg = socket.recv_multipart(zmq.NOBLOCK) # pylint: disable=no-member
                except zmq.ZMQError as e:
                    if e.errno == zmq.EAGAIN: # pylint: disable=no-member
                        # state changed since poll event
                        running_time += time.time() - start_i
                        wait = self._check_timeout(
                            running_time,
                            params["timeout"],
                            watch_dog,
                            params["strategy"],
                            workers_received
                        )
                        if not wait:
                            break

                        continue

                if i == 0:
                    worker, content = msg
                else:
                    # Receive multipart including command message
                    cmd = msg[0]
                    if cmd == b"WORK":
                        worker, content = msg[1:]
                    elif cmd == b"CONNECT":
                        # self.register_workers(msg[1])
                        watch_dog.add_worker(msg[1])
                        n_connected += 1
                        i += 1
                        continue

                # Parse parameters
                if params["quantize"]:
                    self._logger.debug(f"Reconstructing gradients")
                    # shape = W.shape
                    # parameter_temp = reconstruct_approximation(parameter_temp,
                    # shape, r_dtype=W.dtype)
                    worker_params = np.frombuffer(worker_params, dtype=W.dtype)
                    worker_params = worker_params.reshape(W.shape)
                else:
                    parameters, mr, loss = parse_params_response_from_string(content)

                # Determine current index - we will map this back to the global index if worker dies
                if params["strategy"].remap == 2:
                    watch_dog.states[worker].most_representative = \
                        watch_dog.states[worker].lower_bound + mr
                    # self._logger.debug(
                    # f"Min mr={np.min(watch_dog.states[worker].most_representative)}, 
                    # Max mr={np.max(watch_dog.states[worker].most_representative)}")
                else:
                    watch_dog.states[worker].most_representative = \
                        np.min(watch_dog.states[worker].idxs) + mr

                # Accumulate data for each worker
                errors.append(loss)
                d_Wbs.append(parameters)
                samples.append(watch_dog.states[worker].n_samples)

                workers_received.add(worker)

                i += 1
                running_time = 0

        # Aggregate with weighted average
        parameters, epoch_loss = self.aggregate(
            d_Wbs,
            errors,
            params["model"],
            samples,
            params["n_samples"],
            mode=aggregate_mode
        )

        # Average parameters
        assert i > 0
        i -= n_connected

        self._logger.debug("Calculated gradients")
        
        return parameters, epoch_loss, self.state

    def _map_params(self, socket, data, params):
        """Maps model parameters

        Args:
            socket (zmq.Socket): ZMQ socket to push messages to subscribers
            data (numpy.ndarray): Parameter tensor
            params (dict): Additional params to check if need to quantize
        """
        if params["quantize"] == 0:
            multipart = data
            subscribe_msg = b"WORKNODELAY" if params["delay_change"] else b"WORK"

        # Quantized parameters
        elif params["quantize"] == 1:
            self._logger.debug("Distributing quantized parameters")
            # Get message send ready
            msg = data.tostring()
            multipart = [msg]
            subscribe_msg = b"WORK"
            
        self.bcast(socket=socket, data=multipart, subscribe_msg=subscribe_msg)

    def _map(self, socket, data, workers, params, gen_func):
        """Maps data to workers on startup or new worker
        """
        # Distribute data/data indices to work on
        self._logger.debug("Distributor distributing data")
        X_train, y_train = data
        state = params["state"]
        n_samples = params["n_samples"]

        self._logger.debug(f"State={state}")

        batch_size = int(np.ceil(params["n_samples"] / params["n_alive"]))
        batch_gen = gen_func(
            X_train,
            y_train,
            batch_size,
            shuffle=False,
            overlap=params["overlap"]
        )
        
        if "mapping" in params:
            mapping = params["mapping"]

        if params["remap"] != 1:
            self.labels_per_worker = {}

        # Iterate through workers and send
        i = 0
        for worker in workers:

            if worker.state:
                worker.mr_idxs_used = False
                # Get next batch to send
                X_batch, y_batch = next(batch_gen)
                X_batch = X_batch.data
                y_batch = y_batch.data
                self._logger.debug(f"X.shape={X_batch.shape}, y.shape={y_batch.shape}")

                if y_batch.shape[1] > 1:
                    y_b = np.argmax(y_batch, axis=1) 
                else:
                    y_b = y_batch

                if (state == REMAP) and (params["remap"] == 1):
                    classes, dists = np.unique(y_b, return_counts=True)
                    self._logger.debug(f"Classes={classes}")
                    self.labels_per_worker[worker][1][classes] = \
                        self.labels_per_worker[worker][1][classes] + dists
                else:
                    self.labels_per_worker[worker] = np.unique(y_b, return_counts=True)

                # Serialize data
                msg = [setup_to_string(X_batch, y_batch, n_samples, state)]

                # Keep track of samples per worker
                # Redistribute all data points
                if (state == MAP) or params["remap"] == 2:
                    worker.n_samples = X_batch.shape[0]
                    lower_bound = X_batch.shape[0] * i
                    upper_bound = lower_bound + X_batch.shape[0]
                    worker.idxs = np.arange(lower_bound, upper_bound)
                    if worker.most_representative is None:
                        worker.most_representative = np.zeros((params["n_most_rep"],))
                        worker.lower_bound = lower_bound
                        worker.upper_bound = upper_bound
                # Redistribute only most representative data points for dead workers
                elif params["remap"] == 1:
                    worker.n_samples += X_batch.shape[0]
                    lower_bound = X_batch.shape[0] * i
                    upper_bound = lower_bound + X_batch.shape[0]
                    batch_range = np.arange(lower_bound, upper_bound)
                    new_range = np.arange(
                        worker.upper_bound, 
                        worker.upper_bound + batch_range.shape[0]
                    ) 
                    self._logger.debug(
                        f"New range={new_range}, "
                        f"worker max idx={np.max(worker.idxs)}, "
                        f"upper bound={worker.upper_bound}"
                    )
                    worker.upper_bound = worker.upper_bound + batch_range.shape[0]
                    if not worker.mapping:
                        worker.mapping = dict(zip(worker.idxs, worker.idxs))
                    
                    self._logger.debug(f"Batch range shape={batch_range}, i={i}")
                    global_idxs = [mapping.get(j) for j in batch_range]
                    self._logger.debug(f"global idxs={global_idxs}, i={i}")
                    worker.mapping.update(dict(zip(new_range, global_idxs)))
                    worker.idxs = np.hstack((worker.idxs, global_idxs))
                    if worker.most_representative is None:
                        worker.most_representative = np.zeros((params["n_most_rep"],))

                self.send(
                    socket=socket,
                    worker=worker.identity,
                    data=msg,
                    tag=b"WORK"
                )

                i += 1

        self._logger.debug(
            f"Worker ranges={[(np.min(w.idxs), np.max(w.idxs)) for w in workers]}"
        )
        self._logger.debug(f"Labels per worker={self.labels_per_worker}")

    def map(self, socket, data, workers, params, gen_func=None):
        """Sends the data to the necessary destination
                
        Args:
            socket (zmq.Socket): ZMQ socket to push messages to subscribers
            data (numpy.ndarray): Data matrix that will be partitioned and
            distributed to each worker
            workers (distribute.WorkerStates): worker state objects containing
            state of worker and other info
            params (dict): Additional params to send to all workers
            gen_func (generator func): A generator function to distribute the data
        """
        
        state = params["state"]

        # Publish parameters
        if state == MAP_PARAMS:
            self._map_params(socket, data, params)
        else:
            self._map(socket, data, workers, params, gen_func)
