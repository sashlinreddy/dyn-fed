import logging
import zmq.green as zmq
import numpy as np
import time
from fault_tolerant_ml.distribute.states import *
from fault_tolerant_ml.utils.maths import reconstruct_approximation
from fault_tolerant_ml.utils import zhelpers

class Coordinator(object):
    """Responsible for distributing data
    """
    
    def __init__(self):
        
        self._logger = logging.getLogger(f"ftml.distribute.{self.__class__.__name__}")
        self.labels_per_worker = {}

    def _encode(self, params, vars):
        
        return [str(params[i]).encode() for i in vars]

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

    def _decode_msg(self, worker_params):
        """Decode parameters received from workers

        Args:
            worker_params (list of byte strings): W and b params received from corresponding worker

        Returns:
            W (np.ndarray): Parameter matrix as numpy array
            b (np.ndarray): Bias matrix
        """
        # Get data for correponding layer
        Wdata, Wdtype, Wshape, bdata, bdtype, bshape = worker_params

        W = zhelpers.reconstruct_array(Wdata, Wdtype, Wshape)
        b = zhelpers.reconstruct_array(bdata, bdtype, bshape)

        return [W, b]

    def _decode_params(self, model, parameters, worker_params, n_items=6):
        """Collect all parameters across workers to master and decode

        Args:
            model (ftml.Model): Fault tolerant model
            parameters (np.ndarray): Tensor where we store the collected messages
            worker_params (byte string): Multipart message received from worker
            n_items (int): Length of message received from worker that we need to decode (default: 6). 3 messages for the 
            W tensor (data, dtype, shape) and 3 for the bias vector (data, dtype, shape)

        Returns:
            parameters (np.ndarray): Populated parameter tensor
        """

        # Decode multipart message for each layer
        for j, k in zip(np.arange(model.n_layers), np.arange(0, len(worker_params), n_items)):
            
            W, b = self._decode_msg(worker_params[k:k+n_items])
            
            parameters[j][0] = W
            parameters[j][1] = b

        return parameters

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

    def aggregate(self, d_Wbs, errors, model, samples, n_samples, by_loss=False):
        epsilon = 1e-8
        # Iterate through workers and weight parameters by corresponding epoch loss
        parameters = [[np.zeros_like(l.W.data), np.zeros_like(l.b.data)] for l in model.layers]
        n_received_workers = len(errors)
        # n_sample_workers = int(np.ceil(n_received_workers * c))
        # sample_workers = np.random.choice(np.arange(n_received_workers), n_sample_workers)
        workers = np.arange(n_received_workers)
        # sum_es = np.sum([np.exp(errors[i]) for i in sample_workers])
        sum_es = np.sum(np.exp(errors))
        for j in workers:
            weight = 1.0 / n_received_workers  # Default aggregation is the average across workers
            # Weight by loss calculated by worker - worker with highest loss has greater weight
            if by_loss:
                weight = np.exp(errors[j]) / sum_es
            n_samples_worker = samples[j]
            samples_weight = n_samples_worker / n_samples
            self._logger.debug(f"samples_weight={samples_weight}")
            # weight *= samples_weight
            self._logger.debug(f"worker={j}, weight={weight}, loss={errors[j]}")
            for k in np.arange(model.n_layers):
                parameters[k][0] += d_Wbs[j][k][0] * weight if weight > 0 else d_Wbs[j][k][0] * epsilon # For W parameter
                parameters[k][1] += d_Wbs[j][k][1] * weight if weight > 0 else d_Wbs[j][k][1] * epsilon # For b parameter
                # Add some randomness 
                # parameters[k][0] += (np.random.randn(parameters[k][0].shape[0], parameters[k][0].shape[1]) *  samples_weight)
                # parameters[k][1] += (np.random.randn(parameters[k][1].shape[0], parameters[k][1].shape[1]) * samples_weight)
                # parameters[k][0] /= n_received_workers
                # parameters[k][1] /= n_received_workers

        return parameters

    def collect(self, events, socket, params, weight_by_loss=False):
        """Receives gradients from workers

        Args:
            events (dict): Dictionary of events from our poller

        Returns:
            d_W (numpy.ndarray): Our gradient matrix that is aggregated with a weighting according to the number    of samples each worker has
            epoch_loss (float): The loss for this epoch aggregated from each worker, also weighted according to the     work each worker did
        """
        watch_dog = params["watch_dog"]
        strategy = params["strategy"]
        self.state: int = params["state"]
        n_samples: int = params["n_samples"]
        timeout = params["timeout"] # We give x seconds to poll worker if state changed since poll event
        quantize: bool = params["quantize"]
        W: np.ndarray = params["W"]
        model = params["model"]
        
        # parameters: np.ndarray = np.zeros_like(W)
        epoch_loss: int = 0.0

        self._logger.debug(f"Receiving gradients")
        n_alive_workers = watch_dog.n_alive
        self._logger.debug(f"Alive workers={n_alive_workers}")

        i = 0
        running_time = 0
        n_connected = 0

        workers_received = set()

        errs = []
        d_Wbs = []
        samples = []
        parameters = [[np.zeros_like(l.W.data), np.zeros_like(l.b.data)] for l in model.layers]

        while i < n_alive_workers:

            # Timer to calculate running time for an iteration. We can then calculate the running time for 
            # an iteration so that if a state changes since a poll event, we can break if the running time 
            # exceeds the timeout
            start_i = time.time()

            if (socket in events) and (events.get(socket) == zmq.POLLIN):
                try:
                    msg = socket.recv_multipart(zmq.NOBLOCK) # pylint: disable=no-member
                except zmq.ZMQError as e:
                    if e.errno == zmq.EAGAIN: # pylint: disable=no-member
                        # state changed since poll event
                        running_time += time.time() - start_i
                        wait = self._check_timeout(running_time, timeout, watch_dog, strategy, workers_received)
                        if not wait:
                            break

                        continue

                if i == 0:
                    worker, epoch_loss_temp, mr = msg[0:3]
                    worker_params = msg[3:]
                else:
                    # Receive multipart including command message
                    cmd = msg[0]
                    if cmd == b"WORK":
                        worker, epoch_loss_temp, mr = msg[1:4]
                        worker_params = msg[4:]
                    elif cmd == b"CONNECT":
                        # self.register_workers(msg[1])
                        watch_dog.add_worker(msg[1])
                        n_connected += 1
                        i += 1
                        continue

                # Calculate weighting
                samples_for_worker = watch_dog.states[worker].n_samples
                # beta = (samples_for_worker / n_samples)
                beta = 1
                # beta = 1 / n_samples
                # beta = samples_for_worker

                if quantize:
                    self._logger.debug(f"Reconstructing gradients")
                    # shape = W.shape
                    # parameter_temp = reconstruct_approximation(parameter_temp, shape, r_dtype=W.dtype)
                    worker_params = np.frombuffer(worker_params, dtype=W.dtype)
                    worker_params = worker_params.reshape(W.shape)
                else:
                    # Decode parameters
                    parameters = self._decode_params(model, parameters, worker_params)

                # Store most representative points
                mr = np.frombuffer(mr, dtype=np.int)
                # Determine current index - we will map this back to the global index if worker dies
                if strategy.remap == 2:
                    watch_dog.states[worker].most_representative = watch_dog.states[worker].lower_bound + mr
                    # self._logger.debug(f"Min mr={np.min(watch_dog.states[worker].most_representative)}, Max mr={np.max(watch_dog.states[worker].most_representative)}")
                else:
                    watch_dog.states[worker].most_representative = np.min(watch_dog.states[worker].idxs) + mr
                    
                # Decode loss
                epoch_loss_temp = float(epoch_loss_temp.decode())

                # Aggregate loss
                epoch_loss += beta * epoch_loss_temp

                if weight_by_loss:
                    # Accumulate epoch loss for each worker
                    errs.append(epoch_loss_temp)
                    # Accumulate params for each worker
                    d_Wbs.append(parameters)
                    # Accumulate samples for each worker
                    samples.append(samples_for_worker)

                workers_received.add(worker)

                i += 1
                running_time = 0

        if weight_by_loss:
            # Aggregate with weighted average
            parameters = self.aggregate(d_Wbs, errs, model, samples, n_samples, by_loss=weight_by_loss)

        # Average parameters
        assert i > 0
        i -= n_connected
        epoch_loss /= i

        self._logger.debug("Calculated gradients")
        
        return parameters, epoch_loss

    def map(self, socket, data, workers, params, gen_func=None):
        """Sends the data to the necessary destination
        
        Long description
        
        Args:
            data (numpy.ndarray): Data matrix that will be partitioned and distributed to each worker
            workers (distribute.WorkerStates): worker state objects containing state of worker and other info
            params (dict): Additional params to send to all workers
            gen_func (generator func): A generator function to distribute the data
        """
        
        state = params["state"]

        # Publish parameters
        if state == DIST_PARAMS:
                
            if params["quantize"] == 0:
                # Get message send ready
                # msg = data.tostring()
                # dtype = data.dtype.str.encode()
                # shape = str(data.shape).encode()
                # multipart = [msg, dtype, shape]
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

        else:
            # Distribute data/data indices to work on
            self._logger.debug("Distributor distributing data")
            X_train, y_train = data
            batch_size = int(np.ceil(params["n_samples"] / params["n_alive"]))
            batch_gen = gen_func(X_train, y_train, batch_size, shuffle=False)

            # Encode to bytes
            enc_vars = [
                "n_samples", "n_features", "n_classes"
            ]
            multipart_params = self._encode(params, enc_vars)

            state = params["state"]

            self._logger.debug(f"State={state}")
            
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
                    batch_data = np.hstack((X_batch, y_batch))

                    if y_batch.shape[1] > 1:
                        y_b = np.argmax(y_batch, axis=1) 
                    else:
                        y_b = y_batch

                    if (state == REMAP) and (params["remap"] == 1):
                        classes, dists = np.unique(y_b, return_counts=True)
                        self._logger.debug(f"Classes={classes}")
                        self.labels_per_worker[worker][1][classes] = self.labels_per_worker[worker][1][classes] + dists
                    else:
                        self.labels_per_worker[worker] = np.unique(y_b, return_counts=True)

                    # Encode data
                    dtype = batch_data.dtype.str.encode()
                    shape = str(batch_data.shape).encode()
                    msg = batch_data.tostring()

                    # Keep track of samples per worker
                    # Redistribute all data points
                    if (state == MAP) or params["remap"] != 1:
                        worker.n_samples = X_batch.shape[0]
                        lower_bound = X_batch.shape[0] * i
                        upper_bound = lower_bound + X_batch.shape[0]
                        worker.idxs = np.arange(lower_bound, upper_bound)
                        if worker.most_representative is None:
                            worker.most_representative = np.zeros((params["n_most_rep"],))
                            worker.lower_bound = lower_bound
                            worker.upper_bound = upper_bound
                    # Redistribute only most representative data points for dead workers
                    else:
                        worker.n_samples += X_batch.shape[0]
                        lower_bound = X_batch.shape[0] * i
                        upper_bound = lower_bound + X_batch.shape[0]
                        batch_range = np.arange(lower_bound, upper_bound)
                        new_range = np.arange(worker.upper_bound, worker.upper_bound + batch_range.shape[0]) 
                        self._logger.debug(f"New range={new_range}, worker max idx={np.max(worker.idxs)}, upper bound={worker.upper_bound}")
                        worker.upper_bound = worker.upper_bound + batch_range.shape[0]
                        if not worker.mapping:
                            worker.mapping = dict(zip(worker.idxs, worker.idxs))
                        
                        self._logger.debug(f"Batch range shape={batch_range}, i={i}")
                        global_idxs = [mapping.get(j) for j in batch_range]
                        # self._logger.debug(f"global idxs={global_idxs}, i={i}")
                        worker.mapping.update(dict(zip(new_range, global_idxs)))
                        worker.idxs = np.hstack((worker.idxs, global_idxs))
                        if worker.most_representative is None:
                            worker.most_representative = np.zeros((params["n_most_rep"],))

                    multipart_data = [batch_data, dtype, shape]

                    self.send(socket=socket, worker=worker.identity, data=multipart_data, tag=b"WORK")
                    self.send(socket=socket, worker=worker.identity, data=multipart_params, tag=b"WORK")

                    i += 1

            self._logger.debug(f"Worker ranges={[(np.min(w.idxs), np.max(w.idxs)) for w in workers]}")

            self._logger.debug(f"Labels per worker={self.labels_per_worker}")