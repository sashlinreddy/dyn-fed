import logging
import zmq.green as zmq
import numpy as np

from .states import *
class Distributor(object):
    """Responsible for distributing data
    
    Long description
    
    Attributes:
        gen_func (generator func): A generator function to distribute the data
    """
    
    def __init__(self, gen_func):
        
        self._logger = logging.getLogger("ftml")
        self.gen_func = gen_func

    def distribute(self, socket, data, workers, params):
        """Sends the data to the necessary destination
        
        Long description
        
        Args:
            data (numpy.ndarray): Data matrix that will be partitioned and distributed to each worker
            workers (distribute.WorkerStates): worker state objects containing state of worker and other info
            params (dict): Additional params to send to all workers
        """
        
        # Distribute data/data indices to work on
        self._logger.debug("Distributor distributing data")
        X_train, y_train = data
        batch_size = int(np.ceil(params["n_samples"] / params["n_alive"]))
        batch_gen = self.gen_func(X_train, y_train, batch_size)

        # Encode to bytes
        n_samples = str(params["n_samples"]).encode()
        n_features = str(params["n_features"]).encode()
        n_classes = str(params["n_classes"]).encode()
        scenario = str(params["scenario"]).encode()
        n_most_representative = str(params["n_most_representative"]).encode()
        learning_rate = str(params["learning_rate"]).encode()
        delay = str(params["delay"]).encode()

        state = params["state"]

        self._logger.debug(f"State={state}")
        
        if "mapping" in params:
            mapping = params["mapping"]

        labels_per_worker = {}

        # Iterate through workers and send
        i = 0
        for worker in workers:

            if worker.state:
                worker.mr_idxs_used = False
                # Get next batch to send
                X_batch, y_batch = next(batch_gen)
                self._logger.debug(f"X.shape={X_batch.shape}, y.shape={y_batch.shape}")
                batch_data = np.hstack((X_batch, y_batch))

                labels_per_worker[worker] = np.unique(y_batch, return_counts=True)

                # Encode data
                dtype = batch_data.dtype.str.encode()
                shape = str(batch_data.shape).encode()
                msg = batch_data.tostring()

                # Keep track of samples per worker
                # Redistribute all data points
                if (state == MAP) or params["scenario"] != 2:
                    worker.n_samples = X_batch.shape[0]
                    lower_bound = X_batch.shape[0] * i
                    upper_bound = lower_bound + X_batch.shape[0]
                    worker.idxs = np.arange(lower_bound, upper_bound)
                    if worker.most_representative is None:
                        worker.most_representative = np.zeros((params["n_most_representative"],))
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
                        worker.most_representative = np.zeros((params["n_most_representative"],))

                socket.send_multipart([worker.identity, b"WORK", batch_data, dtype, shape])
                socket.send_multipart([worker.identity, b"WORK", n_samples, n_features, n_classes, scenario, n_most_representative, learning_rate, delay])
                i += 1

        self._logger.debug(f"Worker ranges={[(np.min(w.idxs), np.max(w.idxs)) for w in workers]}")

        self._logger.debug(f"Labels per worker={labels_per_worker}")