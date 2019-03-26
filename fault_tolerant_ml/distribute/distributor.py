import logging
import zmq.green as zmq
import numpy as np

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
        self._logger.debug("Distributing data")
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

        # Iterate through workers and send
        i = 0
        for worker in workers:

            if worker.state:
                worker.mr_idxs_used = False
                # Get next batch to send
                X_batch, y_batch = next(batch_gen)
                self._logger.debug(f"X.shape={X_batch.shape}, y.shape={y_batch.shape}")
                batch_data = np.hstack((X_batch, y_batch))

                # Encode data
                dtype = batch_data.dtype.str.encode()
                shape = str(batch_data.shape).encode()
                msg = batch_data.tostring()

                # Keep track of samples per worker
                worker.n_samples = X_batch.shape[0]
                lower_bound = X_batch.shape[0] * i
                upper_bound = lower_bound + X_batch.shape[0]
                worker.idxs = np.arange(lower_bound, upper_bound)
                if worker.most_representative is None:
                    worker.most_representative = np.zeros((params["n_most_representative"],))
                    worker.lower_bound = lower_bound
                    worker.upper_bound = upper_bound

                socket.send_multipart([worker.identity, b"WORK", batch_data, dtype, shape])
                socket.send_multipart([worker.identity, b"WORK", n_samples, n_features, n_classes, scenario, n_most_representative, learning_rate, delay])
                i += 1
