import logging

class Distributor(object):
    """Responsible for distributing data
    
    Long description
    
    Attributes:
        attrib1 (type): Short description of attribute
    """
    
    def __init__(self):
        
        self._logger = logging.getLogger("ftml")

    def distribute(self, data, batch_size, destinations):
        """Sends the data to the necessary destination
        
        Long description
        
        Args:
            arg1 (type): Short description of attribute
        
        Returns:
            arg1 (type): Short description of attribute that's returned
        """
        
        # Distribute data/data indices to work on
        self.logger.debug("Distributing data")
        batch_size = int(np.ceil(self.data.n_samples / self.watch_dog.n_alive))
        batch_gen = self.data.next_batch(self.X_train, self.y_train, batch_size)

        # Encode to bytes
        n_samples = str(self.data.n_samples).encode()
        n_features = str(self.data.n_features).encode()
        n_classes = str(self.data.n_classes).encode()
        scenario = str(self.scenario).encode()
        n_most_representative = str(self.n_most_representative).encode()
        alpha = str(self.alpha).encode()
        delay = str(self.delay).encode()

        # Iterate through workers and send
        i = 0
        for worker in self.watch_dog.states:

            if worker.state:
                worker.mr_idxs_used = False
                # Get next batch to send
                X_batch, y_batch = next(batch_gen)
                self.logger.debug(f"X.shape={X_batch.shape}, y.shape={y_batch.shape}")
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
                    worker.most_representative = np.zeros((self.n_most_representative,))
                    worker.lower_bound = lower_bound
                    worker.upper_bound = upper_bound

                self.ctrl_socket.send_multipart([worker.identity, b"WORK", batch_data, dtype, shape])
                self.ctrl_socket.send_multipart([worker.identity, b"WORK", n_samples, n_features, n_classes, scenario, n_most_representative, alpha, delay])
                i += 1
