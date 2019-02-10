import numpy as np

class BaseData(object):

    def __init__(self, filepath):
        self.filepath = filepath

    def prepare_data(self):
        raise NotImplementedError("Child required to implement prepare data method")

    def next_batch(self, batch_size):
        """
        Returns a batch of the data of size batch_size. If the
        batch_size=1 then it is normal stochastic gradient descent
        Args:
            X: Entire feature dataset
            y: Entire label dataset
            batch_size: selected batch_size; default: 1

        Returns:
            X_batch: random batch of X
            y_batch: random batch of y
        """

        # loop over our dataset `X` in mini-batches of size `batchSize`
        for i in np.arange(0, X.shape[0], batch_size):
            # yield a tuple of the current batched data and labels
            yield (X[i:i + batch_size], y[i:i + batch_size])