import numpy as np
from sklearn.model_selection import train_test_split
class Dataset(object):

    def __init__(self, filepath):
        self.filepath = filepath
        self.X = None
        self.y = None

    @staticmethod
    def one_hot(labels):
        """Returns one hot encoded representation of data 
        
        Args:
            labels (numpy.ndarray): nx1 numpy vector of labels to be one-hot encoded where
                n is the number of samples in the dataset
        
        Returns:
            one_hot (numpy.ndarray): nxd one-hot encoded numpy matrix where d is the number     of unique labels in the 1d numpy vector
        """
        unique_labels = np.unique(labels)
        return (unique_labels == labels[:, np.newaxis]).astype(int)

    @staticmethod
    def next_batch(X, y, batch_size, shuffle=True):
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

        if shuffle:
            idxs = np.arange(X.shape[0])
            np.random.shuffle(idxs)
            X = X[idxs]
            y = y[idxs]

        # loop over our dataset `X` in mini-batches of size `batchSize`
        for i in np.arange(0, X.shape[0], batch_size):
            # yield a tuple of the current batched data and labels
            yield (X[i:i + batch_size], y[i:i + batch_size])

    def prepare_data(self):
        raise NotImplementedError("Child required to implement prepare data method")

    def update_xy(self, idxs):
        X_train = self.X_train[idxs]
        y_train = self.y_train[idxs]
        self.n_samples = X_train.shape[0]
        return X_train, y_train

    def do_split(self, X, y, test_size=0.3):
        """
        Returns a dictionary of the train test split numpy arrays
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        data = {'training' : {'features' : X_train, 'labels' : y_train},
                'testing' : {'features' : X_test, 'labels' : y_test}}

        return data