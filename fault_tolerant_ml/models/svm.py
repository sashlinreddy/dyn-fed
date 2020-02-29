"""SVM Model implementation
"""
from fault_tolerant_ml.layers import Layer
from fault_tolerant_ml.models import ModelV2

class SVM(ModelV2):
    """SVM model
    """
    def __init__(
            self,
            optimizer,
            strategy=None,
            batch_size=64,
            max_iter=300,
            shuffle=True,
            verbose=10,
            **kwargs):
        super().__init__(
            optimizer=optimizer,
            strategy=strategy, 
            batch_size=batch_size,
            max_iter=max_iter, 
            shuffle=shuffle, 
            verbose=verbose, 
            **kwargs
        )
        self._kernel = kwargs.get("kernel", "linear") # Default kernel is linear
        self._n_inputs = kwargs.get("n_inputs")
        self._n_outputs = kwargs.get("n_outputs")

        self.add(Layer(self._n_inputs, self._n_outputs, activation="linear"))

    @property
    def kernel(self):
        """Kernel getter
        """
        return self._kernel

    @property
    def n_inputs(self):
        """Kernel getter
        """
        return self._n_inputs

    @property
    def n_outputs(self):
        """Kernel getter
        """
        return self._n_outputs
