import numpy as np
import logging
from fault_tolerant_ml.activations.activation import Sigmoid, ReLU, Linear
from fault_tolerant_ml.operators import Tensor

class Layer(object):
    
    def __init__(self, n_inputs, n_outputs, W=None, b=None, activation="sigmoid", dtype=np.float32):
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = Tensor(np.random.randn(self.n_inputs, self.n_outputs).astype(dtype) * 0.01, is_param=True)
        self.b = Tensor(np.zeros((1, self.n_outputs)).astype(dtype), is_param=True)

        self._logger = logging.getLogger(f'ftml.layers.{self.__class__.__name__}')
        
        if W is not None:
            self.W = W
        if b is not None:
            self.b = b

        self.activation = activation
        self.act_fn = None
        if self.activation == "relu":
            self.act_fn = ReLU()
        
        elif self.activation == "sigmoid":
            self.act_fn = Sigmoid()
        else:
            # Default to linear
            self.act_fn = Linear()

    def __repr__(self):
        return f"Layer({self.n_inputs}, {self.n_outputs})"
        
    def __call__(self, value):
        
        # Store input tensor for feedforward
        # self.x = value
        # # Store edge
        # self.z = (value @ self.W) + self.b
        # # Store output tensor for feedforward
        # self.y = Tensor(self.act_fn(self.z.data))

        self.x, self.z, self.y = self._forward(value)
        
        return self.y

    def _forward(self, x, retain_derived=True):

        # Store input tensor for feedforward
        x = x
        # Store edge
        z = (x @ self.W) + self.b
        
        # Store output tensor for feedforward
        y = Tensor(self.act_fn(z.data))

        assert np.array_equal(z.data, y.data)
        
        return x, z, y