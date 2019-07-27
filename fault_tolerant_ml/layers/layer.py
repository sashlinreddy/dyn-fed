import numpy as np
from fault_tolerant_ml.activations.activation import Sigmoid, ReLU
from fault_tolerant_ml.operators import Tensor

class Layer(object):
    
    def __init__(self, n_inputs, n_outputs, W=None, b=None, activation="sigmoid", dtype=np.float64):
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = Tensor(np.random.randn(self.n_inputs, self.n_outputs).astype(dtype) * 0.01, is_param=True)
        self.b = Tensor(np.zeros((1, self.n_outputs)).astype(dtype), is_param=True)
        
        if W is not None:
            self.W = W
        if b is not None:
            self.b = b

        self.activation = activation
        self.act_fn = None
        if self.activation == "relu":
            self.act_fn = ReLU()
        else:
            # Default to sigmoid
            self.act_fn = Sigmoid()

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
        y = Tensor(self.act_fn(self.z.data))

        return x, z, y