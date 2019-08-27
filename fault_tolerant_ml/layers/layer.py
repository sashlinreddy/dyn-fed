"""Module that implements layer logic
"""
import logging

import numpy as np

from fault_tolerant_ml.activations.activation import Linear, ReLU, Sigmoid
from fault_tolerant_ml.operators import Tensor


class Layer(object):
    """Layer class

    Attributes:
        n_inputs (int): No. of inputs
        n_output (int): No. of outputs
        W (ftml.Tensor): Parameter tensor
        b (ftml.Tensor): Bias vector
        activation (str): Activation function
        dtype (type): Type of tensors
    """
    def __init__(self, n_inputs, n_outputs, W=None, b=None, activation="sigmoid", dtype=np.float32):
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = Tensor(
            np.random.randn(self.n_inputs, self.n_outputs).astype(dtype) * 0.01,
            is_param=True
        )
        self.b = Tensor(np.zeros((1, self.n_outputs)).astype(dtype), is_param=True)

        self._logger = logging.getLogger(f'ftml.layers.{self.__class__.__name__}')
        
        if W is not None:
            self.W = W
        if b is not None:
            self.b = b

        self.x = None
        self.z = None
        self.y = None

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
        """Perform feedforward for layer given input

        Args:
            value (ftml.Tensor): Input tensor to perform feedforward on

        Returns:
            y (ftml.Tensor): Output tensor
        """
        self.x, self.z, self.y = self._forward(value)
        
        return self.y

    def _forward(self, x):
        # Store input tensor for feedforward
        x = x
        # Store edge
        z = (x @ self.W) + self.b
        
        # Store output tensor for feedforward
        y = Tensor(self.act_fn(z.data))

        assert np.array_equal(z.data, y.data)
        
        return x, z, y
