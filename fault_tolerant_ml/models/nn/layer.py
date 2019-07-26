import numpy as np
from fault_tolerant_ml.activations.activation import Sigmoid, ReLU

class Layer(object):
    
    def __init__(self, n_inputs, n_outputs, W=None, b=None, activation="sigmoid"):
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = np.random.randn(self.n_inputs, self.n_outputs) * 0.01
        self.b = np.zeros((1, self.n_outputs))
        
        if W is not None:
            self.W = W
        if b is not None:
            self.b = b

        self.parameters = {"W": W, "b": b}
        self.derived_vars = {"Z": []}
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}

        self.activation = activation
        self.act_fn = None
        if self.activation == "relu":
            self.act_fn = ReLU()
        else:
            # Default to sigmoid
            self.act_fn = Sigmoid()

    def __repr__(self):
        return f"Layer(n_inputs={self.n_inputs}, n_outputs={self.n_outputs})"
        
    def __call__(self, value):
        
        self.z = np.dot(value, self.W) + self.b
        # self.y = self.act_fn(self.z)
        return self.z

    def _forward(self, X, retain_derived=True):

        pass

    @property
    def shape(self):
        return self.W.shape