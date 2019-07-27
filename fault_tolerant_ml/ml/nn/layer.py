import numpy as np

class Layer(object):
    
    def __init__(self, n_inputs, n_outputs, W=None, b=None):
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = np.random.randn(self.n_inputs, self.n_outputs) * 0.01
        self.b = np.zeros((1, self.n_outputs))
        
        if W is not None:
            self.W = W
        if b is not None:
            self.b = b
        
    def __call__(self, value):
        
        z = np.dot(value, self.W) + self.b
        return z

    @property
    def shape(self):
        return self.W.shape