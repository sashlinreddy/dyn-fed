from typing import Iterator

class Model(object):
    """Base classes for neural networks
    """
    def __init__(self):
        
        self.layers = []

    def parameters(self):
        
        pass
        # for layer in self.layers:
        #     yield 
    
    def forward(self, x):
        """Forward pass
        """
        raise NotImplementedError("Child should override")