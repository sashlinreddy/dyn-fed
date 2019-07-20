class Model(object):
    """Base classes for neural networks
    """
    def __init__(self):
        
        raise NotImplementedError("Child should override")
    
    def forward(self, x):
        """Forward pass
        """
        raise NotImplementedError("Child should override")