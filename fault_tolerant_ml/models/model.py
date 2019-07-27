"""This file contains all logic for the base class of models. For implementation of logistic regression or neural networks,
you should inherit from this class.
"""

from fault_tolerant_ml.operators import Tensor
from fault_tolerant_ml.layers import Layer

def _check_layers(layers):
    """Check if all objects in list are of Layer type

    Args:
        layers (list): List of fault_tolerant_ml.layers.Layer objects

    Returns:
        valid (bool): Whether or not all layers are valid
    """
    valid = True
    type_found = None
    for layer in layers:
        if not isinstance(layer, Layer):
            valid = False
            type_found = type(layer)
            break

    return valid, type_found

class Model():
    """
    Base class for tensor models.

    To create your own model, use as follows:

    Example:
        class LogisticRegression(fault_tolerant.Model)

    Attributes:
        layers (list): List of fault_tolerant_ml.layers.Layer objects
    """
    def __init__(self):
        self.layers = []
        
    def __repr__(self):
        return f"Model([{self.layers}])"
        
    def add(self, layers):
        """Add new layer(s) to model
        """

        if not isinstance(layers, list):
            layers = [layers]
        valid, type_found = _check_layers(layers)
        assert valid, f"All layers should be of type Layer, found {type_found} instead"
        self.layers += layers
            
    def zero_grad(self):
        for layer in self.layers:
            for k, v in layer.__dict__.items():
                if isinstance(v, Tensor):
                    if v.is_param:
                        v.zero_grad()
            
    def forward(self, x):
        
        input_layer = x
        for layer in self.layers:
            y_pred = layer(input_layer)
            input_layer = layer.y
            
        return y_pred