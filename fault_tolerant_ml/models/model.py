from fault_tolerant_ml.operators import Tensor

class Model():
    """
    Base class for tensor models
    """
    def __init__(self):
        self.layers = []
        
    def __repr__(self):
        return f"Model([{self.layers}])"
        
    def parameters(self):
        for layer in self.layers:
            for key, value in layer.parameters.items():
                yield value
        
    def add(self, layers):
        if isinstance(layers, list):
            self.layers += layers
        else:
            self.layers.append(layers)
            
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