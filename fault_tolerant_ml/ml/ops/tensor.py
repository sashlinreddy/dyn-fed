"""This module contains all tensor ops to build the computation graph
"""

import numpy as np

# -----------------------------------------------------------------------------
# Graph
# -----------------------------------------------------------------------------

class Graph():
    
    
    def __init__(self):
        
        self.operations = []
        self.variables = []
        
    def set_as_default(self):
        """
        Sets this Graph instance as the Global Default Graph
        """
        global _default_graph
        _default_graph = self

# -----------------------------------------------------------------------------
# Tensor
# -----------------------------------------------------------------------------

class Tensor():
    """
    This variable is a changeable parameter of the Graph.
    """
    
    def __init__(self, initial_value = None):
        
        self.value = initial_value
        self.output_nodes = []
         
        _default_graph.variables.append(self)
        
    def __repr__(self):
        return f"Tensor({self.value}, dtype={self.value.dtype})"
    
    def __str__(self):
        return f"Tensor({self.value}, dtype={self.value.dtype})"

    @property
    def T(self):
        return self.value.T
    
    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return self.value + other.value

    def __iadd__(self, other):
        return self.value + other.value

    def __sub__(self, other):
        return self.value - other.value

    def __rsub(self, other):
        return self.value - other.value

    def __isub__(self, other):
        return self.value - other.value

    def __mul__(self, other):
        return self.value * other.value

    def __rmul__(self, other):
        return self.value * other.value

    def __imul__(self, other):
        return self.value * other.value

    def __matmul__(self, other):
        return self.value @ other.value

    def __truediv__(self, other):
        return self.value / other.value

    def __neg__(self):
        return -self.value



# -----------------------------------------------------------------------------
# Ops
# -----------------------------------------------------------------------------

class Operation(object):
    """
    An Operation is a node in a "Graph". TensorFlow will also use this concept of a Graph.
    
    This Operation class will be inherited by other classes that actually compute the specific
    operation, such as adding or matrix multiplication.
    """
    
    def __init__(self, input_nodes = []):
        """
        Intialize an Operation
        """
        self.input_nodes = input_nodes # The list of input nodes
        self.output_nodes = [] # List of nodes consuming this node's output
        
        # For every node in the input, we append this operation (self) to the list of
        # the consumers of the input nodes
        for node in input_nodes:
            node.output_nodes.append(self)
        
        # There will be a global default graph (TensorFlow works this way)
        # We will then append this particular operation
        # Append this operation to the list of operations in the currently active default graph
        _default_graph.operations.append(self)
        
    def __repr__(self):
        return f"{self.__class__.__name__}()"
  
    def compute(self):
        """ 
        This is a placeholder function. It will be overwritten by the actual specific operation
        that inherits from this class.
        
        """
        
        raise NotImplementedError

# -----------------------------------------------------------------------------
# Ops child classes
# -----------------------------------------------------------------------------

class add(Operation):
    
    def __init__(self, a, b):
         
        super().__init__([a, b])

    def compute(self, a, b):
         
        self.inputs = [a, b]
        return a.value + b.value

class subtract(Operation):
    
    def __init__(self, a, b):
         
        super().__init__([a, b])

    def compute(self, a, b):
         
        self.inputs = [a, b]
        return a.value - b.value

class multiply(Operation):
     
    def __init__(self, a, b):
        
        super().__init__([a, b])
    
    def compute(self, a, b):
         
        self.inputs = [a, b]
        return a.value * b.value

class divide(Operation):
     
    def __init__(self, a, b):
        
        super().__init__([a, b])
    
    def compute(self, a, b):
         
        self.inputs = [a, b]
        return a.value / b.value

class matmul(Operation):
     
    def __init__(self, a, b):
        
        super().__init__([a, b])
    
    def compute(self, a, b):
         
        self.inputs = [a, b]
        return a.value.dot(b.value)

class neg(Operation):
     
    def __init__(self, a):
        
        super().__init__([a])
    
    def compute(self, a):
         
        self.inputs = [a]
        return -a.value

class square(Operation):
     
    def __init__(self, a):
        
        super().__init__([a])
    
    def compute(self, a):
         
        self.inputs = [a]
        return a.value * a.value

class pow(Operation):
     
    def __init__(self, a, p):
        
        super().__init__([a, p])
    
    def compute(self, a, p):
         
        self.inputs = [a, p]
        return np.power(a.value, p)