import numpy as np
from copy import deepcopy
from typing import Union

def ensure_array(arrayable):
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)

Tensorable = Union['Tensor', 'float', np.ndarray]

def ensure_tensor(tensorable: Tensorable):
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)
    
class Tensor():
    
    def __init__(self, data, is_param=False):
        self.data = ensure_array(data)
        self._is_param = is_param
        self._grad = None
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        
        if is_param:
            self.zero_grad()
        
    def __repr__(self):
        return f"Tensor({self.data}, parameter={self.is_param})"

    def __str__(self):
        return f"Tensor({self.data}, parameter={self.is_param})"
    
    def __matmul__(self, other):
        return Tensor(self.data @ ensure_tensor(other).data)
    
    def __add__(self, other):
        return Tensor(self.data + ensure_tensor(other).data)
    
    def __radd__(self, other):
        return Tensor(self.data + ensure_tensor(other).data)
    
    def __sub__(self, other):
        return Tensor(self.data - ensure_tensor(other).data)
    
    def __rsub__(self, other):
        return Tensor(ensure_tensor(other).data - self.data)
    
    def __mul__(self, other):
        return Tensor(self.data * ensure_tensor(other).data)
    
    def __rmul__(self, other):
        return Tensor(ensure_tensor(other).data * self.data)
    
    def __truediv__(self, other):
        return Tensor(self.data / ensure_tensor(other).data)
        
    def __rtruediv__(self, other):
        return Tensor(ensure_tensor(other).data/ self.data)
    
    def __neg__(self):
        return Tensor(-self.data)
    
    def __getitem__(self, idxs):
        return Tensor(self.data[idxs])
    
    def exp(self):
        """Return new tensor with exp transformation
        """
        return Tensor(np.exp(self.data))
    
    def log(self):
        """Return new tensor with log transformation
        """
        return Tensor(np.log(self.data))
    
    def mean(self, axis=None, dtype=None, out=None):
        """Return a new tensor with mean along some axis
        """
        if dtype is None:
            dtype = self.data.dtype
        return Tensor(np.mean(self.data, axis=axis, dtype=dtype, out=out))
    
    def sum(self, axis=None, out=None, **passkwargs):
        """Return new tensor with numpy sum along an axis
        """
        return Tensor(np.sum(self.data, axis=axis, out=out, **passkwargs))

    def tostring(self):
        """Return numpy array as byte string
        """
        return self.data.tostring()

    def copy(self):
        """Performs deepcopy
        """
        return deepcopy(self)
    
    @property
    def T(self):
        return Tensor(self.data.T)

    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def is_param(self):
        return self._is_param
    
    @property
    def grad(self):
        return self._grad
    
    @grad.setter
    def grad(self, grad):
        self._grad = ensure_tensor(grad)
        
    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.data))