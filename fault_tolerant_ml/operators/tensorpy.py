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
        other = ensure_tensor(other)
        is_param = self.is_param or other.is_param
        return Tensor(self.data @ other.data, is_param=is_param)

    def __rmatmul__(self, other):
        other = ensure_tensor(other)
        is_param = self.is_param or other.is_param
        return Tensor(other.data @ self.data, is_param=is_param)
    
    def __add__(self, other):
        other = ensure_tensor(other)
        is_param = self.is_param or other.is_param
        return Tensor(self.data + other.data, is_param=is_param)
    
    def __radd__(self, other):
        other = ensure_tensor(other)
        is_param = self.is_param or other.is_param
        return Tensor(self.data + other.data, is_param=is_param)
    
    def __sub__(self, other):
        other = ensure_tensor(other)
        is_param = self.is_param or other.is_param
        return Tensor(self.data - other.data, is_param=is_param)
    
    def __rsub__(self, other):
        other = ensure_tensor(other)
        is_param = self.is_param or other.is_param
        return Tensor(other.data - self.data, is_param=is_param)
    
    def __mul__(self, other):
        other = ensure_tensor(other)
        is_param = self.is_param or other.is_param
        return Tensor(self.data * other.data, is_param=is_param)
    
    def __rmul__(self, other):
        other = ensure_tensor(other)
        is_param = self.is_param or other.is_param
        return Tensor(other.data * self.data, is_param=is_param)
    
    def __truediv__(self, other):
        other = ensure_tensor(other)
        is_param = self.is_param or other.is_param
        return Tensor(self.data / other.data, is_param=is_param)
        
    def __rtruediv__(self, other):
        other = ensure_tensor(other)
        is_param = self.is_param or other.is_param
        return Tensor(other.data/ self.data, is_param=is_param)
    
    def __neg__(self):
        is_param = self.is_param
        return Tensor(-self.data, is_param=is_param)

    def __abs__(self):
        is_param = self.is_param
        return Tensor(abs(self.data), is_param=is_param)

    def __pow__(self, p):
        is_param = self.is_param
        return Tensor(self.data ** p, is_param=is_param)
    
    def __getitem__(self, idxs):
        is_param = self.is_param
        return Tensor(self.data[idxs], is_param=is_param)
    
    def exp(self):
        """Return new tensor with exp transformation
        """
        is_param = self.is_param
        return Tensor(np.exp(self.data), is_param=is_param)
    
    def log(self):
        """Return new tensor with log transformation
        """
        is_param = self.is_param
        return Tensor(np.log(self.data), is_param=is_param)
    
    def mean(self, axis=None, dtype=None, out=None):
        """Return a new tensor with mean along some axis
        """
        is_param = self.is_param
        if dtype is None:
            dtype = self.data.dtype
        return Tensor(np.mean(self.data, axis=axis, dtype=dtype, out=out), is_param=is_param)
    
    def sum(self, axis=None, out=None, **passkwargs):
        """Return new tensor with numpy sum along an axis
        """
        is_param = self.is_param
        return Tensor(np.sum(self.data, axis=axis, out=out, **passkwargs), is_param=is_param)

    def sqrt(self, out=None, where=None, **kwargs):
        """Return new tensor with numpy sum along an axis
        """
        is_param = self.is_param
        return Tensor(np.sqrt(self.data), is_param=is_param)

    def argmax(self, axis=None, out=None):
        is_param = self.is_param
        return Tensor(np.argmax(self.data, axis=axis, out=out), is_param=is_param)

    def zeros_like(self, dtype=None, order='K', subok=True):
        is_param = self.is_param
        if dtype is None:
            dtype = self.data.dtype
        return Tensor(np.zeros_like(self.data, dtype=dtype, order=order, subok=subok), is_param=is_param)

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
        is_param = self.is_param
        return Tensor(self.data.T, is_param=is_param)

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