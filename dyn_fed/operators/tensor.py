from typing import List, NamedTuple, Callable, Optional, Union
import numpy as np

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

Arrayable = Union[float, list, np.ndarray]

def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)

Tensorable = Union['Tensor', 'float', np.ndarray]

def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)

class Tensor(object):

    def __init__(
        self, 
        data: Arrayable,
        requires_grad: bool=False,
        depends_on: List[Dependency] = None) -> None:
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other) -> 'Tensor':
        """Gets called if t + other
        """

        return _add(self, ensure_tensor(other))

    def __radd__(self, other) -> 'Tensor':
        """Gets called if + t
        """
        return _add(self, ensure_tensor(other))

    def __iadd__(self, other) -> 'Tensor':
        """Gets called if t += other
        """
        self.data = self.data + ensure_tensor(other).data
        
        return self

    def __sub__(self, other) -> 'Tensor':
        """Gets called if t - other
        """
        return _sub(self, ensure_tensor(other))

    def __rsub__(self, other) -> 'Tensor':
        """Gets called if - t
        """
        return _sub(self, ensure_tensor(other))

    def __isub__(self, other) -> 'Tensor':
        """Gets called if t -= other
        """

        self.data = self.data - ensure_tensor(other).data

        return self

    def __mul__(self, other) -> 'Tensor':
        return _mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> 'Tensor':
        return _mul(self, ensure_tensor(other))

    def __imul__(self, other) -> 'Tensor':
        self.data = self.data * ensure_tensor(other).data

        return self

    def __matmul__(self, other) -> 'Tensor':
        return _matmul(self, other)

    def __truediv__(self, other) -> 'Tensor':
        pass

    def __neg__(self) -> 'Tensor':
        return _neg(self)

    def __pow__(self) -> 'Tensor':
        pass

    def __getitem__(self, idxs) -> 'Tensor':
        return _slice(self, idxs)

    def sum(self) -> 'Tensor':
        return tensor_sum(self)

    def mean(self) -> 'Tensor':
        pass

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        # Setting the data manually means we invalidate the gradient
        self.grad = None

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def backward(self, grad: 'Tensor' = None) -> None:
        
        assert self.requires_grad, "called backwards on non-requires-grad tensor"
        if grad is None:
            if self.shape == ():
                # Seed tensor since we are using reverse automatic differentiation
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        self.grad.data = self.grad.data + grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))


def tensor_sum(t: Tensor) -> Tensor:
    """Takes a tensor and returns the 0-tensor that's the sum of all it's elements
    """

    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """Grad is necessarily a 0-tensor, so each input element contributes that much
            """

            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]

    else:

        depends_on = []
    
    return Tensor(
        data,
        requires_grad,
        depends_on
    )

def _add(t1: Tensor, t2: Tensor) -> Tensor:
    """Takes a tensor and returns the 0-tensor that's the sum of all it's elements
    """

    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim

            for _ in np.arange(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:

            # Handle broacasting properly
            ndims_added = grad.ndim - t2.data.ndim

            for _ in np.arange(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
                
            return grad

        depends_on.append(Dependency(t2, grad_fn2))
    
    return Tensor(
        data,
        requires_grad,
        depends_on
    )

def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    """
    """

    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:

            grad = grad * t2.data

            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim

            for _ in np.arange(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:

            grad = grad * t1.data

            # Handle broacasting properly
            ndims_added = grad.ndim - t2.data.ndim

            for _ in np.arange(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
                
            return grad

        depends_on.append(Dependency(t2, grad_fn2))
    
    return Tensor(
        data,
        requires_grad,
        depends_on
    )

def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on = []

    return Tensor(
        data,
        requires_grad,
        depends_on
    )

def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return _add(t1, _neg(t2))

def _matmul(t1: Tensor, t2: Tensor) -> Tensor:

    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:

            grad = grad @ t2.data.T

            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:

            grad = t1.data.T @ grad
                
            return grad

        depends_on.append(Dependency(t2, grad_fn2))
    
    return Tensor(
        data,
        requires_grad,
        depends_on
    )

def _slice(t: Tensor, *idxs) -> Tensor:
    """
    """
    data = t.data[idxs]
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(data)
            bigger_grad[idxs] = grad
            return bigger_grad

        depends_on = Dependency(t, grad_fn)
    else:

        depends_on = []

    return Tensor(data, requires_grad, depends_on)