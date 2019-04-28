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
    if isinstance(tensorable, Tensorable):
        return tensorable
    else:
        return Tensor(tensorable)

class Tensor(object):

    def __init__(
        self, 
        data: Arrayable,
        requires_grad: bool=False,
        depends_on: List[Dependency] = None) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other) -> 'Tensor':
        """Gets called if t + other
        """

        data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad

        if requires_grad:
            def grad_fn(grad: np.ndarray) -> np.ndarray:
                pass

            depends_on = Dependency(self, grad_fn)
        else:
            depends_on = []


        return 

    def __radd__(self, other) -> 'Tensor':
        """Gets called if + t
        """
        pass

    def __iadd__(self, other) -> 'Tensor':
        """Gets called if t += other
        """

    def __sub__(self, other) -> 'Tensor':
        """Gets called if t - other
        """

        data = self.data - other.data

    def __rsub__(self, other) -> 'Tensor':
        """Gets called if - t
        """

    def __isub__(self, other) -> 'Tensor':
        """Gets called if t -= other
        """

    def __mul__(self, other) -> 'Tensor':
        pass

    def __rmul__(self, other) -> 'Tensor':
        pass

    def __imul__(self, other) -> 'Tensor':
        pass

    def __matmul__(self, other) -> 'Tensor':
        pass

    def __truediv__(self, other) -> 'Tensor':
        pass

    def __neg__(self) -> 'Tensor':
        
        data = -self.data
        requires_grad = self.requires_grad

        if requires_grad:
            def grad_fn(grad: np.ndarray) -> np.ndarray:
                pass

    def __pow__(self) -> 'Tensor':
        pass

    def sum(self) -> 'Tensor':
        data = self.data.sum()
        requires_grad = self.requires_grad

        if requires_grad:
            def grad_fn(grad: np.ndarray) -> np.ndarray:
                return grad * np.ones_like(self.data)

            depends_on = [Dependency(self, grad_fn)]
        else:
            depends_on = []

        return Tensor(
            data,
            requires_grad,
            depends_on
        )

    def mean(self) -> 'Tensor':
        pass

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def backward(self, grad: 'Tensor' = None) -> None:
        
        if grad is None:
            if self.shape == ():
                # Seed tensor since we are using reverse automatic differentiation
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        self.grad.data = self.grad.data + grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)