import numpy as np

class Dual(object):
    """Short description
    
    Long description
    
    Attributes:
        val (float): Value
        derivative (float): derivative=1 means we are going to derive w.r.t this variable
    """
    def __init__(self, val=0, derivative=0):
        self._val = val
        self._derivative = derivative

    # getters / setters
    @property
    def derivative(self):
        return self._derivative

    @derivative.setter
    def derivative(self, derivative):
        self._derivative = derivative

    @property
    def shape(self):
        return self._val.shape

    @property
    def T(self):
        return self._val.T

    # operators
    def __add__(self, other):
        return Dual(
            val=self._val + other._val, 
            derivative=self._derivative + other._derivative
        )

    def __sub__(self, other):
        return Dual(
            val=self._val - other._val, 
            derivative=self._derivative - other._derivative
        )
    
    def __mul__(self, other):
        return Dual(
            val=self._val * other._val, 
            derivative=self._derivative * other._val + self._val * other._derivative
        )

    def __truediv__(self, other):
        return Dual(
            val=self._val + other._val, 
            derivative=(self._derivative + other._val - self._val * other._derivative) / other._val * other._val
        )

    def __neg__(self):
        return Dual(
            val=-self._val,
            derivative=-self._derivative
        )

    def __str__(self):
        return f"Dual(val={self._val})"

    # maths    
    def __pow__(self, p):
        return Dual(
            val=np.power(self._val, p),
            derivative=p * self._derivative * np.power(self._val, p-1)
        )

    def sin(self):
        return Dual(
            val=np.sin(self._val),
            derivative=np.cos(self._val)
        )

    def cos(self):
        return Dual(
            val=np.cos(self._val),
            derivative=-np.sin(self._val)
        )

    def exp(self):
        return Dual(
            val=np.exp(self._val),
            derivative=np.exp(self._val)
        )

    def log(self):
        return Dual(
            val=np.log(self._val),
            derivative=self._derivative / self._val
        )

    def abs(self):
        sign = 0 if self._val == 0 else self._val / abs(self._val)
        return Dual(
            val=abs(self._val),
            derivative=self._derivative * sign
        )

    def pow(self, p):
        return Dual(
            val=np.power(self._val, p),
            derivative=p * self._derivative * np.power(self._val, p-1)
        )

    def sum(self):
        return Dual(
            val=np.sum(self._val),
            derivative=np.sum(self._derivative)
        )

    def mean(self):
        return Dual(
            val=np.sum(self._val) / self._val.shape[0],
            derivative=np.sum(self._derivative) / self._derivative.shape[0]
        )
