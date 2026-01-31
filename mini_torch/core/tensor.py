from __future__ import annotations
import numpy as np

class Tensor:
    def __init__(self, data, dtype=np.float32, requires_grad=False):
        self.dtype = dtype
        self.data = np.asarray(data, dtype=self.dtype)

        self.requires_grad = requires_grad
        self.grad = None

        self._prev = set()
        self._backward = lambda: None

        self.state_dict = {"m": 0, "v": 0, "v_max": 0}

    def numerical_grad(self, f, x, eps=1e-6):
        from .autograd import numerical_grad
        return numerical_grad(f, x, eps)
    
    def unbroadcast(self, grad, shape):
        from .autograd import unbroadcast
        return unbroadcast(grad, shape)

    def __add__(self, other) -> Tensor:
        from .ops import __add__
        return __add__(self, other)
    
    def __sub__(self, other) -> Tensor:
        from .ops import __sub__
        return __sub__(self, other)
    
    def __rsub__(self, other) -> Tensor:
        from .ops import __rsub__
        return __rsub__(other, self)

    def __mul__(self, other) -> Tensor:
        from .ops import __mul__
        return __mul__(self, other)
    
    def __truediv__(self, other) -> Tensor:
        from .ops import __truediv__
        return __truediv__(self, other)
    
    def __neg__(self) -> Tensor:
        from .ops import __neg__
        return __neg__(self)
    
    def __pow__(self, power) -> Tensor:
        from .ops import __pow__
        return __pow__(self, power)
    
    def __matmul__(self, other) -> Tensor:
        from .ops import __matmul__
        return __matmul__(self, other)

    def __getitem__(self, *idx):
        from .ops import __getitem__
        return __getitem__(self, idx)

    def sum(self) -> Tensor:
        from .ops import sum
        return sum(self)
    
    def mean(self) -> Tensor:
        from .ops import mean
        return mean(self)
    
    def reshape(self, shape) -> Tensor:
        from .ops import reshape
        return reshape(self, shape)

    def T(self, axes = None) -> Tensor:
        from .ops import T
        return T(self, axes)

    def size(self):
        return self.data.size

    def shape(self):
        return self.data.shape

    def backward(self) -> None:
        from .autograd import backward
        backward(self)