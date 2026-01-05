import numpy as np
from tensor import Tensor

def __add__(tns: Tensor, other) -> Tensor:
    other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=tns.requires_grad)
    out = Tensor(tns.data + other.data, requires_grad=tns.requires_grad or other.requires_grad)
    out._prev = {tns, other}

    def _backward():
        if tns.requires_grad:
            grad = tns.unbroadcast(out.grad, tns.data.shape)
            tns.grad = tns.grad + grad if tns.grad is not None else grad
        if other.requires_grad:
            grad = tns.unbroadcast(out.grad, other.data.shape)
            other.grad = other.grad + grad if other.grad is not None else grad
    
    out._backward = _backward
    return out

def __sub__(tns: Tensor, other) -> Tensor:
    other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=tns.requires_grad)
    out = Tensor(tns.data - other.data, requires_grad=tns.requires_grad or other.requires_grad)
    out._prev = {tns, other}

    def _backward():
        if tns.requires_grad:
            grad = tns.unbroadcast(out.grad, tns.data.shape)
            tns.grad = tns.grad + grad if tns.grad is not None else out.grad
        if other.requires_grad:
            grad = tns.unbroadcast(out.grad, other.data.shape)
            other.grad = other.grad - grad if other.grad is not None else -out.grad # type: ignore
    
    out._backward = _backward
    return out

def __mul__(tns: Tensor, other) -> Tensor:
    other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=tns.requires_grad)
    out = Tensor(tns.data * other.data, requires_grad=tns.requires_grad or other.requires_grad)
    out._prev = {tns, other}

    def _backward():
        if tns.requires_grad:
            grad = out.grad * other.data
            grad = tns.unbroadcast(grad, tns.data.shape)
            tns.grad = tns.grad + grad if tns.grad is not None else grad
        if other.requires_grad:
            grad = out.grad * tns.data
            grad = tns.unbroadcast(grad, other.data.shape)
            other.grad = other.grad + grad if other.grad is not None else grad

    out._backward = _backward
    return out

def __truediv__(tns: Tensor, other) -> Tensor:
    other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=tns.requires_grad)
    out = Tensor(tns.data / other.data, requires_grad=tns.requires_grad or other.requires_grad)
    out._prev = {tns, other}

    def _backward():
        if tns.requires_grad:
            grad_x = out.grad / other.data
            grad_x = tns.unbroadcast(grad_x, tns.data.shape)
            tns.grad = tns.grad + grad_x if tns.grad is not None else grad_x
        if other.requires_grad:
            grad_y = -out.grad * tns.data / (other.data ** 2) # type: ignore
            grad_y = tns.unbroadcast(grad_y, other.data.shape)
            other.grad = other.grad + grad_y if other.grad is not None else grad_y
    
    out._backward = _backward
    return out

def __neg__(tns: Tensor) -> Tensor:
    out = Tensor(-tns.data, requires_grad=tns.requires_grad)
    out._prev = {tns}

    def _backward():
        if tns.requires_grad:
            grad = tns.unbroadcast(out.grad, tns.data.shape)
            tns.grad = tns.grad - grad if tns.grad is not None else -grad # type: ignore

    out._backward = _backward
    return out

def __pow__(tns: Tensor, power) -> Tensor:
    assert isinstance(power, (int, float)), "only scalar powers supported"
    out = Tensor(tns.data ** power, requires_grad=tns.requires_grad)
    out._prev = {tns}

    def _backward():
        if tns.requires_grad:
            grad = power * (tns.data ** (power - 1)) * out.grad # type: ignore
            grad = tns.unbroadcast(grad, tns.data.shape)
            tns.grad = tns.grad + grad if tns.grad is not None else grad

    out._backward = _backward
    return out

def __matmul__(tns: Tensor, other) -> Tensor:
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(tns.data @ other.data, requires_grad=tns.requires_grad or other.requires_grad)
    out._prev = {tns, other}

    def _backward():
        if tns.requires_grad:
            grad_A = out.grad @ other.data.T
            grad_A = grad_A.unbroadcast(grad_A, tns.data.shape)
            tns.grad = tns.grad + grad_A if tns.grad is not None else grad_A
        if other.requires_grad:
            grad_A = tns.data.T @ out.grad
            grad_A = grad_A.unbroadcast(grad_A, other.data.shape)
            tns.grad = other.grad + grad_A if other.grad is not None else grad_A  

    out._backward = _backward
    return out

def sum(tns: Tensor) -> Tensor:
    out = Tensor(tns.data.sum(), requires_grad=tns.requires_grad)
    out._prev = {tns}

    def _backward():
        if tns.requires_grad:
            grad = out.grad * np.ones_like(tns.data)
            tns.grad = tns.grad + grad if tns.grad is not None else grad
        
    out._backward = _backward
    return out

def mean(tns: Tensor) -> Tensor:
    out = Tensor(tns.data.mean(), requires_grad=tns.requires_grad)
    out._prev = {tns}

    def _backward():
        if tns.requires_grad:
            grad = np.ones_like(tns.data) * (out.grad / tns.data.size) # type: ignore
            tns.grad = tns.grad + grad if tns.grad is not None else grad
    
    out._backward = _backward
    return out

def reshape(tns: Tensor, shape) -> Tensor:
    def prod(shape):
        p = 1
        for s in shape:
            p *= s
        return p
    
    if prod(shape) != prod(tns.data.shape):
        raise ValueError(
            f"reshape of shape {shape} not available for tensor of shape {tns.data.shape}"
        )

    out = Tensor(tns.data.reshape(shape), requires_grad=tns.requires_grad)
    out._prev = {tns}

    def _backward():
        if tns.requires_grad:
            grad = out.grad.reshape(tns.data.shape)  # type: ignore
            tns.grad = grad if tns.grad is None else tns.grad + grad

    out._backward = _backward
    return out