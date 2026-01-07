from __future__ import annotations
import numpy as np
from .tensor import Tensor
from .autograd import unbroadcast

def __add__(tns: Tensor, other) -> Tensor:
    other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=tns.requires_grad)
    out = Tensor(tns.data + other.data, requires_grad=tns.requires_grad or other.requires_grad)
    out._prev = {tns, other}

    def _backward():
        if tns.requires_grad:
            grad = unbroadcast(out.grad, tns.data.shape)
            tns.grad = tns.grad + grad if tns.grad is not None else grad
        if other.requires_grad:
            grad = unbroadcast(out.grad, other.data.shape)
            other.grad = other.grad + grad if other.grad is not None else grad
    
    out._backward = _backward
    return out

def __sub__(tns: Tensor, other) -> Tensor:
    other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=tns.requires_grad)
    out = Tensor(tns.data - other.data, requires_grad=tns.requires_grad or other.requires_grad)
    out._prev = {tns, other}

    def _backward():
        if tns.requires_grad:
            grad = unbroadcast(out.grad, tns.data.shape)
            tns.grad = tns.grad + grad if tns.grad is not None else out.grad
        if other.requires_grad:
            grad = unbroadcast(out.grad, other.data.shape)
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
            grad = unbroadcast(grad, tns.data.shape)
            tns.grad = tns.grad + grad if tns.grad is not None else grad
        if other.requires_grad:
            grad = out.grad * tns.data
            grad = unbroadcast(grad, other.data.shape)
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
            grad_x = unbroadcast(grad_x, tns.data.shape)
            tns.grad = tns.grad + grad_x if tns.grad is not None else grad_x
        if other.requires_grad:
            grad_y = -out.grad * tns.data / (other.data ** 2) # type: ignore
            grad_y = unbroadcast(grad_y, other.data.shape)
            other.grad = other.grad + grad_y if other.grad is not None else grad_y
    
    out._backward = _backward
    return out

def __neg__(tns: Tensor) -> Tensor:
    out = Tensor(-tns.data, requires_grad=tns.requires_grad)
    out._prev = {tns}

    def _backward():
        if tns.requires_grad:
            grad = unbroadcast(out.grad, tns.data.shape)
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
            grad = unbroadcast(grad, tns.data.shape)
            tns.grad = tns.grad + grad if tns.grad is not None else grad

    out._backward = _backward
    return out

def __matmul__(tns: Tensor, other) -> Tensor:
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(tns.data @ other.data, requires_grad=tns.requires_grad or other.requires_grad)
    out._prev = {tns, other}

    def _backward():
        A = tns.data
        B = other.data
        G = out.grad

        A2 = np.atleast_2d(A)
        B2 = np.atleast_2d(B)
        G2 = np.atleast_2d(G)

        if tns.requires_grad:
            grad_A = G2 @ B2.T
            grad_A = grad_A.reshape(A.shape)
            grad_A = unbroadcast(grad_A, A.shape)
            tns.grad = grad_A if tns.grad is None else tns.grad + grad_A

        if other.requires_grad:
            grad_B = G2.T @ A2
            grad_B = grad_B.reshape(B.shape)
            grad_B = unbroadcast(grad_B, B.shape)
            other.grad = grad_B if other.grad is None else other.grad + grad_B

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

def T(tns: Tensor) -> Tensor:
    out = Tensor(tns.data.T, requires_grad=tns.requires_grad)
    out._prev = {tns}

    def _backward():
        if tns.requires_grad:
            grad = out.grad.T
            tns.grad = tns.grad + grad if tns.grad is not None else grad
    
    out._backward = _backward
    return out