import numpy as np

from mini_torch.core.tensor import Tensor
from mini_torch.nn.module import Module

class SGD(Module):
    def __init__(self, parameters, lr, momentum=0.0, nesterov=False, grad_clipping=False):
        super().__init__()
        self.parameters = parameters
        self.lr = lr
        self.grad_clipping = grad_clipping

        self.momentum = momentum
        self.nesterov = nesterov

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    def step(self):
        for p in self.parameters:
            assert isinstance(p, Tensor)

            if p.grad is None:
                continue

            if self.grad_clipping:
                np.clip(p.grad, -1.0, 1.0, out=p.grad)

            if self.nesterov:
                w_look_ahead = p - self.momentum * p.velocity

            v_new = self.momentum * p.velocity + (1 - self.momentum) * p.grad
            p.data -= self.lr * v_new

            p.velocity = v_new