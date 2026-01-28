import numpy as np

from mini_torch.core.tensor import Tensor
from mini_torch.nn.module import Module

class SGD(Module):
    def __init__(self, parameters, lr, momentum=0.0, dampening=0, weight_decay=0, nesterov=False, maximize=False, grad_clipping=False):
        super().__init__()
        self.parameters = parameters
        self.lr = lr
        self.maximize = maximize
        self.grad_clipping = grad_clipping

        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        self.t = 0

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    def step(self):
        self.t += 1
        for p in self.parameters:
            assert isinstance(p, Tensor)

            if p.grad is None:
                continue

            if self.grad_clipping:
                np.clip(p.grad, -1.0, 1.0, out=p.grad)

            g = p.grad
            if self.maximize:
                g = -g
            
            if self.weight_decay != 0:
                g += self.weight_decay * p.data

            b = g
            if self.momentum != 0:
                if self.t > 1:
                    b = self.momentum * b + (1 - self.dampening) * g
                
                if self.nesterov:
                    g += self.momentum * b
                else:
                    g = b
            
            p.data -= self.lr * g