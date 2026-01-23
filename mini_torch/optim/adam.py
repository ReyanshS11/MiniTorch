import numpy as np

from mini_torch.core.tensor import Tensor
from mini_torch.nn.module import Module

class Adam(Module):
    def __init__(self, parameters, lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, maximize=False, grad_clipping=False):
        super().__init__()
        self.parameters = parameters
        self.lr = lr
        self.grad_clipping = grad_clipping

        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.amsgrad = amsgrad
        self.maximize = maximize

        self.m = 0
        self.v = 0
        
        if amsgrad:
            self.v_max = 0
        
        self.t = 0

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

            self.t += 1

            g = p.grad
            if self.maximize:
                g = -g

            if self.weight_decay != 0:
                g += self.weight_decay * p

            self.m = self.beta1 * self.m + (1 - self.beta1) * g
            self.v = self.beta2 * self.v + (1 - self.beta2) * g**2

            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)

            if self.amsgrad:
                self.v_max = max(self.v_max, self.v)
                v_hat = self.v_max / (1 - self.beta2**self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)