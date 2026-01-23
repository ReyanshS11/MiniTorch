import numpy as np

from mini_torch.core.tensor import Tensor
from mini_torch.nn.module import Module

class Adam(Module):
    def __init__(self, parameters, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, maximize=False, grad_clipping=False):
        super().__init__()
        self.parameters = parameters
        self.lr = lr
        self.grad_clipping = grad_clipping

        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.amsgrad = amsgrad
        self.maximize = maximize

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

            m, v, v_max = p.state_dict["m"], p.state_dict["v"], 0

            if self.amsgrad:
                v_max = p.state_dict["v_max"]

            g = p.grad
            if self.maximize:
                g = -g

            if self.weight_decay != 0:
                g += self.weight_decay * p.data

            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g**2

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            if self.amsgrad:
                v_max = np.maximum(v_max, v)
                v_hat = v_max / (1 - self.beta2**self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.state_dict.update({"m": m, "v": v, "v_max": v_max})