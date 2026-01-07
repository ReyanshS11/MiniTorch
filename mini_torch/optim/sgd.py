import numpy as np

from mini_torch.core.tensor import Tensor
from mini_torch.nn.module import Module

class SGD(Module):
    def __init__(self, parameters, lr):
        super().__init__()
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None

    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue
            p.data -= self.lr * p.grad