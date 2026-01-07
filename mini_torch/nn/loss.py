import numpy as np

from mini_torch.core.tensor import Tensor
from .module import Module

class MSELoss(Module):
    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        difference = y_hat - y
        mse = (difference**2).mean()

        return mse