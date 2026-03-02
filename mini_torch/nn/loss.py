import warnings

import numpy as np

from mini_torch.core.tensor import Tensor
from mini_torch.core.ops import log, clamp

from .module import Module

class MSELoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        difference = y_hat - y

        if self.reduction == "mean":
            mse = (difference**2).mean()
        elif self.reduction == "sum":
            mse = (difference**2).sum()
        elif self.reduction == "none":
            mse = difference**2
        else:
            raise ValueError (
                f"""Reduction of {self.reduction} not available. Must be one of ["none", "mean", "sum"]."""
            )

        return mse
    
class BCELoss(Module):
    def __init__(self, weight=None, reduction="mean"):
        self.weight = weight
        self.reduction = reduction

    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        bce = -self.weight * (y * clamp(log(y_hat), min=-100) + (1 - y) * clamp(log(1 - y_hat), min=-100)) if self.weight is not None else y * clamp(log(y_hat), min=-100) + (1 - y) * clamp(log(1 - y_hat), min=-100)

        if self.reduction == "mean":
            bce = bce.mean()
        elif self.reduction == "sum":
            bce = bce.sum()
        elif self.reduction == "none":
            bce = bce
        else:
            raise ValueError (
                f"""Reduction of {self.reduction} not available. Must be one of ["none", "mean", "sum"]."""
            )
        
        return bce
    
class BCEWithLogitsLoss(Module):
    def __init__(self, weight=None, reduction="mean"):
        warnings.warn("As of now, this is the same as BCELoss. Sigmoid will be added later.")

        self.weight = weight
        self.reduction = reduction

    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        bce = -self.weight * (y * clamp(log(y_hat), min=-100) + (1 - y) * clamp(log(1 - y_hat), min=-100)) if self.weight is not None else y * clamp(log(y_hat), min=-100) + (1 - y) * clamp(log(1 - y_hat), min=-100)

        if self.reduction == "mean":
            bce = bce.mean()
        elif self.reduction == "sum":
            bce = bce.sum()
        elif self.reduction == "none":
            bce = bce
        else:
            raise ValueError (
                f"""Reduction of {self.reduction} not available. Must be one of ["none", "mean", "sum"]."""
            )
        
        return bce