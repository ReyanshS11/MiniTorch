from mini_torch.core.tensor import Tensor
from mini_torch.nn.layers import Linear
from mini_torch.nn.loss import MSELoss
from mini_torch.optim.sgd import SGD

import torch
import torch.nn as nn
import numpy as np

if __name__ == "__main__":
    x = Tensor([5.0, 5.0, 5.0], requires_grad=True)
    y = Tensor([1.0], requires_grad=False)

    model = Linear(3, 1)
    optim = SGD(model.parameters(), lr=0.01)
    loss_fn = MSELoss()

    for step in range(10):
        optim.zero_grad()

        out = model(x)
        loss = loss_fn(out, y)

        loss.backward()
        optim.step()

        print(loss.data)