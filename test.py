from mini_torch.core.tensor import Tensor
from mini_torch.core.ops import max
from mini_torch.nn.layers import Linear, ReLU, Conv1d, Conv2d, Conv3d
from mini_torch.nn.loss import MSELoss
from mini_torch.optim.sgd import SGD
from mini_torch.data.dataset import Dataset
from mini_torch.data.dataloader import DataLoader

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

x = Tensor(np.random.randn(4, 3, 10, 10, 10), requires_grad=True)

m = Conv3d(3, 1, (3, 3, 5), 2)
m = m(x)

loss = m.sum()

loss.backward()

print("M ", m.data.shape, "X ", x.grad.shape)