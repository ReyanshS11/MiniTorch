import torch
import torch.nn as nn
import numpy as np

x = torch.tensor([[9.0, 1.0, 6.0], [1.0, 7.0, 2.0]], requires_grad=True)
y = torch.tensor([[4.0, 7.0], [2.0, 5.0], [8.0, 1.0]], requires_grad=True)

z = x @ y

print(x, "\n\n", y)