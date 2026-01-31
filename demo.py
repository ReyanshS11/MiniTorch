from mini_torch.core.tensor import Tensor
from mini_torch.nn.layers import Linear, Conv1d, ReLU
from mini_torch.nn.loss import MSELoss, BCELoss
from mini_torch.optim.sgd import SGD
from mini_torch.optim.adam import Adam
from mini_torch.data.dataset import Dataset
from mini_torch.data.dataloader import DataLoader

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

data = pd.read_csv("Housing.csv")
data = data.replace({
    "yes": 1,
    "no": 0,
    "furnished": 2,
    "semi-furnished": 1,
    "unfurnished": 0
})

data = (data - data.mean()) / (data.std() + 1e-8)

data_np = data.to_numpy(dtype=np.float32)

X = data_np[:, 1:]
y = data_np[:, 0:1]

class TestData(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        input_tns = self.x[idx]
        label_tns = self.y[idx]

        return input_tns, label_tns

if __name__ == "__main__":
    dataset = TestData(X, y)
    dataset = DataLoader(dataset, 4)

    model = Linear(12, 1)

    optim = SGD(model.parameters(), lr=1e-3)
    loss_fn = MSELoss()

    for epoch in range(10):
        total_loss = []

        for batch in dataset:
            x = batch[0]
            y = batch[1]

            optim.zero_grad()

            out = model(x)
            loss = loss_fn(out, y)

            loss.backward()
            optim.step()

            total_loss.append(loss.data)

        print(f"{epoch}: {np.mean(total_loss)}")