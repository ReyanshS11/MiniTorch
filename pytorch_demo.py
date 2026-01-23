import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# -----------------------------
# Load and preprocess data
# -----------------------------
data = pd.read_csv("Housing.csv")
data = data.replace({
    "yes": 1,
    "no": 0,
    "furnished": 2,
    "semi-furnished": 1,
    "unfurnished": 0
})

# Normalize
data = (data - data.mean()) / (data.std() + 1e-8)

# Convert to numpy
data_np = data.to_numpy(dtype=np.float32)

# First column = target, rest = features
X = data_np[:, 1:]
y = data_np[:, 0:1]   # keep as (N, 1)

# -----------------------------
# Dataset
# -----------------------------
class TestData(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------
# Dataloader
# -----------------------------
dataset = TestData(X, y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# -----------------------------
# Model
# -----------------------------
model = nn.Linear(12, 1)

# -----------------------------
# Optimizer and loss
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(10):
    total_loss = []

    for x, y in loader:
        optimizer.zero_grad()

        out = model(x)
        loss = loss_fn(out, y)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    print(f"{epoch}: {np.mean(total_loss)}")