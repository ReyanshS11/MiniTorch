# 🔥 mini-torch

A **from-scratch deep learning framework** built in pure Python and NumPy, inspired by PyTorch.

This project is for **learning purposes** and focuses on implementing:

- Autograd
- Tensors with gradient tracking
- Neural network layers
- Loss functions
- Optimizers
- DataLoader & Dataset abstractions
- Training loops and models

---

## 📦 Features

- **Tensor Class**:
  - Gradient tracking (`requires_grad=True`)
  - Backpropagation via `.backward()`
  - Basic ops: +, -, *, /, @, **, //
  - Extra ops: sum, mean, reshape, transpose, stack, zeros/zeros_like, ones/ones_like, max, min, stack, cat 

- **Neural Network API**
  - `Module` base class
  - Layers: `Linear`, `Conv1d`, `Conv2d`, `Conv3d`, `ReLU`
  - Losses: `MSELoss`, `BCELoss`, `BCEWithLogitsLoss` (work in progress)

- **Optimizers**
  - `SGD`
  - `Adam`

- **Training Utilities**
  - `Dataset`
  - `DataLoader`

---

## 🧠 Project Structure

```
mini_torch/
│
├── core/
|   ├── tensor.py          # Tensor + autograd engine
│   ├── autograd.py
│   └── ops.py 
│
├── nn/
│   ├── module.py          # Base Module class
│   ├── layers.py          # Linear, Conv1d, ReLU, etc.
│   └── loss.py            # MSELoss, BCELoss, etc.
│
├── optim/
│   ├── sgd.py
│   └── adam.py
│
├── data/
│   ├── dataset.py
│   └── dataloader.py
│
└── utils/
```

---

## 🚀 Example Usage

### 1️⃣ Creating Tensors

```python
from mini_torch.core.tensor import Tensor

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = Tensor([4.0, 5.0, 6.0], requires_grad=True)

z = (x * y).sum()
z.backward()

print(x.grad)
print(y.grad)
```

---

### 2️⃣ Building a Model

```python
from mini_torch.nn.layers import Linear, ReLU
from mini_torch.nn.module import Module

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(10, 32)
        self.relu = ReLU()
        self.fc2 = Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

---

### 3️⃣ Training Loop

```python
from mini_torch.optim.sgd import SGD
from mini_torch.nn.loss import MSELoss

model = MLP()
optimizer = SGD(model.parameters(), lr=0.01)
criterion = MSELoss()

for epoch in range(100):
    preds = model(x)
    loss = criterion(preds, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Loss:", loss.data)
```

---

## 🛠️ Supported Operations (So Far)

- Elementwise:
  - `+`, `-`, `*`, `/`
  - `log`, `pow`, `neg`, `rsub`
  - `clamp`
- Matrix:
  - `@` (matmul)
- Reductions:
  - `sum`, `mean`, `max`, `min`
- Shape / Tensor Transform:
  - `reshape`, `T` (transpose), `stack`, `cat`
- Tensor Creation:
  - `zeros`, `ones`, `zeros_like`, `ones-like` 
- Activations:
  - `ReLU`

---

## Speed Comparison (vs. PyTorch)

![Comparison Graph](./benchmark/training_time_comparison_MiniTorch_PyTorch.png)

**All tests were ran on the default benchmark found in ./benchmark 

---

## 🎯 Goals of This Project

- Understand **how PyTorch works internally**
- Learn how **gradients and autograd** work
- Learn how **optimizers** and **layers** are implemented

---

## ⚠️ Disclaimer

This is a **learning framework**, not meant for production use.

- ❌ Not very optmized
- ❌ No GPU

---

---

## 🧩 Future Work

- [ ] CrossEntropyLoss
- [ ] BatchNorm / LayerNorm
- [ ] More activations (Sigmoid, Tanh, GELU)
- [ ] More ops
- [ ] CNN training on MNIST
- [ ] Serialization (`save` / `load`)

---
