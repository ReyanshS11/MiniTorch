import numpy as np
import torch

from .init import _kaiming_init
from mini_torch.core.tensor import Tensor
from mini_torch.core.ops import cat, zeros_like, maximum, stack
from .module import Module

from typing import Sequence

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, a = 5):
        super().__init__()
        self.a = a

        self.weight = Tensor(_kaiming_init((out_features, in_features), out_features, self.a, 11), requires_grad=True)

        if bias:
            self.bias = Tensor(_kaiming_init((out_features), out_features, self.a, 11), requires_grad=True)
        else:
            self.bias = None

    def __call__(self, x: Tensor) -> Tensor:
        x = x @ self.weight.T()

        if self.bias is not None:
            x = x + self.bias
            
        return x
    
class ReLU(Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, x: Tensor) -> Tensor:
        return maximum(zeros_like(x), x)
    
class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, a=5):
        super().__init__()
        self.a = a

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = Tensor(_kaiming_init((self.out_channels, self.in_channels, self.kernel_size), self.out_channels, self.a, 11))
        self.bias = Tensor(_kaiming_init((self.out_channels), self.out_channels, self.a, 11), requires_grad=True) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        if len(x.shape()) == 3:
            B, C, W = x.shape()
        elif len(x.shape()) == 2:
            container = []
            container.append(x)
            x = stack(container)

            B, C, W = x.shape()
        else:
            raise RuntimeError(
                f"shape {x.shape()} not valid for Conv1d"
            )

        C_out = self.out_channels
        C_in = self.in_channels
        K = self.kernel_size

        S = self.stride
        P = self.padding
        D = self.dilation

        W_out = (W + 2*P - D*(K-1) - 1) // S + 1
        
        rows = []
        for b in range(B):
            for i in range(W_out):
                row = []

                start = i * S - P

                for c in range(C_in):
                    for u in range(K):
                        x_index = start + D * u
                        
                        if 0 <= x_index < W:
                            row.append(x[b, c, x_index])
                        else:
                            row.append(Tensor(0.0))

                rows.append(stack(row))

        X_col = stack(rows)

        W_flat = self.weight.reshape((C_out, C_in * K))
        out_flat = X_col @ W_flat.T()

        out = out_flat.reshape((B, W_out, C_out)).T((0, 2, 1))

        if self.bias is not None:
            out = out + self.bias

        return out
    
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, a=5):
        super().__init__()
        self.a = a

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        self.weight = Tensor(_kaiming_init((self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]), self.out_channels, self.a, 11))
        self.bias = Tensor(_kaiming_init((self.out_channels), self.out_channels, self.a, 11), requires_grad=True) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        if len(x.shape()) == 4:
            B, C, H, W = x.shape()
        elif len(x.shape()) == 3:
            container = []
            container.append(x)
            x = stack(container)

            B, C, H, W = x.shape()
        else:
            raise RuntimeError(
                f"shape {x.shape()} not valid for Conv2d"
            )

        C_out = self.out_channels
        C_in = self.in_channels
        K_H = self.kernel_size[0]
        K_W = self.kernel_size[1]

        S = self.stride
        P = self.padding
        D = self.dilation

        H_out = (H + 2*P[0] - D[0]*(K_H-1) - 1) // S[0] + 1
        W_out = (W + 2*P[1] - D[1]*(K_W-1) - 1) // S[1] + 1
        
        rows = []
        for b in range(B):
            for i in range(H_out):
                for j in range(W_out):
                    patch = []

                    start_H = i * S[0] - P[0]
                    start_W = j * S[1] - P[1]

                    for c in range(C_in):
                        for u in range(K_H):
                            for v in range(K_W):
                                H_index = start_H + D[0] * u
                                W_index = start_W + D[1] * v
                                
                                if 0 <= H_index < H and 0 <= W_index < W:
                                    patch.append(x[b, c, H_index, W_index])
                                else:
                                    patch.append(Tensor(0.0))

                    rows.append(stack(patch))

        X_col = stack(rows)

        W_flat = self.weight.reshape((C_out, C_in * K_H * K_W))
        out_flat = X_col @ W_flat.T()

        out = out_flat.reshape((B, H_out, W_out, C_out)).T((0, 3, 1, 2))

        if self.bias is not None:
            out = out + self.bias

        return out
    
class Conv3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, a=5):
        super().__init__()
        self.a = a

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)

        self.weight = Tensor(_kaiming_init((self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]), self.out_channels, self.a, 11))
        self.bias = Tensor(_kaiming_init((self.out_channels), self.out_channels, self.a, 11), requires_grad=True) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        if len(x.shape()) == 5:
            B, C, D, H, W = x.shape()
        elif len(x.shape()) == 4:
            container = []
            container.append(x)
            x = stack(container)

            B, C, D, H, W = x.shape()
        else:
            raise RuntimeError(
                f"shape {x.shape()} not valid for Conv2d"
            )

        C_out = self.out_channels
        C_in = self.in_channels
        K_D = self.kernel_size[0]
        K_H = self.kernel_size[1]
        K_W = self.kernel_size[2]

        S = self.stride
        P = self.padding
        D = self.dilation

        D_out = (H + 2*P[0] - D[0]*(K_H-1) - 1) // S[0] + 1
        H_out = (H + 2*P[1] - D[1]*(K_H-1) - 1) // S[1] + 1
        W_out = (W + 2*P[2] - D[2]*(K_W-1) - 1) // S[2] + 1
        
        rows = []
        for b in range(B):
            for i in range(D_out):
                for j in range(H_out):
                    for k in range(W_out):
                        patch = []

                        start_D = i * S[0] - P[0]
                        start_H = j * S[1] - P[1]
                        start_W = k * S[2] - P[2]

                        for c in range(C_in):
                            for t in range(K_D):
                                for u in range(K_H):
                                    for v in range(K_W):
                                        D_index = start_D + D[0] * t
                                        H_index = start_H + D[1] * u
                                        W_index = start_W + D[2] * v
                                        
                                        if 0 <= H_index < H and 0 <= W_index < W:
                                            patch.append(x[b, c, D_index, H_index, W_index])
                                        else:
                                            patch.append(Tensor(0.0))

                        rows.append(stack(patch))

        X_col = stack(rows)

        W_flat = self.weight.reshape((C_out, C_in * K_D * K_H * K_W))
        out_flat = X_col @ W_flat.T()

        out = out_flat.reshape((B, D_out, H_out, W_out, C_out)).T((0, 4, 1, 2, 3))

        if self.bias is not None:
            out = out + self.bias

        return out




