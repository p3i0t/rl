from typing import Iterable
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_size: int, hidden_sizes: Iterable[int], out_size: int):
        super().__init__()
        sizes = [in_size] + list(hidden_sizes) + [out_size]

        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers[:-1])

    def forward(self, x):
        return self.model(x)