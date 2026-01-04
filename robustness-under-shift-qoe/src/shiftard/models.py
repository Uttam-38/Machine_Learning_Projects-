import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in: int, hidden_sizes=(64, 64), dropout=0.1):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
