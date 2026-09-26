import torch
from torch import nn


class Residual(nn.Module):

    def __init__(self, res: nn.Module):
        super().__init__()
        self.res = res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res(x)
