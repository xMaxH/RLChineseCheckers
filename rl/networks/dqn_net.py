"""MLP Q-network for Chinese Checkers DQN."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class DQNNet(nn.Module):
    """Fully-connected Q-network.

    Input  : flat observation of size ``in_features`` (default 6*121=726).
    Output : Q-values over ``num_actions`` (default 10*121=1210).
    """

    def __init__(
        self,
        in_features: int,
        num_actions: int,
        hidden_sizes: Sequence[int] = (512, 512),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
