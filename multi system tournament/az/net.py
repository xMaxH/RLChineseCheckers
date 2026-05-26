"""ResNet trunk + policy head + MaxN 6-vector value head.

Input:
  board: (B, BOARD_CHANNELS, 121) float
  glob:  (B, 8) float
Output:
  policy_logits: (B, 1210)
  value:         (B, 6) tanh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NetConfig, NUM_CELLS


class ResBlock(nn.Module):
    def __init__(self, width: int, mix_cells: bool):
        super().__init__()
        self.c1 = nn.Conv1d(width, width, kernel_size=1, bias=False)
        self.b1 = nn.BatchNorm1d(width)
        self.c2 = nn.Conv1d(width, width, kernel_size=1, bias=False)
        self.b2 = nn.BatchNorm1d(width)
        self.mix_cells = mix_cells
        if mix_cells:
            # Learned NUM_CELLS x NUM_CELLS mixing matrix to propagate info across hex cells.
            self.mixer = nn.Parameter(torch.zeros(NUM_CELLS, NUM_CELLS))
            nn.init.xavier_uniform_(self.mixer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        if self.mix_cells:
            h = h + torch.matmul(h, self.mixer)
        return F.relu(x + h)


class AZNet(nn.Module):
    def __init__(self, cfg: NetConfig = NetConfig()):
        super().__init__()
        self.cfg = cfg
        self.stem_conv = nn.Conv1d(cfg.in_channels, cfg.width, kernel_size=1, bias=False)
        self.stem_bn = nn.BatchNorm1d(cfg.width)
        # Learned positional embedding shape (width, 121)
        self.pos = nn.Parameter(torch.zeros(cfg.width, NUM_CELLS))
        nn.init.normal_(self.pos, std=0.02)
        # Trunk: 8 residual blocks, alternating with mixer
        self.blocks = nn.ModuleList([
            ResBlock(cfg.width, mix_cells=(i % 2 == 1)) for i in range(cfg.blocks)
        ])
        # Policy head
        self.pol_conv = nn.Conv1d(cfg.width, cfg.policy_head_channels, kernel_size=1, bias=False)
        self.pol_bn = nn.BatchNorm1d(cfg.policy_head_channels)
        self.pol_fc = nn.Linear(cfg.policy_head_channels * NUM_CELLS, cfg.num_actions)
        # Value head
        self.val_fc1 = nn.Linear(cfg.width + cfg.global_dim, cfg.value_hidden)
        self.val_fc2 = nn.Linear(cfg.value_hidden, cfg.num_player_slots)

    def forward(self, board: torch.Tensor, glob: torch.Tensor):
        x = self.stem_conv(board)               # (B, W, 121)
        x = x + self.pos.unsqueeze(0)           # add positional
        x = F.relu(self.stem_bn(x))
        for blk in self.blocks:
            x = blk(x)
        # Policy
        p = F.relu(self.pol_bn(self.pol_conv(x)))
        p = p.flatten(1)
        policy_logits = self.pol_fc(p)
        # Value
        v_pool = x.mean(dim=2)                  # (B, W)
        v = torch.cat([v_pool, glob], dim=1)
        v = F.relu(self.val_fc1(v))
        value = torch.tanh(self.val_fc2(v))
        return policy_logits, value
