"""FIFO replay buffer. Samples are added per-game so we can drop entire
games whose terminal_reason is not WIN/DRAW_CHAIN."""

from typing import List, Dict, Optional
from collections import deque
import numpy as np

from .config import NUM_CELLS, BOARD_CHANNELS, NUM_ACTIONS, MAX_PLAYERS


class Sample:
    __slots__ = ('board', 'glob', 'pi', 'v')

    def __init__(self, board: np.ndarray, glob: np.ndarray, pi: np.ndarray, v: np.ndarray):
        self.board = board    # (BOARD_CHANNELS, 121) float32
        self.glob = glob      # (8,) float32
        self.pi = pi          # (1210,) float32 — visit-count distribution (normalized)
        self.v = v            # (6,) float32 — outcome from canonical to-move's perspective; NaN for absent slots


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: deque[Sample] = deque(maxlen=capacity)

    def add_game(self, samples: List[Sample]) -> int:
        """Add all samples from a single game. Returns count added.

        Caller is responsible for filtering out non-WIN games before calling.
        """
        for s in samples:
            self.buf.append(s)
        return len(samples)

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """Uniform-random sample of `n` items. Returns dict of stacked arrays."""
        idxs = rng.integers(0, len(self.buf), size=n)
        boards = np.empty((n, BOARD_CHANNELS, NUM_CELLS), dtype=np.float32)
        globs = np.empty((n, 8), dtype=np.float32)
        pis = np.empty((n, NUM_ACTIONS), dtype=np.float32)
        vs = np.empty((n, MAX_PLAYERS), dtype=np.float32)
        for i, idx in enumerate(idxs):
            s = self.buf[idx]
            boards[i] = s.board
            globs[i] = s.glob
            pis[i] = s.pi
            vs[i] = s.v
        return {"boards": boards, "globs": globs, "pis": pis, "vs": vs}
