"""Metrics logger.

Writes a CSV of per-episode stats and keeps rolling-window summaries
in memory. Optionally renders a summary plot on demand.
"""

from __future__ import annotations

import csv
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class EpisodeStats:
    episode: int
    stage: str
    steps: int
    agent_moves: int
    return_sum: float
    pins_home: int
    outcome: str
    epsilon: float = 0.0
    loss: Optional[float] = None
    loop_revisit_events: int = 0
    loop_aba_events: int = 0


class MetricsLogger:
    def __init__(self, out_dir: str, window: int = 200):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.csv_path = os.path.join(self.out_dir, "episodes.csv")
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "episode",
                "stage",
                "steps",
                "agent_moves",
                "return_sum",
                "pins_home",
                "outcome",
                "epsilon",
                "loss",
                "loop_revisit_events",
                "loop_aba_events",
            ],
        )
        if self._csv_file.tell() == 0:
            self._writer.writeheader()
            self._csv_file.flush()

        self.window = int(window)
        self._recent: Deque[EpisodeStats] = deque(maxlen=self.window)
        self._all: List[EpisodeStats] = []

    # ------------------------------------------------------------------
    def log(self, stats: EpisodeStats) -> None:
        self._recent.append(stats)
        self._all.append(stats)
        self._writer.writerow(
            {
                "episode": stats.episode,
                "stage": stats.stage,
                "steps": stats.steps,
                "agent_moves": stats.agent_moves,
                "return_sum": f"{stats.return_sum:.4f}",
                "pins_home": stats.pins_home,
                "outcome": stats.outcome,
                "epsilon": f"{stats.epsilon:.4f}",
                "loss": "" if stats.loss is None else f"{stats.loss:.6f}",
                "loop_revisit_events": stats.loop_revisit_events,
                "loop_aba_events": stats.loop_aba_events,
            }
        )
        self._csv_file.flush()

    # ------------------------------------------------------------------
    def rolling(self) -> Dict[str, float]:
        if not self._recent:
            return {}
        wins = sum(1 for s in self._recent if s.outcome == "win")
        moves = [s.agent_moves for s in self._recent if s.outcome == "win"]
        returns = [s.return_sum for s in self._recent]
        pins = [s.pins_home for s in self._recent]
        loop_revisit = [s.loop_revisit_events for s in self._recent]
        loop_aba = [s.loop_aba_events for s in self._recent]
        return {
            "n": len(self._recent),
            "win_rate": wins / len(self._recent),
            "avg_moves_if_win": (sum(moves) / len(moves)) if moves else float("nan"),
            "avg_return": sum(returns) / len(returns),
            "avg_pins_home": sum(pins) / len(pins),
            "avg_loop_revisit_events": sum(loop_revisit) / len(loop_revisit),
            "avg_loop_aba_events": sum(loop_aba) / len(loop_aba),
        }

    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def plot(self, path: Optional[str] = None) -> Optional[str]:
        """Render matplotlib summary; returns saved path if available."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return None
        if not self._all:
            return None

        eps = [s.episode for s in self._all]
        returns = [s.return_sum for s in self._all]
        pins = [s.pins_home for s in self._all]
        moves = [s.agent_moves for s in self._all]
        win_flags = [1 if s.outcome == "win" else 0 for s in self._all]

        # Rolling win rate
        w = 100
        roll = []
        total = 0
        buf: Deque[int] = deque(maxlen=w)
        for x in win_flags:
            buf.append(x)
            roll.append(sum(buf) / len(buf))

        fig, axes = plt.subplots(2, 2, figsize=(11, 6))
        axes[0, 0].plot(eps, returns)
        axes[0, 0].set_title("Return per episode")
        axes[0, 1].plot(eps, pins)
        axes[0, 1].set_title("Pins in goal")
        axes[1, 0].plot(eps, moves)
        axes[1, 0].set_title("Agent moves")
        axes[1, 1].plot(eps, roll)
        axes[1, 1].set_title(f"Rolling win-rate (w={w})")
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out = path or os.path.join(self.out_dir, "summary.png")
        fig.savefig(out, dpi=100)
        plt.close(fig)
        return out
