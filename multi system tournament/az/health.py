"""Per-chunk health metrics + auto-kill flags."""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
import json
import os


@dataclass
class HealthMonitor:
    log_path: str
    history_len: int = 20
    chunks_with_zero_wins: int = 0
    chunks_with_no_replay_growth: int = 0
    last_replay_size: int = 0
    eval_unique_streak_below: int = 0
    eval_margin_history: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    value_loss_history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    kill_reason: Optional[str] = None

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

    def record_chunk(self, metrics: Dict) -> None:
        # Append to jsonl
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        wins = metrics.get("chunk_wins", 0)
        replay_size = metrics.get("replay_size", 0)
        kept_samples = metrics.get("chunk_kept_samples", 0)
        # Don't penalise bootstrap/dagger chunks for having no WIN-terminated games —
        # the model starts random and DAgger intentionally collects MAX_MOVES samples.
        if not metrics.get("bootstrap", False):
            if wins == 0:
                self.chunks_with_zero_wins += 1
            else:
                self.chunks_with_zero_wins = 0
        if kept_samples > 0:
            self.chunks_with_no_replay_growth = 0
        elif replay_size <= self.last_replay_size:
            self.chunks_with_no_replay_growth += 1
        else:
            self.chunks_with_no_replay_growth = 0
        self.last_replay_size = replay_size

        # Skip recording val=0.0 from bootstrap/dagger chunks — those have all-NaN value
        # targets so the masked loss is trivially 0, which would poison the divergence check.
        if "value_loss" in metrics and not metrics.get("bootstrap", False):
            self.value_loss_history.append(metrics["value_loss"])

        # Auto-kill rules
        if self.chunks_with_zero_wins >= 10:
            self.kill_reason = (
                f"0 WIN-terminated games for {self.chunks_with_zero_wins} consecutive chunks"
            )
        elif self.chunks_with_no_replay_growth >= 5:
            self.kill_reason = (
                f"replay didn't grow for {self.chunks_with_no_replay_growth} consecutive chunks"
            )
        elif len(self.value_loss_history) >= 5:
            recent_min = min(self.value_loss_history)
            if metrics.get("value_loss", 0.0) > 2 * recent_min and metrics.get("value_loss", 0.0) > 1.0:
                self.kill_reason = (
                    f"value_loss diverging: now {metrics['value_loss']:.3f} > 2x recent_min {recent_min:.3f}"
                )

    def record_eval(self, metrics: Dict) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"eval": metrics}) + "\n")
        # Eval-based kills disabled — run to completion and inspect results manually.

    def should_stop(self) -> bool:
        return self.kill_reason is not None
