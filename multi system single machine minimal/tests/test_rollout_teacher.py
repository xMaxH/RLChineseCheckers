"""Tests for rollout-teacher policy labels."""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from az.config import SelfPlayConfig
from az.selfplay import _teacher_policy_target
from az.sim import Sim


def _midgame(seed: int = 11) -> Sim:
    sim = Sim(2, seed=seed)
    rng = random.Random(seed)
    for _ in range(12):
        col = sim.current_colour()
        legal = sim.legal_moves(col)
        movable = [(pid, dests) for pid, dests in legal.items() if dests]
        if not movable:
            sim.skip_no_moves()
            continue
        pid, dests = rng.choice(movable)
        sim.apply_move(pid, rng.choice(dests))
    return sim


def test_rollout_teacher_is_deterministic():
    sim = _midgame()
    col = sim.current_colour()
    legal = sim.legal_moves(col)
    cfg = SelfPlayConfig(
        heuristic_rollout_targets=True,
        heuristic_rollouts_per_move=1,
        heuristic_rollout_pool_cap=8,
        heuristic_rollout_score_temperature=250.0,
    )

    pid1, to1, pi1 = _teacher_policy_target(
        sim, col, legal, random.Random(1), cfg, max_moves=200,
    )
    pid2, to2, pi2 = _teacher_policy_target(
        sim, col, legal, random.Random(999), cfg, max_moves=200,
    )

    assert (pid1, to1) == (pid2, to2)
    assert np.array_equal(pi1, pi2)
    assert np.isclose(float(pi1.sum()), 1.0)


if __name__ == "__main__":
    test_rollout_teacher_is_deterministic()
    print("test_rollout_teacher OK")
