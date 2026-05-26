#!/usr/bin/env python3
"""Measure the ACTUAL tournament agent (alphazero_method) vs the heuristic.

Unlike az/eval.py (which always runs full MCTS), this exercises the real
move-chooser that alphazero_method.choose_move_alphazero dispatches to, so
the win-rate and per-move timing reflect what the tournament will see.

Run from the working dir:
    python tools/eval_tournament.py --mode mcts_pool --games 8 --players 2
"""
import argparse
import random
import time

import numpy as np

from az.config import MCTSConfig
from az.sim import Sim
from az.heuristic import heuristic_choose_move
from az.eval import _compute_player_score
import alphazero_method as azm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="mcts_pool",
                   help="value_rollout | value_pool | mcts_pool | heuristic_rollout | heuristic_pool | raw | mcts")
    p.add_argument("--n-sim", type=int, default=None,
                   help="Override MCTS simulations for mcts/mcts_pool")
    p.add_argument("--root-only", action="store_true",
                   help="For mcts_pool: apply the heuristic pool only at the root")
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--max-moves", type=int, default=400)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()

    azm._ensure_loaded()
    if a.n_sim is not None:
        cfg_kwargs = dict(
            n_sim=a.n_sim,
            c_puct=1.5,
            batch_leaves=32,
            dirichlet_alpha=0.0,
            dirichlet_eps=0.0,
            virtual_loss=1.0,
            shaping_enabled=True,
            shaping_scale=0.15,
            shaping_goal_weight=0.5,
        )
        azm._MCTS_CFG = MCTSConfig(**cfg_kwargs)
        azm._MCTS_POOL_CFG = MCTSConfig(
            **cfg_kwargs,
            restrict_to_pool=True,
            restrict_pool_root_only=a.root_only,
        )
    choosers = {
        "heuristic_rollout": azm._choose_heuristic_pool_rollout,
        "heuristic_pool": azm._choose_heuristic_pool_rerank,
        "value_rollout": azm._choose_value_rollout,
        "value_pool": azm._choose_value_pool,
        "mcts_pool": azm._choose_mcts_pool,
        "raw": azm._choose_raw_policy,
        "mcts": azm._choose_mcts,
    }
    chooser = choosers[a.mode]
    rng = random.Random(a.seed)

    wins = finished = max_moves_hit = 0
    margins, move_times = [], []
    t_start = time.perf_counter()
    for g in range(a.games):
        sim = Sim(a.players, seed=rng.randrange(2 ** 31))
        cand = sim.colours[g % a.players]
        while not sim.is_terminal:
            col = sim.current_colour()
            legal = sim.legal_moves(col)
            if not any(legal.values()):
                sim.skip_no_moves()
                continue
            if col == cand:
                t0 = time.perf_counter()
                pid, to = chooser(sim, col, legal)
                move_times.append(time.perf_counter() - t0)
            else:
                pid, to = heuristic_choose_move(sim, col, legal, rng=rng)
            sim.apply_move(pid, to)
            if sim.move_count >= a.max_moves and not sim.is_terminal:
                sim.force_max_moves()
        if sim.terminal_reason in ("WIN", "DRAW_CHAIN"):
            finished += 1
            if sim.winner == cand:
                wins += 1
        if sim.terminal_reason == "MAX_MOVES":
            max_moves_hit += 1
        cs = _compute_player_score(sim, cand, sim.move_count_by_colour[cand])
        opp = max(_compute_player_score(sim, c, sim.move_count_by_colour[c])["final_score"]
                  for c in sim.colours if c != cand)
        margins.append(cs["final_score"] - opp)
        print(f"  game {g + 1}/{a.games}: term={sim.terminal_reason} "
              f"winner={sim.winner} cand={'WIN' if sim.winner == cand else '-'}")

    mt = np.array(move_times)
    print(f"\nmode={a.mode}  games={a.games}  players={a.players}  "
          f"({time.perf_counter() - t_start:.0f}s wall)")
    print(f"  win_rate={wins / a.games:.3f}  wins={wins}/{a.games}  "
          f"finished={finished}  max_moves={max_moves_hit}")
    print(f"  score_margin_mean={np.mean(margins):.0f}")
    print(f"  move_time  mean={mt.mean() * 1000:.0f}ms  p50={np.percentile(mt, 50) * 1000:.0f}ms  "
          f"p95={np.percentile(mt, 95) * 1000:.0f}ms  max={mt.max() * 1000:.0f}ms  n={len(mt)}")
    print(f"  moves over 2.0s budget: {(mt > 2.0).sum()}/{len(mt)}  "
          f"over 1.5s: {(mt > 1.5).sum()}/{len(mt)}")


if __name__ == "__main__":
    main()
