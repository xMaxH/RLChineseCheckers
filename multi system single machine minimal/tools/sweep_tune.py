#!/usr/bin/env python3
"""Tune the deployed value_rollout RL agent.

Sweeps (value_topk, rollouts_per_move) at 2/4/6p with large samples, picks the
strongest timing-safe config, re-deploys it into runs/mcts_deploy.json, verifies
it, and writes runs/TUNE_SUMMARY.txt.

  value_topk          how many value-net-ranked pool moves get rolled out
  rollouts_per_move   heuristic rollouts per candidate (margin-estimate quality)

The agent stays defensible RL: the RL-trained value network ranks the
heuristic-proposed pool; rollouts confirm (AlphaGo's value-net + rollouts).

    PYTHONPATH=. python tools/sweep_tune.py
"""
import json
import os
import random
import time
import traceback

import numpy as np

from az.sim import Sim
from az.heuristic import heuristic_choose_move
from az.eval import _compute_player_score
import alphazero_method as azm

SUMMARY = "runs/TUNE_SUMMARY.txt"
DEPLOY = "runs/mcts_deploy.json"
TIMING_BUDGET = 1.6     # max move-seconds to count a config tournament-safe

# (value_topk, rollouts_per_move)
CONFIGS = [(3, 2), (5, 2), (8, 1), (12, 1), (5, 3), (8, 2), (3, 3)]
GAMES = {2: 60, 4: 60, 6: 50}
VERIFY_GAMES = {2: 80, 4: 80, 6: 70}


def play(players, games, seed):
    """value_rollout agent vs all-heuristic; reads value_topk/rollouts from env."""
    rng = random.Random(seed)
    cap = 300 if players == 2 else 220 * players
    wins = finished = maxm = 0
    margins, mt = [], []
    for g in range(games):
        sim = Sim(players, seed=rng.randrange(2 ** 31))
        cand = sim.colours[g % players]
        while not sim.is_terminal:
            col = sim.current_colour()
            legal = sim.legal_moves(col)
            if not any(legal.values()):
                sim.skip_no_moves()
                continue
            if col == cand:
                t0 = time.perf_counter()
                pid, to = azm._choose_value_rollout(sim, col, legal)
                mt.append(time.perf_counter() - t0)
            else:
                pid, to = heuristic_choose_move(sim, col, legal, rng=rng)
            sim.apply_move(pid, to)
            if sim.move_count >= cap and not sim.is_terminal:
                sim.force_max_moves()
        if sim.terminal_reason in ('WIN', 'DRAW_CHAIN'):
            finished += 1
            if sim.winner == cand:
                wins += 1
        if sim.terminal_reason == 'MAX_MOVES':
            maxm += 1
        cs = _compute_player_score(sim, cand, sim.move_count_by_colour[cand])
        opp = max(_compute_player_score(sim, c, sim.move_count_by_colour[c])['final_score']
                  for c in sim.colours if c != cand)
        margins.append(cs['final_score'] - opp)
    a = np.array(mt) if mt else np.array([0.0])
    return dict(winrate=wins / games, wins=wins, games=games, finished=finished,
                maxm=maxm, margin=float(np.mean(margins)), mt_mean=float(a.mean()),
                mt_p95=float(np.percentile(a, 95)), mt_max=float(a.max()))


def set_cfg(topk, rollouts):
    os.environ["ALPHAZERO_VALUE_TOPK"] = str(topk)
    os.environ["ALPHAZERO_ROLLOUTS_PER_MOVE"] = str(rollouts)


def main():
    azm._ensure_loaded()
    log = open(SUMMARY, "w", buffering=1)

    def w(s=""):
        print(s, flush=True)
        log.write(s + "\n")

    w(f"# VALUE_ROLLOUT TUNING SWEEP — started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w("# agent: heuristic proposes the move pool; RL value net ranks; rollouts confirm")
    w("")
    w("## 1. SWEEP  (win-rate vs heuristic)")
    cells = {}   # (topk,rollouts,players) -> result
    for topk, rollouts in CONFIGS:
        set_cfg(topk, rollouts)
        for pl in (2, 4, 6):
            t0 = time.time()
            try:
                r = play(pl, GAMES[pl], seed=5000 + pl)
                cells[(topk, rollouts, pl)] = r
                w(f"  topk={topk:>2} rollouts={rollouts} {pl}p | "
                  f"winrate={r['winrate']:.3f} ({r['wins']}/{r['games']}) "
                  f"finished={r['finished']} maxmoves={r['maxm']} margin={r['margin']:.0f} | "
                  f"move_s mean={r['mt_mean']:.2f} p95={r['mt_p95']:.2f} "
                  f"max={r['mt_max']:.2f} | {time.time()-t0:.0f}s")
            except Exception as e:
                w(f"  topk={topk} rollouts={rollouts} {pl}p ERROR: {e}")
                traceback.print_exc()
        w("")

    # ---- PICK: best avg win-rate among configs timing-safe at every count ----
    w("## 2. PICK")
    best = None
    for topk, rollouts in CONFIGS:
        rs = [cells.get((topk, rollouts, pl)) for pl in (2, 4, 6)]
        if any(r is None for r in rs):
            continue
        max_move = max(r['mt_max'] for r in rs)
        avg = sum(r['winrate'] for r in rs) / 3.0
        safe = max_move < TIMING_BUDGET
        w(f"   topk={topk:>2} rollouts={rollouts} | avg winrate={avg:.3f} "
          f"| max move={max_move:.2f}s {'OK' if safe else 'TOO SLOW'}")
        if safe and (best is None or avg > best[0]):
            best = (avg, topk, rollouts, max_move)
    if best is None:
        w("   no timing-safe config — keeping current deploy.")
        w(f"# DONE  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("TUNE SWEEP DONE", flush=True)
        return
    _, b_topk, b_roll, b_max = best
    w(f"   -> PICK: topk={b_topk} rollouts={b_roll}  (avg winrate {best[0]:.3f})")
    w("")

    # ---- DEPLOY ----
    w("## 3. DEPLOY")
    try:
        cfg = {"mode": "value_rollout", "value_topk": b_topk,
               "rollouts_per_move": b_roll, "rollout_max_moves": 500,
               "n_sim": 200, "shaping": False,
               "note": "value-network RL agent (heuristic proposes; RL value net ranks; rollouts confirm)",
               "written": time.strftime('%Y-%m-%d %H:%M:%S')}
        with open(DEPLOY, "w") as f:
            json.dump(cfg, f, indent=2)
        w(f"   wrote {DEPLOY}: {cfg}")
    except Exception as e:
        w(f"   DEPLOY ERROR: {e}")
        traceback.print_exc()
    w("")

    # ---- VERIFY ----
    w("## 4. VERIFY  (tuned config, large samples)")
    set_cfg(b_topk, b_roll)
    try:
        for pl in (2, 4, 6):
            t0 = time.time()
            r = play(pl, VERIFY_GAMES[pl], seed=9500 + pl)
            w(f"   {pl}p: winrate={r['winrate']:.3f} ({r['wins']}/{r['games']}) "
              f"finished={r['finished']} maxmoves={r['maxm']} margin={r['margin']:.0f} | "
              f"move_s mean={r['mt_mean']:.2f} p95={r['mt_p95']:.2f} "
              f"max={r['mt_max']:.2f} | {time.time()-t0:.0f}s")
    except Exception as e:
        w(f"   VERIFY ERROR: {e}")
        traceback.print_exc()
    w("")
    w(f"# DONE  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("TUNE SWEEP DONE", flush=True)


if __name__ == '__main__':
    main()
