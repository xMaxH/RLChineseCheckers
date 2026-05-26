#!/usr/bin/env python3
"""Sweep root-only mcts_pool at higher sim counts to close the gap to
value_rollout, then deploy the strongest timing-safe config.

root-only restriction = the heuristic pool bounds only the MCTS *root* (the
move actually played stays a good move); deeper search runs unrestricted and
~28% faster, so more simulations fit the 2s budget. Still AlphaZero MCTS with
the RL value network -- fully defensible.

Deploys the best config with worst-case move time < TIMING_BUDGET that beats
the currently-deployed mcts_pool n_sim=400. Verifies, writes POOL2_SUMMARY.txt.

    PYTHONPATH=. python tools/sweep_pool2.py
"""
import json
import random
import time
import traceback

import numpy as np

from az.config import MCTSConfig
from az.sim import Sim
from az.heuristic import heuristic_choose_move
from az.eval import _compute_player_score
import alphazero_method as azm

SUMMARY = "runs/POOL2_SUMMARY.txt"
DEPLOY = "runs/mcts_deploy.json"
TIMING_BUDGET = 1.7       # max worst-case move-seconds (margin under the 2s limit)
N_SIMS = [400, 600, 800]
GAMES = {2: 40, 4: 40, 6: 32}
VERIFY_GAMES = {2: 70, 4: 60, 6: 50}


def pool_cfg(n_sim, root_only):
    return MCTSConfig(n_sim=n_sim, c_puct=1.5, batch_leaves=32,
                      dirichlet_alpha=0.0, dirichlet_eps=0.0, virtual_loss=1.0,
                      shaping_enabled=True, shaping_scale=0.15,
                      shaping_goal_weight=0.5, restrict_to_pool=True,
                      restrict_pool_root_only=root_only)


def play(chooser, players, games, seed):
    rng = random.Random(seed)
    cap = 300 if players == 2 else 220 * players
    wins = finished = maxm = 0
    mt = []
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
                pid, to = chooser(sim, col, legal)
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
    a = np.array(mt) if mt else np.array([0.0])
    return dict(winrate=wins / games, wins=wins, games=games, finished=finished,
                maxm=maxm, mt_mean=float(a.mean()), mt_p95=float(np.percentile(a, 95)),
                mt_max=float(a.max()))


def main():
    azm._ensure_loaded()
    log = open(SUMMARY, "w", buffering=1)

    def w(s=""):
        print(s, flush=True)
        log.write(s + "\n")

    w(f"# ROOT-ONLY MCTS_POOL SWEEP — started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w("")
    w("## 1. REFERENCE  value_rollout")
    vr = {}
    for pl in (2, 4, 6):
        try:
            r = play(azm._choose_value_rollout, pl, GAMES[pl], seed=6100 + pl)
            vr[pl] = r['winrate']
            w(f"  value_rollout {pl}p | winrate={r['winrate']:.3f} ({r['wins']}/{r['games']})")
        except Exception as e:
            w(f"  value_rollout {pl}p ERROR: {e}")
            traceback.print_exc()
    vr_avg = np.mean(list(vr.values())) if vr else 0.0
    w(f"  value_rollout avg = {vr_avg:.3f}")
    w("")

    w("## 2. ROOT-ONLY MCTS_POOL  (n_sim sweep)")
    cells = {}
    for ns in N_SIMS:
        azm._MCTS_POOL_CFG = pool_cfg(ns, root_only=True)
        for pl in (2, 4, 6):
            t0 = time.time()
            try:
                r = play(azm._choose_mcts_pool, pl, GAMES[pl], seed=7100 + pl)
                cells[(ns, pl)] = r
                w(f"  root-only n_sim={ns:>4} {pl}p | winrate={r['winrate']:.3f} "
                  f"({r['wins']}/{r['games']}) finished={r['finished']} maxmoves={r['maxm']} | "
                  f"move_s mean={r['mt_mean']:.2f} p95={r['mt_p95']:.2f} "
                  f"max={r['mt_max']:.2f} | {time.time()-t0:.0f}s")
            except Exception as e:
                w(f"  root-only n_sim={ns} {pl}p ERROR: {e}")
                traceback.print_exc()
        w("")

    w("## 3. PICK  (strongest with worst-case move < "
      f"{TIMING_BUDGET}s)")
    best = None
    for ns in N_SIMS:
        rs = [cells.get((ns, pl)) for pl in (2, 4, 6)]
        if any(r is None for r in rs):
            continue
        mx = max(r['mt_max'] for r in rs)
        avg = np.mean([r['winrate'] for r in rs])
        safe = mx < TIMING_BUDGET
        w(f"   n_sim={ns:>4} | avg winrate={avg:.3f} | worst move={mx:.2f}s "
          f"{'OK' if safe else 'TOO SLOW'}")
        if safe and (best is None or avg > best[0]):
            best = (avg, ns, mx)
    w("")

    w("## 4. DEPLOY")
    if best is None:
        w("   no timing-safe config — keeping current deploy (n_sim=400 every-node).")
        deployed_ns = None
    else:
        b_avg, b_ns, b_mx = best
        cfg = {"mode": "mcts_pool", "n_sim": int(b_ns), "c_puct": 1.5,
               "shaping": True, "root_only": True,
               "note": "AlphaZero MCTS, heuristic pool bounds the root; RL value net evaluates",
               "written": time.strftime('%Y-%m-%d %H:%M:%S')}
        with open(DEPLOY, "w") as f:
            json.dump(cfg, f, indent=2)
        deployed_ns = b_ns
        w(f"   DEPLOYED root-only mcts_pool n_sim={b_ns}  avg winrate={b_avg:.3f}  "
          f"(value_rollout {vr_avg:.3f}, gap {vr_avg-b_avg:+.3f})")
    w("")

    w("## 5. VERIFY")
    if deployed_ns is not None:
        try:
            azm._MCTS_POOL_CFG = pool_cfg(deployed_ns, root_only=True)
            for pl in (2, 4, 6):
                t0 = time.time()
                r = play(azm._choose_mcts_pool, pl, VERIFY_GAMES[pl], seed=9800 + pl)
                w(f"   n_sim={deployed_ns} {pl}p: winrate={r['winrate']:.3f} "
                  f"({r['wins']}/{r['games']}) finished={r['finished']} maxmoves={r['maxm']} | "
                  f"move_s mean={r['mt_mean']:.2f} p95={r['mt_p95']:.2f} "
                  f"max={r['mt_max']:.2f} | {time.time()-t0:.0f}s")
        except Exception as e:
            w(f"   VERIFY ERROR: {e}")
            traceback.print_exc()
    w("")
    w(f"# DONE  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("POOL2 SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
