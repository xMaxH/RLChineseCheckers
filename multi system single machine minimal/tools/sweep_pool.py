#!/usr/bin/env python3
"""Sweep the mcts_pool agent and deploy it if it is competitive.

mcts_pool = AlphaZero MCTS restricted to the heuristic candidate-move pool,
with the RL-trained value network evaluating leaves. No hand-crafted rollouts
decide the move -- the maximally defensible RL agent.

Sweeps n_sim at 2/4/6p, measures value_rollout as the reference, and deploys
mcts_pool when its average win-rate is within LOSS_BUDGET of value_rollout
(the user accepts ~10% loss for a more defensible RL agent). Verifies and
writes runs/POOL_SWEEP_SUMMARY.txt.

    PYTHONPATH=. python tools/sweep_pool.py
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

SUMMARY = "runs/POOL_SWEEP_SUMMARY.txt"
DEPLOY = "runs/mcts_deploy.json"
LOSS_BUDGET = 0.10        # deploy mcts_pool if within this avg win-rate of value_rollout
TIMING_BUDGET = 1.6       # max move-seconds for a config to be tournament-safe
N_SIMS = [200, 400, 800]
GAMES = {2: 36, 4: 30, 6: 24}
VERIFY_GAMES = {2: 60, 4: 50, 6: 40}


def pool_cfg(n_sim, c_puct=1.5):
    return MCTSConfig(n_sim=n_sim, c_puct=c_puct, batch_leaves=32,
                      dirichlet_alpha=0.0, dirichlet_eps=0.0, virtual_loss=1.0,
                      shaping_enabled=True, shaping_scale=0.15,
                      shaping_goal_weight=0.5, restrict_to_pool=True)


def play(chooser, players, games, seed):
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
        cs = _compute_player_score(sim, cand, sim.move_count_by_colour[cand])
        opp = max(_compute_player_score(sim, c, sim.move_count_by_colour[c])['final_score']
                  for c in sim.colours if c != cand)
        margins.append(cs['final_score'] - opp)
    a = np.array(mt) if mt else np.array([0.0])
    return dict(winrate=wins / games, wins=wins, games=games, finished=finished,
                maxm=maxm, margin=float(np.mean(margins)), mt_mean=float(a.mean()),
                mt_p95=float(np.percentile(a, 95)), mt_max=float(a.max()))


def main():
    azm._ensure_loaded()
    log = open(SUMMARY, "w", buffering=1)

    def w(s=""):
        print(s, flush=True)
        log.write(s + "\n")

    w(f"# MCTS_POOL SWEEP — started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w("# mcts_pool = AlphaZero MCTS over the heuristic candidate pool; RL value net evaluates leaves")
    w("")

    # ---- reference: value_rollout (current deployed agent / fallback) ----
    w("## 1. REFERENCE  value_rollout (vs heuristic)")
    vr = {}
    for pl in (2, 4, 6):
        try:
            r = play(azm._choose_value_rollout, pl, GAMES[pl], seed=6000 + pl)
            vr[pl] = r
            w(f"  value_rollout {pl}p | winrate={r['winrate']:.3f} ({r['wins']}/{r['games']}) "
              f"| move_s max={r['mt_max']:.2f}")
        except Exception as e:
            w(f"  value_rollout {pl}p ERROR: {e}")
            traceback.print_exc()
    vr_avg = np.mean([vr[pl]['winrate'] for pl in vr]) if vr else 0.0
    w(f"  value_rollout avg win-rate = {vr_avg:.3f}")
    w("")

    # ---- mcts_pool sweep over n_sim ----
    w("## 2. MCTS_POOL  (n_sim sweep, vs heuristic)")
    cells = {}
    for ns in N_SIMS:
        azm._MCTS_POOL_CFG = pool_cfg(ns)
        for pl in (2, 4, 6):
            t0 = time.time()
            try:
                r = play(azm._choose_mcts_pool, pl, GAMES[pl], seed=7000 + pl)
                cells[(ns, pl)] = r
                w(f"  mcts_pool n_sim={ns:>4} {pl}p | winrate={r['winrate']:.3f} "
                  f"({r['wins']}/{r['games']}) finished={r['finished']} maxmoves={r['maxm']} "
                  f"margin={r['margin']:.0f} | move_s mean={r['mt_mean']:.2f} "
                  f"p95={r['mt_p95']:.2f} max={r['mt_max']:.2f} | {time.time()-t0:.0f}s")
            except Exception as e:
                w(f"  mcts_pool n_sim={ns} {pl}p ERROR: {e}")
                traceback.print_exc()
        w("")

    # ---- pick best timing-safe mcts_pool config ----
    w("## 3. PICK")
    best = None
    for ns in N_SIMS:
        rs = [cells.get((ns, pl)) for pl in (2, 4, 6)]
        if any(r is None for r in rs):
            continue
        mx = max(r['mt_max'] for r in rs)
        avg = np.mean([r['winrate'] for r in rs])
        safe = mx < TIMING_BUDGET
        w(f"   n_sim={ns:>4} | avg winrate={avg:.3f} | max move={mx:.2f}s "
          f"{'OK' if safe else 'TOO SLOW'}")
        if safe and (best is None or avg > best[0]):
            best = (avg, ns, mx)
    w("")

    # ---- deploy decision ----
    w("## 4. DEPLOY DECISION")
    if best is None:
        w("   no timing-safe mcts_pool config — keeping value_rollout.")
        deployed = "value_rollout"
    else:
        pool_avg, b_ns, b_mx = best
        gap = vr_avg - pool_avg
        w(f"   best mcts_pool: n_sim={b_ns}  avg winrate={pool_avg:.3f}  "
          f"(value_rollout {vr_avg:.3f}, gap {gap:+.3f}; budget {LOSS_BUDGET})")
        if gap <= LOSS_BUDGET:
            cfg = {"mode": "mcts_pool", "n_sim": int(b_ns), "c_puct": 1.5,
                   "shaping": True,
                   "note": "AlphaZero MCTS over heuristic pool; RL value net evaluates leaves",
                   "written": time.strftime('%Y-%m-%d %H:%M:%S')}
            with open(DEPLOY, "w") as f:
                json.dump(cfg, f, indent=2)
            deployed = "mcts_pool"
            w(f"   within budget -> DEPLOYED mcts_pool n_sim={b_ns} "
              f"(the more-defensible RL agent).")
        else:
            deployed = "value_rollout"
            w(f"   gap {gap:.3f} exceeds budget {LOSS_BUDGET} -> kept value_rollout. "
              f"Review manually.")
    w("")

    # ---- verify the deployed agent ----
    w(f"## 5. VERIFY  ({deployed}, large samples)")
    try:
        if deployed == "mcts_pool":
            azm._MCTS_POOL_CFG = pool_cfg(best[1])
            chooser = azm._choose_mcts_pool
        else:
            chooser = azm._choose_value_rollout
        for pl in (2, 4, 6):
            t0 = time.time()
            r = play(chooser, pl, VERIFY_GAMES[pl], seed=9700 + pl)
            w(f"   {deployed} {pl}p: winrate={r['winrate']:.3f} ({r['wins']}/{r['games']}) "
              f"finished={r['finished']} maxmoves={r['maxm']} | move_s mean={r['mt_mean']:.2f} "
              f"p95={r['mt_p95']:.2f} max={r['mt_max']:.2f} | {time.time()-t0:.0f}s")
    except Exception as e:
        w(f"   VERIFY ERROR: {e}")
        traceback.print_exc()
    w("")
    w(f"# DONE  {time.strftime('%Y-%m-%d %H:%M:%S')}  deployed={deployed}")
    print("POOL SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
