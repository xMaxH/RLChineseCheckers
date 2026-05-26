#!/usr/bin/env python3
"""Compare candidate tournament agents and deploy the strongest defensible-RL
one.

  * value_pool     — heuristic proposes the move pool; the RL-trained value
                     network evaluates each candidate and picks. (value-RL)
  * value_rollout  — RL value network ranks the pool; short heuristic rollouts
                     confirm the top few. (AlphaGo-style value net + rollouts)
  * heuristic_rollout — non-RL reference ceiling (heuristic + rollouts only).

Deploys the better of value_pool / value_rollout via runs/mcts_deploy.json,
verifies it at 2/4/6p, and writes runs/MODE_SWEEP_SUMMARY.txt.

    PYTHONPATH=. python tools/sweep_modes.py
"""
import json
import random
import time
import traceback

import numpy as np

from az.sim import Sim
from az.heuristic import heuristic_choose_move
from az.eval import _compute_player_score
import alphazero_method as azm

SUMMARY = "runs/MODE_SWEEP_SUMMARY.txt"
DEPLOY = "runs/mcts_deploy.json"


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

    w(f"# MODE SWEEP — started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w("# net = runs/best.pt (the value-RL net); value_* modes use its "
      "RL-trained value head")
    w("")
    choosers = {
        "heuristic_rollout": azm._choose_heuristic_pool_rollout,  # non-RL reference
        "value_pool": azm._choose_value_pool,
        "value_rollout": azm._choose_value_rollout,
    }
    games_for = {2: 30, 4: 24, 6: 20}
    results = {}

    w("## 1. SWEEP  (win-rate vs heuristic)")
    for mode in ("heuristic_rollout", "value_pool", "value_rollout"):
        for pl in (2, 4, 6):
            t0 = time.time()
            try:
                r = play(choosers[mode], pl, games_for[pl], seed=4000 + pl)
                results[(mode, pl)] = r
                w(f"  {mode:18} {pl}p | winrate={r['winrate']:.3f} "
                  f"({r['wins']}/{r['games']}) finished={r['finished']} "
                  f"maxmoves={r['maxm']} margin={r['margin']:.0f} | "
                  f"move_s mean={r['mt_mean']:.2f} max={r['mt_max']:.2f} "
                  f"| {time.time()-t0:.0f}s")
            except Exception as e:
                w(f"  {mode} {pl}p ERROR: {e}")
                traceback.print_exc()
        w("")

    # ---- PICK: best of the two defensible-RL modes, by avg win-rate ----
    w("## 2. PICK  (strongest of value_pool / value_rollout — both defensible RL)")
    avgs = {}
    for mode in ("value_pool", "value_rollout"):
        wr = [results[(mode, pl)]['winrate'] for pl in (2, 4, 6) if (mode, pl) in results]
        if wr:
            avgs[mode] = sum(wr) / len(wr)
            w(f"   {mode}: avg winrate over 2/4/6p = {avgs[mode]:.3f}")
    pick = None
    if avgs:
        pick = max(avgs, key=avgs.get)
        if "value_pool" in avgs and avgs.get("value_rollout", 0.0) - avgs["value_pool"] <= 0.03:
            pick = "value_pool"   # near-tie -> prefer the purer value-RL agent
        w(f"   -> PICK: {pick}")
    w("")

    # ---- DEPLOY ----
    w("## 3. DEPLOY")
    if pick:
        try:
            cfg = {"mode": pick, "n_sim": 200, "shaping": False,
                   "note": "value-network RL agent; heuristic proposes the move pool",
                   "written": time.strftime('%Y-%m-%d %H:%M:%S')}
            with open(DEPLOY, "w") as f:
                json.dump(cfg, f, indent=2)
            w(f"   wrote {DEPLOY}: mode={pick}")
            w("   tournament now fields the value-network RL agent.")
        except Exception as e:
            w(f"   DEPLOY ERROR: {e}")
            traceback.print_exc()
    else:
        w("   no pick — left hybrid in place.")
    w("")

    # ---- VERIFY ----
    w("## 4. VERIFY  (deployed mode, more games)")
    if pick:
        try:
            for pl, games in [(2, 40), (4, 30), (6, 30)]:
                t0 = time.time()
                r = play(choosers[pick], pl, games, seed=9100 + pl)
                w(f"   {pick} {pl}p: winrate={r['winrate']:.3f} ({r['wins']}/{games}) "
                  f"finished={r['finished']} maxmoves={r['maxm']} | "
                  f"move_s mean={r['mt_mean']:.2f} p95={r['mt_p95']:.2f} "
                  f"max={r['mt_max']:.2f} | {time.time()-t0:.0f}s")
        except Exception as e:
            w(f"   VERIFY ERROR: {e}")
            traceback.print_exc()
    w("")
    w("## 5. NOTES")
    w("   heuristic_rollout above is the non-RL reference ceiling (heuristic + "
      "rollouts, net not meaningfully used).")
    w("   Deployed agent is a value-based RL agent: the heuristic proposes "
      "candidate moves, the RL-trained value network selects.")
    w("   Revert: delete runs/mcts_deploy.json (-> heuristic_rollout hybrid).")
    w(f"# DONE  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("MODE SWEEP DONE", flush=True)


if __name__ == '__main__':
    main()
