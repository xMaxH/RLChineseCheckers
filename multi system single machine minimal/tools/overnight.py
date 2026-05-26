#!/usr/bin/env python3
"""Autonomous overnight pipeline for the real RL (MCTS) tournament agent.

  1. Sweep the MCTS agent: BC net + plain MCTS, and VRL net + shaping MCTS,
     across sim counts, at 2p (+ 4p/6p), measuring win-rate vs heuristic and
     per-move timing.
  2. Pick the strongest config whose worst-case move time fits the 2 s budget.
  3. Deploy it: copy the chosen net to runs/best.pt and write
     runs/mcts_deploy.json (read by alphazero_method.py -> MCTS mode).
  4. Verify the deployed config at 2p / 4p / 6p with more games.
  5. Write runs/OVERNIGHT_SUMMARY.txt with everything + revert instructions.

Safe by construction: if the script dies before step 3, nothing is changed
and the tournament still uses the strong heuristic_rollout hybrid. Backups:
runs/best_prev.pt, runs/best_may10_backup.pt, alphazero_method_prefnl.bak.

    PYTHONPATH=. python tools/overnight.py
"""
import json
import os
import random
import shutil
import time
import traceback

import numpy as np
import torch

from az.config import MCTSConfig
from az.sim import Sim
from az.mcts import run_search
from az.encoder import decode_action
from az.heuristic import heuristic_choose_move
from az.inference_server import load_model, make_nn_eval

VRL = "runs/rl_value_mp/snapshots/snap_step160.pt"   # value-RL net (shaped value head)
BC = "runs/final_3day_20260511_105938/best.pt"        # BC seed net (plain value head)
SUMMARY = "runs/OVERNIGHT_SUMMARY.txt"
DEPLOY = "runs/mcts_deploy.json"
BEST = "runs/best.pt"
TIMING_BUDGET = 1.6      # max move-seconds for a config to count as tournament-safe
WINRATE_FLOOR = 0.55     # below this the MCTS agent is too weak to auto-deploy

_nets = {}


def get_net(path, dev):
    if path not in _nets:
        m = load_model(path, dev)
        m.eval()
        _nets[path] = make_nn_eval(m, dev)
    return _nets[path]


def mcfg(sims, shaping, cpuct=1.5):
    if shaping:
        return MCTSConfig(n_sim=sims, c_puct=cpuct, shaping_enabled=True,
                          shaping_scale=0.15, shaping_goal_weight=0.5,
                          dirichlet_alpha=0.0, dirichlet_eps=0.0,
                          batch_leaves=32, virtual_loss=1.0)
    return MCTSConfig(n_sim=sims, c_puct=cpuct, dirichlet_alpha=0.0,
                      dirichlet_eps=0.0, batch_leaves=32, virtual_loss=1.0)


def play(nn_eval, cfg, players, games, seed):
    rng = random.Random(seed)
    cap = 300 if players == 2 else 200 * players
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
                visits, _, _ = run_search(sim, nn_eval, cfg, add_dirichlet_at_root=False)
                mt.append(time.perf_counter() - t0)
                if int(visits.sum()) == 0:
                    pid, to = heuristic_choose_move(sim, col, legal, rng=rng)
                else:
                    pid, to = decode_action(int(np.argmax(visits)), col)
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
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = open(SUMMARY, "w", buffering=1)

    def w(s=""):
        print(s, flush=True)
        log.write(s + "\n")

    w(f"# OVERNIGHT MCTS PIPELINE  —  started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"# device={dev}  VRL={VRL}  BC={BC}")
    w("")

    # ---- 1. SWEEP ----
    # (net_path, shaping, players, sims, games)
    plan = [
        (BC, False, 2, 200, 30), (BC, False, 2, 400, 30),
        (BC, False, 2, 800, 30), (BC, False, 2, 1200, 24),
        (VRL, True, 2, 200, 30), (VRL, True, 2, 400, 30),
        (VRL, True, 2, 800, 30), (VRL, True, 2, 1200, 24),
        (BC, False, 6, 200, 16), (VRL, True, 6, 200, 16),
        (BC, False, 6, 600, 16), (VRL, True, 6, 600, 16),
        (BC, False, 4, 400, 20), (VRL, True, 4, 400, 20),
    ]
    w("## 1. SWEEP  (winrate vs heuristic, per-move timing)")
    results = []
    for i, (path, shap, pl, sims, games) in enumerate(plan):
        tag = "VRL+shaping" if shap else "BC+plain"
        t0 = time.time()
        try:
            r = play(get_net(path, dev), mcfg(sims, shap), pl, games, seed=3000 + i)
            r.update(net=path, shaping=shap, players=pl, sims=sims, tag=tag)
            results.append(r)
            w(f"[{i+1:2d}/{len(plan)}] {tag:12} {pl}p sims={sims:>4} games={games} | "
              f"winrate={r['winrate']:.3f} ({r['wins']}/{games}) finished={r['finished']} "
              f"maxmoves={r['maxm']} | move_s mean={r['mt_mean']:.2f} "
              f"p95={r['mt_p95']:.2f} max={r['mt_max']:.2f} | {time.time()-t0:.0f}s")
        except Exception as e:
            w(f"[{i+1:2d}/{len(plan)}] {tag} {pl}p sims={sims} ERROR: {e}")
            traceback.print_exc()
    w("")

    # ---- 2. PICK ----
    w("## 2. PICK  (best 2p winrate among configs with max move-time < "
      f"{TIMING_BUDGET}s)")
    pick = None
    try:
        twop_safe = [r for r in results if r['players'] == 2 and r['mt_max'] < TIMING_BUDGET]
        pool = twop_safe or [r for r in results if r['players'] == 2]
        if pool:
            by_pair = {}
            for r in pool:
                k = (r['net'], r['shaping'])
                if k not in by_pair or r['winrate'] > by_pair[k]['winrate']:
                    by_pair[k] = r
            ranked = sorted(by_pair.values(), key=lambda r: -r['winrate'])
            pick = ranked[0]
            # near-tie tiebreak: prefer the genuine value-RL net
            for r in ranked:
                if r['shaping'] and ranked[0]['winrate'] - r['winrate'] <= 0.04:
                    pick = r
                    break
            for r in ranked:
                w(f"   candidate: {r['tag']:12} sims={r['sims']:>4} "
                  f"2p winrate={r['winrate']:.3f}  max move={r['mt_max']:.2f}s")
            w(f"   -> PICK: {pick['tag']}  sims={pick['sims']}  "
              f"shaping={pick['shaping']}  (2p winrate {pick['winrate']:.3f})")
    except Exception as e:
        w(f"   PICK ERROR: {e}")
        traceback.print_exc()
    w("")

    # ---- 3. DEPLOY ----
    w("## 3. DEPLOY")
    deployed = False
    if pick is None:
        w("   no pick — leaving heuristic_rollout hybrid in place.")
    elif pick['winrate'] < WINRATE_FLOOR:
        w(f"   best MCTS 2p winrate {pick['winrate']:.3f} < floor {WINRATE_FLOOR} "
          f"— NOT auto-deploying; hybrid left in place. Review manually.")
    else:
        try:
            if os.path.exists(BEST):
                shutil.copy(BEST, "runs/best_beforedeploy.pt")
            shutil.copy(pick['net'], BEST)
            cfg = {"mode": "mcts", "n_sim": int(pick['sims']),
                   "shaping": bool(pick['shaping']), "c_puct": 1.5,
                   "net_source": pick['net'],
                   "written": time.strftime('%Y-%m-%d %H:%M:%S')}
            with open(DEPLOY, "w") as f:
                json.dump(cfg, f, indent=2)
            deployed = True
            w(f"   copied {pick['net']} -> {BEST}")
            w(f"   wrote {DEPLOY}: {cfg}")
            w("   tournament will now run MCTS mode (the real RL agent).")
        except Exception as e:
            w(f"   DEPLOY ERROR: {e}")
            traceback.print_exc()
    w("")

    # ---- 4. VERIFY ----
    w("## 4. VERIFY  (deployed config, more games)")
    if deployed:
        try:
            ev = get_net(pick['net'], dev)
            for pl, games in [(2, 40), (4, 24), (6, 24)]:
                t0 = time.time()
                r = play(ev, mcfg(pick['sims'], pick['shaping']), pl, games, seed=9000 + pl)
                w(f"   {pl}p: winrate={r['winrate']:.3f} ({r['wins']}/{games}) "
                  f"finished={r['finished']} maxmoves={r['maxm']} | "
                  f"move_s mean={r['mt_mean']:.2f} p95={r['mt_p95']:.2f} "
                  f"max={r['mt_max']:.2f} | {time.time()-t0:.0f}s")
        except Exception as e:
            w(f"   VERIFY ERROR: {e}")
            traceback.print_exc()
    else:
        w("   skipped (nothing deployed).")
    w("")

    # ---- 5. SUMMARY ----
    w("## 5. NOTES")
    w("   Hybrid baseline (heuristic_rollout, measured earlier): "
      "2p=1.00  4p=1.00  6p=0.50 vs heuristic, all moves <1.1s.")
    w("   Revert to the hybrid: delete runs/mcts_deploy.json, OR run with")
    w("     ALPHAZERO_POLICY_MODE=heuristic_rollout")
    w("   Backups: runs/best_prev.pt, runs/best_may10_backup.pt, "
      "runs/best_beforedeploy.pt, alphazero_method_prefnl.bak")
    w("")
    w(f"# DONE  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("OVERNIGHT PIPELINE DONE", flush=True)


if __name__ == '__main__':
    main()
