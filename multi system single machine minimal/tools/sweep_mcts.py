#!/usr/bin/env python3
"""Sweep the real MCTS agent: win-rate vs heuristic + per-move timing.

Tests the two *correct* net/MCTS pairings:
  * BC net  + plain MCTS    — raw +-1-trained value head, no shaping.
  * VRL net + shaping MCTS  — value-RL net (value head learned the shaped
    return E[T+Phi]-Phi(s)); shaping MCTS adds Phi(leaf) back so the leaf
    value is unbiased. Running the VRL net in plain MCTS biases the value.

Finds the strongest sim count that still fits the 2 s/turn budget. One
result line per cell is appended to --out as it completes.

    PYTHONPATH=. python tools/sweep_mcts.py --out runs/mcts_sweep2.txt
"""
import argparse
import random
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
from az.eval import _compute_player_score


def play_cell(nn_eval, players, sims, cpuct, shaping, games, seed):
    """Play `games` MCTS-agent-vs-heuristic games; return win-rate + timing."""
    if shaping:
        cfg = MCTSConfig(n_sim=sims, c_puct=cpuct, shaping_enabled=True,
                         shaping_scale=0.15, shaping_goal_weight=0.5,
                         dirichlet_alpha=0.0, dirichlet_eps=0.0,
                         batch_leaves=32, virtual_loss=1.0)
    else:
        cfg = MCTSConfig(n_sim=sims, c_puct=cpuct,
                         dirichlet_alpha=0.0, dirichlet_eps=0.0,
                         batch_leaves=32, virtual_loss=1.0)
    rng = random.Random(seed)
    cap = 300 if players == 2 else 200 * players
    wins = finished = maxmoves = 0
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
            maxmoves += 1
        cs = _compute_player_score(sim, cand, sim.move_count_by_colour[cand])
        opp = max(_compute_player_score(sim, c, sim.move_count_by_colour[c])['final_score']
                  for c in sim.colours if c != cand)
        margins.append(cs['final_score'] - opp)
    a = np.array(mt)
    return dict(winrate=wins / games, wins=wins, finished=finished, maxmoves=maxmoves,
                margin=float(np.mean(margins)), mt_mean=float(a.mean()),
                mt_p95=float(np.percentile(a, 95)), mt_max=float(a.max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vrl', default='runs/rl_value_mp/snapshots/snap_step160.pt')
    p.add_argument('--bc', default='runs/final_3day_20260511_105938/best.pt')
    p.add_argument('--out', default='runs/mcts_sweep2.txt')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()

    dev = torch.device(a.device)
    nets = {}

    def get(key):
        if key not in nets:
            path = a.vrl if key == 'VRL' else a.bc
            m = load_model(path, dev)
            m.eval()
            nets[key] = make_nn_eval(m, dev)
        return nets[key]

    # (net, players, sims, c_puct, shaping, games)
    plan = [
        ('BC',  2, 200,  1.5, False, 30), ('BC',  2, 400,  1.5, False, 30),
        ('BC',  2, 800,  1.5, False, 30), ('BC',  2, 1200, 1.5, False, 24),
        ('VRL', 2, 200,  1.5, True,  30), ('VRL', 2, 400,  1.5, True,  30),
        ('VRL', 2, 800,  1.5, True,  30), ('VRL', 2, 1200, 1.5, True,  24),
        ('BC',  6, 200,  1.5, False, 16), ('VRL', 6, 200,  1.5, True,  16),
        ('BC',  6, 600,  1.5, False, 16), ('VRL', 6, 600,  1.5, True,  16),
        ('BC',  4, 400,  1.5, False, 20), ('VRL', 4, 400,  1.5, True,  20),
    ]
    out = open(a.out, 'a', buffering=1)
    hdr = f"# MCTS sweep2 started {time.strftime('%Y-%m-%d %H:%M:%S')}"
    print(hdr, flush=True)
    out.write(hdr + '\n')
    for i, (netk, pl, sims, cpuct, shap, games) in enumerate(plan):
        t0 = time.time()
        tag = f"{netk}{'+shap' if shap else '+plain'}"
        try:
            r = play_cell(get(netk), pl, sims, cpuct, shap, games, seed=2000 + i)
            line = (f"[{i+1:2d}/{len(plan)}] {tag:9} {pl}p sims={sims:>4} cpuct={cpuct} "
                    f"games={games} | winrate={r['winrate']:.3f} ({r['wins']}/{games}) "
                    f"finished={r['finished']} maxmoves={r['maxmoves']} margin={r['margin']:.0f} "
                    f"| move_s mean={r['mt_mean']:.2f} p95={r['mt_p95']:.2f} max={r['mt_max']:.2f} "
                    f"| {time.time()-t0:.0f}s")
        except Exception as e:
            line = f"[{i+1:2d}/{len(plan)}] {tag} {pl}p sims={sims} ERROR: {e}"
            traceback.print_exc()
        print(line, flush=True)
        out.write(line + '\n')
    out.write(f"# done {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("SWEEP DONE", flush=True)


if __name__ == '__main__':
    main()
