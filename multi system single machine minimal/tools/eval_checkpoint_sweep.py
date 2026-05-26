#!/usr/bin/env python3
"""Evaluate a checkpoint across 2p-6p and optionally compare to a baseline."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from az.config import MCTSConfig
from tools.check_bc import load_net, run_eval


def parse_games(text: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("--games must look like 2:40,3:30,4:30,5:20,6:20")
        k, v = part.split(":", 1)
        out[int(k)] = int(v)
    for n in out:
        if n < 2 or n > 6:
            raise ValueError("player counts must be between 2 and 6")
    return out


def checkpoint_step(path: str) -> int | None:
    m = re.search(r"snap_step(\d+)\.pt$", path)
    return int(m.group(1)) if m else None


def eval_one(
    label: str,
    ckpt: str,
    device: str,
    games_by_count: Dict[int, int],
    seed: int,
    max_moves: int,
    top_k: int,
    rollouts_per_move: int,
) -> Dict:
    print(f"\n[eval] loading {label}: {ckpt}")
    net = load_net(ckpt, device)
    mcts_cfg = MCTSConfig(n_sim=1)
    rows = {}
    for n_players, n_games in sorted(games_by_count.items()):
        wr, margin, exact, pool, max_pct = run_eval(
            "greedy",
            net,
            device,
            mcts_cfg,
            n_games,
            seed + n_players * 1000,
            num_players=n_players,
            max_moves=max_moves,
            track_match=True,
            teacher_cfg=None,
            greedy_scope="heuristic-rollout",
            rollout_top_k=top_k,
            rollouts_per_move=rollouts_per_move,
            verbose=False,
        )
        rows[str(n_players)] = {
            "games": n_games,
            "win_rate": wr,
            "margin": margin,
            "exact": exact,
            "pool": pool,
            "max_moves_pct": max_pct,
        }
        print(
            f"[eval] {label:<10} {n_players}p "
            f"win={wr:5.1%} margin={margin:7.0f} max={max_pct:5.1%} pool={pool:5.1%}"
        )
    if str(device).startswith("cuda"):
        del net
        torch.cuda.empty_cache()
    return {"label": label, "ckpt": ckpt, "step": checkpoint_step(ckpt), "rows": rows}


def weighted_score(result: Dict) -> float:
    # Simple scalar for monitoring trend. Positive means both winning and margin
    # improved; 6p/5p matter most because they were the weak points.
    weights = {"2": 0.08, "3": 0.12, "4": 0.18, "5": 0.27, "6": 0.35}
    total = 0.0
    for k, row in result["rows"].items():
        w = weights.get(k, 0.0)
        total += w * (row["win_rate"] * 100.0 + row["margin"] / 30.0)
    return total


def print_compare(candidate: Dict, baseline: Dict | None) -> None:
    print("\n" + "=" * 82)
    print(f"{'COUNT':<6} {'CAND WIN':>9} {'BASE WIN':>9} {'DELTA':>8} {'CAND M':>9} {'BASE M':>9} {'DELTA M':>9}")
    print("-" * 82)
    for k in sorted(candidate["rows"], key=int):
        c = candidate["rows"][k]
        if baseline is not None and k in baseline["rows"]:
            b = baseline["rows"][k]
            print(
                f"{k+'p':<6} {c['win_rate']:>8.1%} {b['win_rate']:>8.1%} "
                f"{(c['win_rate']-b['win_rate']):>+7.1%} "
                f"{c['margin']:>9.0f} {b['margin']:>9.0f} {(c['margin']-b['margin']):>+9.0f}"
            )
        else:
            print(f"{k+'p':<6} {c['win_rate']:>8.1%} {'n/a':>9} {'n/a':>8} {c['margin']:>9.0f}")
    print("=" * 82)
    print(f"candidate_score={weighted_score(candidate):.2f}")
    if baseline is not None:
        print(f"baseline_score={weighted_score(baseline):.2f} delta={weighted_score(candidate)-weighted_score(baseline):+.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--baseline-ckpt", default=None)
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    ap.add_argument("--summary-jsonl", default=None, help="Append compact monitor row here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--games", default="2:30,3:20,4:20,5:12,6:12")
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--max-moves", type=int, default=500)
    ap.add_argument("--policy-rollout-top-k", type=int, default=3)
    ap.add_argument("--policy-rollouts-per-move", type=int, default=2)
    args = ap.parse_args()

    games_by_count = parse_games(args.games)
    candidate = eval_one(
        "candidate",
        args.ckpt,
        args.device,
        games_by_count,
        args.seed,
        args.max_moves,
        args.policy_rollout_top_k,
        args.policy_rollouts_per_move,
    )
    baseline = None
    if args.baseline_ckpt:
        baseline = eval_one(
            "baseline",
            args.baseline_ckpt,
            args.device,
            games_by_count,
            args.seed,
            args.max_moves,
            args.policy_rollout_top_k,
            args.policy_rollouts_per_move,
        )
    print_compare(candidate, baseline)

    payload = {
        "candidate": candidate,
        "baseline": baseline,
        "candidate_score": weighted_score(candidate),
        "baseline_score": weighted_score(baseline) if baseline else None,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.summary_jsonl:
        compact = {
            "ckpt": args.ckpt,
            "step": candidate["step"],
            "candidate_score": payload["candidate_score"],
            "baseline_score": payload["baseline_score"],
            "delta_score": (
                payload["candidate_score"] - payload["baseline_score"]
                if payload["baseline_score"] is not None else None
            ),
            "rows": candidate["rows"],
        }
        p = Path(args.summary_jsonl)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(compact) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
