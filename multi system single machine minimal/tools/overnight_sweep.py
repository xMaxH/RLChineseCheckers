#!/usr/bin/env python3
"""Run an unattended training sweep and evaluation bundle.

The sweep is intentionally sequential: this machine has one useful GPU, and
parallel training jobs would only turn memory pressure into noise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _write_jsonl(path: Path, obj: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def _run(cmd: List[str], log_path: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"# started {_ts()}\n")
        log.write("# " + " ".join(cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        rc = proc.wait()
        log.write(f"\n# finished {_ts()} rc={rc}\n")
        return rc


def _summarize_health(run_dir: Path) -> Dict:
    health = run_dir / "health.jsonl"
    rows = []
    if not health.exists():
        return {"health_missing": True}
    for line in health.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "chunk" in obj and "policy_loss" in obj:
            rows.append(obj)
    if not rows:
        return {"health_chunks": 0}
    out = {
        "health_chunks": len(rows),
        "last_chunk": rows[-1]["chunk"],
        "last_step": rows[-1].get("train_step_total"),
        "last_policy_loss": rows[-1].get("policy_loss"),
        "last_value_loss": rows[-1].get("value_loss"),
        "last_entropy": rows[-1].get("policy_entropy"),
        "last_wins": rows[-1].get("chunk_wins"),
        "last_max_moves": rows[-1].get("chunk_max_moves"),
        "last_replay_size": rows[-1].get("replay_size"),
    }
    for span in (10, 50):
        part = rows[-span:]
        if part:
            out[f"last{span}_policy_loss_mean"] = statistics.mean(r["policy_loss"] for r in part)
            out[f"last{span}_value_loss_mean"] = statistics.mean(r["value_loss"] for r in part)
            out[f"last{span}_wins_mean"] = statistics.mean(r["chunk_wins"] for r in part)
            out[f"last{span}_max_moves_mean"] = statistics.mean(r["chunk_max_moves"] for r in part)
    return out


_TABLE_RE = re.compile(
    r"^(greedy|mcts|heuristic)\s+"
    r"([0-9.]+)%\s+"
    r"([-0-9.]+)\s+"
    r"([0-9.]+)%\s+"
    r"(?:([0-9.]+)%|n/a)\s+"
    r"(?:([0-9.]+)%|n/a)\s*$"
)


def _parse_check_bc(log_path: Path, prefix: str) -> Dict:
    out: Dict[str, float] = {}
    if not log_path.exists():
        return out
    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _TABLE_RE.match(raw.strip())
        if not m:
            continue
        mode, win, margin, max_mov, exact, pool = m.groups()
        key = f"{prefix}_{mode}"
        out[f"{key}_win_rate"] = float(win) / 100.0
        out[f"{key}_margin"] = float(margin)
        out[f"{key}_max_moves"] = float(max_mov) / 100.0
        if exact is not None:
            out[f"{key}_exact"] = float(exact) / 100.0
        if pool is not None:
            out[f"{key}_pool"] = float(pool) / 100.0
    return out


def _experiments(seed_ckpt: str) -> List[Dict]:
    common = [
        "--seed-ckpt", seed_ckpt,
        "--max-chunks", "1000",
        "--num-workers", "6",
        "--bootstrap-chunks", "9999",
        "--heuristic-rollout-targets",
        "--heuristic-rollouts-per-move", "1",
        "--heuristic-rollout-pool-cap", "8",
        "--heuristic-rollout-score-temperature", "250",
        "--games-per-chunk", "32",
        "--min-samples-to-train", "4000",
        "--batch-size", "4096",
        "--sample-per-step", "8192",
        "--min-train-steps", "75",
        "--lr", "5e-5",
        "--entropy-bonus", "0.0",
        "--eval-every", "9999",
    ]
    return [
        {
            "name": "rt_dagger_t025_cap8",
            "hours": 3.0,
            "args": common + ["--dagger-in-bootstrap", "--dagger-policy-temperature", "0.25"],
        },
        {
            "name": "rt_dagger_t0_cap8",
            "hours": 3.0,
            "args": common + ["--dagger-in-bootstrap", "--dagger-policy-temperature", "0.0"],
        },
        {
            "name": "rt_dagger_t05_cap8",
            "hours": 3.0,
            "args": common + ["--dagger-in-bootstrap", "--dagger-policy-temperature", "0.5"],
        },
        {
            "name": "rt_bc_cap8",
            "hours": 2.0,
            "args": common,
        },
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="Sweep output root under runs/")
    ap.add_argument("--seed-ckpt", default="runs/dagger_from_bc_2h/final.pt")
    ap.add_argument("--skip-mcts", action="store_true")
    args = ap.parse_args()

    sweep_root = Path(args.root) if args.root else ROOT / "runs" / f"overnight_sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    if not sweep_root.is_absolute():
        sweep_root = ROOT / sweep_root
    sweep_root.mkdir(parents=True, exist_ok=True)
    summary_path = sweep_root / "summary.jsonl"
    commands_path = sweep_root / "commands.sh"

    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONUNBUFFERED", "1")

    seed_ckpt = args.seed_ckpt
    if not (ROOT / seed_ckpt).exists():
        raise SystemExit(f"missing seed checkpoint: {seed_ckpt}")

    experiments = _experiments(seed_ckpt)
    with commands_path.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        for exp in experiments:
            run_dir = sweep_root / exp["name"]
            train_cmd = [
                sys.executable, "train_run.py",
                "--out", str(run_dir.relative_to(ROOT)),
                "--hours", str(exp["hours"]),
                "--name", exp["name"],
            ] + exp["args"]
            f.write(" ".join(train_cmd) + "\n")

    _write_jsonl(summary_path, {
        "event": "sweep_start",
        "time": _ts(),
        "root": str(sweep_root.relative_to(ROOT)),
        "seed_ckpt": seed_ckpt,
        "experiments": [{"name": e["name"], "hours": e["hours"], "args": e["args"]} for e in experiments],
    })

    for exp in experiments:
        run_dir = sweep_root / exp["name"]
        train_cmd = [
            sys.executable, "train_run.py",
            "--out", str(run_dir.relative_to(ROOT)),
            "--hours", str(exp["hours"]),
            "--name", exp["name"],
        ] + exp["args"]
        _write_jsonl(summary_path, {"event": "train_start", "time": _ts(), "name": exp["name"], "cmd": train_cmd})
        rc = _run(train_cmd, run_dir / "train.console.log", env)
        health_summary = _summarize_health(run_dir)
        _write_jsonl(summary_path, {
            "event": "train_end",
            "time": _ts(),
            "name": exp["name"],
            "rc": rc,
            **health_summary,
        })
        final_ckpt = run_dir / "final.pt"
        if rc != 0 or not final_ckpt.exists():
            _write_jsonl(summary_path, {
                "event": "eval_skipped",
                "time": _ts(),
                "name": exp["name"],
                "reason": "train_failed_or_missing_final",
            })
            continue

        greedy_cmd = [
            sys.executable, "tools/check_bc.py",
            "--ckpt", str(final_ckpt.relative_to(ROOT)),
            "--device", "cuda",
            "--n-games", "60",
            "--skip-mcts",
        ]
        greedy_log = run_dir / "eval_greedy_60.log"
        _write_jsonl(summary_path, {"event": "eval_greedy_start", "time": _ts(), "name": exp["name"], "cmd": greedy_cmd})
        greedy_rc = _run(greedy_cmd, greedy_log, env)
        greedy_summary = _parse_check_bc(greedy_log, "greedy60")
        _write_jsonl(summary_path, {
            "event": "eval_greedy_end",
            "time": _ts(),
            "name": exp["name"],
            "rc": greedy_rc,
            **greedy_summary,
        })

        teacher_cmd = [
            sys.executable, "tools/check_bc.py",
            "--ckpt", str(final_ckpt.relative_to(ROOT)),
            "--device", "cuda",
            "--n-games", "20",
            "--skip-mcts",
            "--match-teacher", "rollout",
            "--teacher-rollouts-per-move", "1",
            "--teacher-rollout-pool-cap", "8",
            "--teacher-score-temperature", "250",
        ]
        teacher_log = run_dir / "eval_rollout_teacher_20.log"
        _write_jsonl(summary_path, {"event": "eval_teacher_start", "time": _ts(), "name": exp["name"], "cmd": teacher_cmd})
        teacher_rc = _run(teacher_cmd, teacher_log, env)
        teacher_summary = _parse_check_bc(teacher_log, "teacher20")
        _write_jsonl(summary_path, {
            "event": "eval_teacher_end",
            "time": _ts(),
            "name": exp["name"],
            "rc": teacher_rc,
            **teacher_summary,
        })

        if not args.skip_mcts:
            mcts_cmd = [
                sys.executable, "tools/check_bc.py",
                "--ckpt", str(final_ckpt.relative_to(ROOT)),
                "--device", "cuda",
                "--n-games", "16",
                "--mcts-sims", "50",
            ]
            mcts_log = run_dir / "eval_mcts50_16.log"
            _write_jsonl(summary_path, {"event": "eval_mcts_start", "time": _ts(), "name": exp["name"], "cmd": mcts_cmd})
            mcts_rc = _run(mcts_cmd, mcts_log, env)
            mcts_summary = _parse_check_bc(mcts_log, "mcts50")
            _write_jsonl(summary_path, {
                "event": "eval_mcts_end",
                "time": _ts(),
                "name": exp["name"],
                "rc": mcts_rc,
                **mcts_summary,
            })

    _write_jsonl(summary_path, {"event": "sweep_end", "time": _ts()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
