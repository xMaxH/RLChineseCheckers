"""Run the full curriculum 2p -> 6p, warm-starting each stage from the previous best."""

import argparse
import os
import shutil
import torch

from .config import (
    verify_stage, overnight_stage_2p,
    stage_2_3p, stage_2_4p, stage_2_5p, stage_2_6p,
)
from .train import train_one_stage


STAGES_BY_NAME = {
    "verify": verify_stage,
    "2p": overnight_stage_2p,
    "2_3p": stage_2_3p,
    "2_4p": stage_2_4p,
    "2_5p": stage_2_5p,
    "2_6p": stage_2_6p,
}

DEFAULT_FULL = ["2p", "2_3p", "2_4p", "2_5p", "2_6p"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", type=str, default=",".join(DEFAULT_FULL),
                   help="Comma-separated stage names to run sequentially")
    p.add_argument("--root", type=str, default="runs/full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed-ckpt", type=str, default=None,
                   help="Optional warm-start checkpoint for the first stage")
    p.add_argument("--bootstrap-chunks", type=int, default=2)
    p.add_argument("--max-chunks", type=int, default=200)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.root, exist_ok=True)

    prev_best = args.seed_ckpt
    for s_name in args.stages.split(","):
        s_name = s_name.strip()
        if s_name not in STAGES_BY_NAME:
            raise SystemExit(f"unknown stage '{s_name}'. valid: {list(STAGES_BY_NAME)}")
        stage = STAGES_BY_NAME[s_name]()
        out_dir = os.path.join(args.root, s_name)
        print(f"\n===== STARTING STAGE {s_name} → {out_dir}, warm_from={prev_best}")
        best_path = train_one_stage(
            stage, out_dir, seed_ckpt=prev_best, device=device,
            rng_seed=args.seed, bootstrap_chunks=args.bootstrap_chunks,
            max_chunks=args.max_chunks,
        )
        prev_best = best_path
        # Promote to runs/best.pt at the project root for tournament inference
        global_best = os.path.join("runs", "best.pt")
        os.makedirs("runs", exist_ok=True)
        shutil.copy2(best_path, global_best)
        print(f"[curriculum] published {best_path} -> {global_best}")

    print("\n===== CURRICULUM COMPLETE")


if __name__ == "__main__":
    main()
