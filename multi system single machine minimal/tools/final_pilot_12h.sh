#!/usr/bin/env bash
set -euo pipefail

RUN="${1:-runs/final_reranker_pilot_12h_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN"

echo "[driver] run=$RUN"
echo "[driver] started=$(date -Is)"

python3 train_run.py \
  --out "$RUN" \
  --name final_reranker_pilot_12h \
  --hours 12 \
  --max-chunks 5000 \
  --seed 20260510 \
  --seed-ckpt runs/best.pt \
  --device cuda \
  --num-workers 6 \
  --mcts-sims 64 \
  --player-counts 2,3,4,5,6 \
  --player-weights 0.10,0.15,0.20,0.25,0.30 \
  --games-per-chunk 20 \
  --bootstrap-chunks 999999 \
  --heuristic-rollout-targets \
  --heuristic-rollouts-per-move 2 \
  --heuristic-rollout-pool-cap 10 \
  --heuristic-rollout-score-temperature 0 \
  --max-moves-2p 500 \
  --max-moves-multi 500 \
  --replay-capacity 1000000 \
  --min-samples-to-train 8000 \
  --batch-size 4096 \
  --sample-per-step 8192 \
  --min-train-steps 40 \
  --lr 2e-5 \
  --entropy-bonus 0.0 \
  --eval-every 999999

CKPT="$RUN/final.pt"
VERIFY="$RUN/final_verify"
mkdir -p "$VERIFY"

echo "[driver] training_finished=$(date -Is)"
echo "[driver] verifying checkpoint=$CKPT"

python3 tools/check_bc.py \
  --ckpt "$CKPT" \
  --device cuda \
  --n-games 100 \
  --num-players 2 \
  --skip-mcts \
  --greedy-scope heuristic-rollout \
  --policy-rollout-top-k 3 \
  --policy-rollouts-per-move 2 \
  --quiet | tee "$VERIFY/verify_2p.log"

python3 tools/check_bc.py \
  --ckpt "$CKPT" \
  --device cuda \
  --n-games 60 \
  --num-players 3 \
  --max-moves 500 \
  --skip-mcts \
  --greedy-scope heuristic-rollout \
  --policy-rollout-top-k 3 \
  --policy-rollouts-per-move 2 \
  --quiet | tee "$VERIFY/verify_3p.log"

python3 tools/check_bc.py \
  --ckpt "$CKPT" \
  --device cuda \
  --n-games 60 \
  --num-players 4 \
  --max-moves 500 \
  --skip-mcts \
  --greedy-scope heuristic-rollout \
  --policy-rollout-top-k 3 \
  --policy-rollouts-per-move 2 \
  --quiet | tee "$VERIFY/verify_4p.log"

python3 tools/check_bc.py \
  --ckpt "$CKPT" \
  --device cuda \
  --n-games 40 \
  --num-players 5 \
  --max-moves 500 \
  --skip-mcts \
  --greedy-scope heuristic-rollout \
  --policy-rollout-top-k 3 \
  --policy-rollouts-per-move 2 \
  --quiet | tee "$VERIFY/verify_5p.log"

python3 tools/check_bc.py \
  --ckpt "$CKPT" \
  --device cuda \
  --n-games 40 \
  --num-players 6 \
  --max-moves 500 \
  --skip-mcts \
  --greedy-scope heuristic-rollout \
  --policy-rollout-top-k 3 \
  --policy-rollouts-per-move 2 \
  --quiet | tee "$VERIFY/verify_6p.log"

python3 - "$VERIFY" <<'PY'
import re
import sys
from pathlib import Path

verify = Path(sys.argv[1])
pat = re.compile(
    r"^(greedy|heuristic)\s+([0-9.]+)%\s+([-0-9.]+)\s+([0-9.]+)%"
    r"\s+(?:([0-9.]+)%|n/a)\s+(?:([0-9.]+)%|n/a)",
    re.M,
)
print("\n[driver] summary")
for log in sorted(verify.glob("verify_*.log")):
    text = log.read_text(errors="replace")
    rows = {m.group(1): m.groups()[1:] for m in pat.finditer(text)}
    if "greedy" not in rows or "heuristic" not in rows:
        print(f"{log.name}: incomplete")
        continue
    gw, gm, gmax, gexact, gpool = rows["greedy"]
    hw, hm, hmax, _, _ = rows["heuristic"]
    print(
        f"{log.name}: greedy win={gw}% margin={gm} max={gmax}% "
        f"vs heuristic win={hw}% margin={hm} max={hmax}%"
    )
PY

echo "[driver] finished=$(date -Is)"
