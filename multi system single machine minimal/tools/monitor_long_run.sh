#!/usr/bin/env bash
set -euo pipefail

RUN="${1:?usage: tools/monitor_long_run.sh RUN_DIR [BASELINE_CKPT] [INTERVAL_MINUTES]}"
BASELINE="${2:-runs/best.pt}"
INTERVAL_MIN="${3:-360}"
DEVICE="${DEVICE:-cuda}"
GAMES="${GAMES:-2:30,3:20,4:20,5:12,6:12}"

EVAL_DIR="$RUN/live_eval"
mkdir -p "$EVAL_DIR"

echo "[monitor] run=$RUN"
echo "[monitor] baseline=$BASELINE"
echo "[monitor] interval=${INTERVAL_MIN}m games=$GAMES device=$DEVICE"
echo "[monitor] summary=$EVAL_DIR/summary.jsonl"

last_step=""
while true; do
  latest="$(find "$RUN/snapshots" -maxdepth 1 -name 'snap_step*.pt' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "[monitor] no snapshots yet; sleeping ${INTERVAL_MIN}m"
    sleep "$((INTERVAL_MIN * 60))"
    continue
  fi

  step="${latest#snap_step}"
  step="${step%.pt}"
  ckpt="$RUN/snapshots/$latest"
  if [[ "$step" == "$last_step" ]]; then
    echo "[monitor] latest step $step already evaluated; sleeping ${INTERVAL_MIN}m"
    sleep "$((INTERVAL_MIN * 60))"
    continue
  fi

  echo "[monitor] evaluating step=$step ckpt=$ckpt at $(date -Is)"
  python3 tools/eval_checkpoint_sweep.py \
    --ckpt "$ckpt" \
    --baseline-ckpt "$BASELINE" \
    --device "$DEVICE" \
    --games "$GAMES" \
    --max-moves 500 \
    --policy-rollout-top-k 3 \
    --policy-rollouts-per-move 2 \
    --out "$EVAL_DIR/eval_step${step}.json" \
    --summary-jsonl "$EVAL_DIR/summary.jsonl" \
    2>&1 | tee "$EVAL_DIR/eval_step${step}.log"

  last_step="$step"
  echo "[monitor] done step=$step; sleeping ${INTERVAL_MIN}m"
  sleep "$((INTERVAL_MIN * 60))"
done
