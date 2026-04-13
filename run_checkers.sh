#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/multi system single machine minimal/run_game.py"

if [[ ! -f "$LAUNCHER" ]]; then
  echo "Launcher not found: $LAUNCHER" >&2
  exit 1
fi

python3 "$LAUNCHER" "$@"
