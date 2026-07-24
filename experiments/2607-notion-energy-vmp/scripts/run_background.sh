#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CONFIG_PATH RUN_DIR [extra CLI args...]" >&2
  exit 2
fi

CONFIG_PATH="$1"
RUN_DIR="$2"
shift 2

mkdir -p "$RUN_DIR"
if [[ -f "$RUN_DIR/run.pid" ]] && kill -0 "$(cat "$RUN_DIR/run.pid")" 2>/dev/null; then
  echo "A run is already active with PID $(cat "$RUN_DIR/run.pid")" >&2
  exit 1
fi

nohup stdbuf -oL -eL .venv/bin/python -u scripts/run_notion_energy_experiment.py \
  --config "$CONFIG_PATH" \
  --run-dir "$RUN_DIR" \
  "$@" \
  >"$RUN_DIR/console.log" 2>&1 &

PID=$!
echo "$PID" >"$RUN_DIR/run.pid"
echo "Started PID $PID"
echo "Console: tail -F $RUN_DIR/console.log"
echo "Solver:  tail -F $RUN_DIR/solver.log"
