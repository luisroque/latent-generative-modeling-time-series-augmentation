#!/usr/bin/env bash
set -euo pipefail

# Run downstream forecasting experiment. Creates conda env lgta if it does
# not exist (no need to activate any env before running).
#
# Usage:
#   bash scripts/run_all_benchmarks.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ENV="lgta"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TS="$(date +%Y%m%d-%H%M%S)"

RESULTS_DIR="$ROOT_DIR/assets/results/downstream_forecasting/$TS"
LOG_DIR="$RESULTS_DIR/logs"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

conda_env_exists() {
  conda run -n "$1" true 2>/dev/null
}

ensure_lgta_env() {
  if conda_env_exists "$MAIN_ENV"; then
    echo "[run_all_benchmarks] Using existing env: $MAIN_ENV"
    return
  fi
  echo "[run_all_benchmarks] Creating env: $MAIN_ENV"
  conda create -n "$MAIN_ENV" "python=$PYTHON_VERSION" -y
  conda run -n "$MAIN_ENV" pip install -r "$ROOT_DIR/requirements.txt"
}

ensure_lgta_env

echo "[run_all_benchmarks] Root dir: $ROOT_DIR"
echo "[run_all_benchmarks] Timestamp: $TS"

mkdir -p "$LOG_DIR"

echo "[run_all_benchmarks] Running downstream_forecasting in env: $MAIN_ENV"
conda run --no-capture-output -n "$MAIN_ENV" python -u -m lgta.experiments.downstream_forecasting \
  --output-dir "$RESULTS_DIR" 2>&1 | tee "$LOG_DIR/run_$TS.log"

echo "[run_all_benchmarks] Output files:"
ls -la "$RESULTS_DIR" | sed 's/^/[run_all_benchmarks] /'

echo "[run_all_benchmarks] Done."
