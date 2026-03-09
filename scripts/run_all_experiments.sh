#!/usr/bin/env bash
set -euo pipefail

# Run all downstream forecasting experiments: all datasets, single-var and 3var,
# for both TSTR and downstream_task. Then writes COMBINED_RESULTS.
#
# Usage:
#   bash scripts/run_all_experiments.sh
#   bash scripts/run_all_experiments.sh --method lgta
#
# Optional: pass extra args (e.g. --method lgta) and they are forwarded to
# each experiment invocation.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ENV="lgta"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TS="$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/assets/results/downstream_forecasting}"
LOG_DIR="$RESULTS_DIR/logs"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

conda_env_exists() {
  conda run -n "$1" true 2>/dev/null
}

ensure_lgta_env() {
  if conda_env_exists "$MAIN_ENV"; then
    echo "[run_all_experiments] Using existing env: $MAIN_ENV"
    return
  fi
  echo "[run_all_experiments] Creating env: $MAIN_ENV"
  conda create -n "$MAIN_ENV" "python=$PYTHON_VERSION" -y
  conda run -n "$MAIN_ENV" pip install -r "$ROOT_DIR/requirements.txt"
}

ensure_lgta_env
mkdir -p "$LOG_DIR"

run_phase() {
  local phase_name="$1"
  shift
  echo ""
  echo "[run_all_experiments] ========== $phase_name =========="
  conda run --no-capture-output -n "$MAIN_ENV" python -u -m lgta.experiments.downstream_forecasting \
    --output-dir "$RESULTS_DIR" \
    --all-datasets \
    "$@" \
    "${EXTRA_ARGS[@]}"
}

EXTRA_ARGS=("$@")

echo "[run_all_experiments] Root: $ROOT_DIR | Results: $RESULTS_DIR | Log: $LOG_DIR/run_all_experiments_$TS.log"
echo "[run_all_experiments] Extra args: ${EXTRA_ARGS[*]:-(none)}"

{
  run_phase "Single var — TSTR" --eval-mode TSTR
  run_phase "Single var — downstream_task" --eval-mode downstream_task
  run_phase "3var — TSTR" --eval-mode TSTR --variant-transformations jitter scaling magnitude_warp
  run_phase "3var — downstream_task" --eval-mode downstream_task --variant-transformations jitter scaling magnitude_warp

  echo ""
  echo "[run_all_experiments] ========== Combine summary =========="
  conda run --no-capture-output -n "$MAIN_ENV" python -u -m lgta.experiments.downstream_forecasting \
    --output-dir "$RESULTS_DIR" \
    --combine-only
} 2>&1 | tee "$LOG_DIR/run_all_experiments_$TS.log"

echo "[run_all_experiments] Done. Results: $RESULTS_DIR (COMBINED_RESULTS.md / COMBINED_RESULTS.csv)"
