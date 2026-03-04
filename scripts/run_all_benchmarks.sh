#!/usr/bin/env bash
set -euo pipefail

# Orchestrate downstream forecasting runs in different environments and
# produce a combined report. Self-contained: creates conda envs lgta and
# lgta-tsdiff if they do not exist (no need to activate any env before running).
#
# - lgta: main env (requirements.txt).
# - lgta-tsdiff: same as lgta plus TSDiff (requirements.txt then requirements-tsdiff.txt with --no-deps).
#
# Usage:
#   bash scripts/run_all_benchmarks.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ENV="lgta"
TSDIFF_ENV="lgta-tsdiff"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TS="$(date +%Y%m%d-%H%M%S)"

RESULTS_ROOT="$ROOT_DIR/assets/results/downstream_forecasting/multi_env/$TS"
LOG_DIR="$RESULTS_ROOT/logs"
WITH_TSDIFF_DIR="$RESULTS_ROOT/with_tsdiff"
NO_TSDIFF_DIR="$RESULTS_ROOT/no_tsdiff"
FINAL_REPORT="$RESULTS_ROOT/FINAL_DOWNSTREAM_RESULTS.md"

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

tsdiff_env_ready() {
  conda run -n "$TSDIFF_ENV" python -c "import torch" 2>/dev/null
}

ensure_tsdiff_env() {
  if conda_env_exists "$TSDIFF_ENV"; then
    if tsdiff_env_ready; then
      echo "[run_all_benchmarks] Using existing env: $TSDIFF_ENV"
      return
    fi
    echo "[run_all_benchmarks] Env $TSDIFF_ENV exists but deps missing; installing..."
  else
    echo "[run_all_benchmarks] Creating env: $TSDIFF_ENV"
    conda create -n "$TSDIFF_ENV" "python=$PYTHON_VERSION" -y
  fi
  conda run -n "$TSDIFF_ENV" pip install -r "$ROOT_DIR/requirements.txt"
  conda run -n "$TSDIFF_ENV" pip install --no-deps -r "$ROOT_DIR/requirements-tsdiff.txt"
}

ensure_lgta_env
ensure_tsdiff_env

echo "[run_all_benchmarks] Root dir: $ROOT_DIR"
echo "[run_all_benchmarks] Timestamp: $TS"

mkdir -p "$LOG_DIR" "$WITH_TSDIFF_DIR" "$NO_TSDIFF_DIR"
echo "[run_all_benchmarks] Logs dir: $LOG_DIR"
echo "[run_all_benchmarks] Results dirs: $WITH_TSDIFF_DIR , $NO_TSDIFF_DIR"

echo "[run_all_benchmarks] Running downstream_forecasting (mode=tsdiff) in env: $TSDIFF_ENV"
conda run --no-capture-output -n "$TSDIFF_ENV" python -u -m lgta.experiments.downstream_forecasting \
  --mode tsdiff --output-dir "$WITH_TSDIFF_DIR" 2>&1 | tee "$LOG_DIR/tsdiff_env_$TS.log"

echo "[run_all_benchmarks] Running downstream_forecasting (mode=lgta) in env: $MAIN_ENV"
conda run --no-capture-output -n "$MAIN_ENV" python -u -m lgta.experiments.downstream_forecasting \
  --mode lgta --output-dir "$NO_TSDIFF_DIR" 2>&1 | tee "$LOG_DIR/main_env_$TS.log"

echo "[run_all_benchmarks] Merging results into: $FINAL_REPORT"
conda run --no-capture-output -n "$MAIN_ENV" python -u -m lgta.experiments.merge_downstream_results \
  --results-dir "$RESULTS_ROOT" 2>&1 | tee "$LOG_DIR/merge_$TS.log"

echo "[run_all_benchmarks] Output files:"
ls -la "$WITH_TSDIFF_DIR" "$NO_TSDIFF_DIR" "$RESULTS_ROOT" | sed 's/^/[run_all_benchmarks] /'

echo "[run_all_benchmarks] Done."

