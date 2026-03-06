#!/usr/bin/env bash
# Remove only forecaster prediction caches so LSTM/Linear are re-run with scaling.
# Run from repo root. Pass dataset name(s) as args, or none to clear all datasets from config.

set -e
CACHE_ROOT="assets/cache/downstream_forecasting"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ $# -eq 0 ]; then
  datasets=()
  while IFS= read -r name; do
    [ -n "$name" ] && datasets+=( "$name" )
  done < <(conda run -n lgta python -c "
import sys
sys.path.insert(0, '.')
from lgta.experiments.downstream_forecasting import DEFAULT_DATASET_CONFIGS
for name, _ in DEFAULT_DATASET_CONFIGS:
    print(name)
")
else
  datasets=("$@")
fi

for dataset in "${datasets[@]}"; do
  dir="$CACHE_ROOT/$dataset"
  [ ! -d "$dir" ] && continue
  echo "Clearing forecaster caches in $dir"
  for method_dir in "$dir"/Original "$dir"/LGTA_* "$dir"/TimeGANGenerator "$dir"/TimeVAEGenerator "$dir"/DiffusionTSGenerator "$dir"/Direct*; do
    [ -d "$method_dir" ] || continue
    rm -f "$method_dir"/predictions_LSTM.npy "$method_dir"/predictions_Linear.npy \
          "$method_dir"/fitted_LSTM.npy "$method_dir"/fitted_Linear.npy 2>/dev/null || true
  done
done
echo "Done. Re-run downstream forecasting to regenerate predictions with scaling."
