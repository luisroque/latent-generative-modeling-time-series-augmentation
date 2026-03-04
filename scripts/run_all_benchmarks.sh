#!/usr/bin/env bash
set -euo pipefail

# Orchestrate downstream forecasting runs in different environments and
# produce a combined report.
#
# - Main env (lgta): runs all benchmarks that do not require special deps.
# - Benchmarks env (lgta-benchmarks): same experiment but with TSDiff
#   available if its dependencies are installed there.
#
# Usage:
#   bash scripts/run_all_benchmarks.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ENV="lgta"
BENCH_ENV="lgta-benchmarks"
TS="$(date +%Y%m%d-%H%M%S)"

RESULTS_ROOT="$ROOT_DIR/assets/results/downstream_forecasting/multi_env/$TS"
LOG_DIR="$RESULTS_ROOT/logs"
WITH_TSDIFF_DIR="$RESULTS_ROOT/with_tsdiff"
NO_TSDIFF_DIR="$RESULTS_ROOT/no_tsdiff"
FINAL_REPORT="$RESULTS_ROOT/FINAL_DOWNSTREAM_RESULTS.md"

cd "$ROOT_DIR"

echo "[run_all_benchmarks] Root dir: $ROOT_DIR"
echo "[run_all_benchmarks] Timestamp: $TS"

mkdir -p "$LOG_DIR" "$WITH_TSDIFF_DIR" "$NO_TSDIFF_DIR"
echo "[run_all_benchmarks] Logs dir: $LOG_DIR"
echo "[run_all_benchmarks] Results dirs: $WITH_TSDIFF_DIR , $NO_TSDIFF_DIR"

echo "[run_all_benchmarks] Running downstream_forecasting in benchmarks env: $BENCH_ENV"
OUTPUT_DIR="$WITH_TSDIFF_DIR" conda run --no-capture-output -n "$BENCH_ENV" python -u - << 'PY' 2>&1 | tee "$LOG_DIR/benchmarks_env_$TS.log"
import os
from pathlib import Path
from lgta.experiments.downstream_forecasting import ExperimentConfig, run_downstream_forecasting

print("[benchmarks-env] Starting downstream_forecasting with TSDiff-capable env...")
cfg = ExperimentConfig()
cfg.output_dir = Path(os.environ["OUTPUT_DIR"])
print(f"[benchmarks-env] Using output_dir={cfg.output_dir}")
results = run_downstream_forecasting(cfg)
print(f"[benchmarks-env] Finished downstream_forecasting. Collected {len(results)} rows.")
PY

echo "[run_all_benchmarks] Running downstream_forecasting in main env: $MAIN_ENV"
OUTPUT_DIR="$NO_TSDIFF_DIR" conda run --no-capture-output -n "$MAIN_ENV" python -u - << 'PY' 2>&1 | tee "$LOG_DIR/main_env_$TS.log"
import os
from pathlib import Path
from lgta.experiments.downstream_forecasting import ExperimentConfig, run_downstream_forecasting

print("[main-env] Starting downstream_forecasting without TSDiff...")
cfg = ExperimentConfig()
cfg.output_dir = Path(os.environ["OUTPUT_DIR"])
print(f"[main-env] Using output_dir={cfg.output_dir}")
results = run_downstream_forecasting(cfg)
print(f"[main-env] Finished downstream_forecasting. Collected {len(results)} rows.")
PY

echo "[run_all_benchmarks] Merging results into: $FINAL_REPORT"
conda run --no-capture-output -n "$MAIN_ENV" python -u - << 'PY' 2>&1 | tee "$LOG_DIR/merge_$TS.log"
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "assets" / "results" / "downstream_forecasting"  # unused but kept for clarity
paths = [
    Path("assets/results/downstream_forecasting/multi_env") / Path("'"$TS"'") / "with_tsdiff" / "DOWNSTREAM_RESULTS.md",
    Path("assets/results/downstream_forecasting/multi_env") / Path("'"$TS"'") / "no_tsdiff" / "DOWNSTREAM_RESULTS.md",
]

rows = {}
header = None
for p in paths:
    if not p.exists():
        continue
    lines = p.read_text().strip().splitlines()
    if not lines:
        continue
    if header is None and len(lines) >= 3:
        header = lines[1:3]  # table header + separator
    for line in lines[3:]:
        if not line.strip().startswith("|"):
            continue
        method = line.split("|")[1].strip()
        # Prefer rows from with_tsdiff for methods that appear in both.
        rows[method] = line

out_lines = ["# Combined Downstream Forecasting Results", ""]
out_lines.extend(header or [])
for method in sorted(rows.keys()):
    out_lines.append(rows[method])

out_path = Path("assets/results/downstream_forecasting/multi_env") / Path("'"$TS"'") / "FINAL_DOWNSTREAM_RESULTS.md"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(out_lines) + "\n")
print(f"Wrote combined report to {out_path}")
PY

echo "[run_all_benchmarks] Output files:"
ls -la "$WITH_TSDIFF_DIR" "$NO_TSDIFF_DIR" "$RESULTS_ROOT" | sed 's/^/[run_all_benchmarks] /'

echo "[run_all_benchmarks] Done."

