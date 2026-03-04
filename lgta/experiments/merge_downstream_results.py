"""
Merge DOWNSTREAM_RESULTS.md from lgta and tsdiff runs into a single report.

Reads with_tsdiff/DOWNSTREAM_RESULTS.md and no_tsdiff/DOWNSTREAM_RESULTS.md,
combines rows by method (preferring with_tsdiff for duplicates), and writes
FINAL_DOWNSTREAM_RESULTS.md.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse

RESULTS_FILENAME = "DOWNSTREAM_RESULTS.md"
OUTPUT_FILENAME = "FINAL_DOWNSTREAM_RESULTS.md"


def merge_downstream_results(
    with_tsdiff_dir: Path,
    no_tsdiff_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Merge markdown result tables from two dirs into one. Prefer with_tsdiff rows for duplicate methods."""
    paths = [
        no_tsdiff_dir / RESULTS_FILENAME,
        with_tsdiff_dir / RESULTS_FILENAME,
    ]
    rows: dict[str, str] = {}
    header: list[str] | None = None
    for p in paths:
        if not p.exists():
            continue
        lines = p.read_text().strip().splitlines()
        if not lines:
            continue
        if header is None and len(lines) >= 3:
            header = lines[1:3]
        for line in lines[3:]:
            if not line.strip().startswith("|"):
                continue
            method = line.split("|")[1].strip()
            rows[method] = line
    out_lines = ["# Combined Downstream Forecasting Results", ""]
    out_lines.extend(header or [])
    for method in sorted(rows.keys()):
        out_lines.append(rows[method])
    out_path = output_path or (with_tsdiff_dir.parent / OUTPUT_FILENAME)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge downstream forecasting results from lgta and tsdiff runs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Parent dir containing with_tsdiff/ and no_tsdiff/ (e.g. multi_env/TS).",
    )
    parser.add_argument(
        "--with-tsdiff",
        type=Path,
        default=None,
        help="Dir with DOWNSTREAM_RESULTS.md from tsdiff env run.",
    )
    parser.add_argument(
        "--no-tsdiff",
        type=Path,
        default=None,
        help="Dir with DOWNSTREAM_RESULTS.md from lgta env run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for merged report (default: results-dir/FINAL_DOWNSTREAM_RESULTS.md).",
    )
    args = parser.parse_args()
    if args.results_dir is not None:
        with_tsdiff_dir = args.results_dir / "with_tsdiff"
        no_tsdiff_dir = args.results_dir / "no_tsdiff"
        output_path = args.output or (args.results_dir / OUTPUT_FILENAME)
    elif args.with_tsdiff is not None and args.no_tsdiff is not None:
        with_tsdiff_dir = args.with_tsdiff
        no_tsdiff_dir = args.no_tsdiff
        output_path = args.output or (with_tsdiff_dir.parent / OUTPUT_FILENAME)
    else:
        parser.error("Either --results-dir or both --with-tsdiff and --no-tsdiff are required.")
    out = merge_downstream_results(with_tsdiff_dir, no_tsdiff_dir, output_path)
    print(f"Wrote combined report to {out}")


if __name__ == "__main__":
    main()
