"""
Synthesis quality experiment: compare synthetic vs real data using pymdma metrics.

Uses the same cache as downstream_forecasting; run that experiment first to
populate synthetic data, then run this script to compute metrics.
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from typing import Any

import numpy as np

from lgta.evaluation.metrics import MetricsAggregator

FREQ_TO_SAMPLING_FREQ: dict[str, int] = {
    "Y": 1,
    "A": 1,
    "Q": 4,
    "M": 12,
    "W": 52,
    "D": 365,
    "H": 8760,
}


def _run_synthesis_quality_for_config(
    cfg: Any,
) -> dict[str, dict[str, dict[str, float]]]:
    """Load cache, compute metrics per method; return results_by_method."""
    from lgta.experiments.downstream_forecasting import (
        _cache_dir,
        _has_shared_test_data,
        _load_shared_test_data,
        _load_synthetic_variants,
        _method_dir_for,
        _methods_with_synthetic,
    )

    cache_dir = _cache_dir(cfg)
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Cache dir not found: {cache_dir}. "
            "Run downstream_forecasting first to generate synthetic data."
        )
    if not _has_shared_test_data(cache_dir):
        raise FileNotFoundError(
            f"Shared test data missing in {cache_dir}. "
            "Run downstream_forecasting first."
        )

    X_orig, _, _, _, _, _, _, _ = _load_shared_test_data(cache_dir, cfg.window_size)
    methods = _methods_with_synthetic(cache_dir, cfg)
    if not methods:
        raise FileNotFoundError(
            f"No methods with cached synthetic data in {cache_dir}. "
            "Run downstream_forecasting for at least one method (e.g. LGTA)."
        )

    sampling_freq = FREQ_TO_SAMPLING_FREQ.get(cfg.freq.upper().strip(), 1)
    feature_cache = cache_dir / "tsfel_features"
    aggregator = MetricsAggregator(
        sampling_freq=sampling_freq, cache_dir=feature_cache
    )
    results_by_method: dict[str, dict[str, dict[str, float]]] = {}

    for method_name in methods:
        method_dir = _method_dir_for(cache_dir, method_name, cfg)
        variants = _load_synthetic_variants(method_dir, cfg.n_variants)
        X_synthetic = variants[0]
        metrics = aggregator.compute_metrics_single(X_orig, X_synthetic)
        results_by_method[method_name] = metrics

    return results_by_method


def _flatten_for_json(
    results_by_method: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    """Convert nested results to a list of records (one per method) for JSON."""
    rows: list[dict[str, Any]] = []
    for method, categories in results_by_method.items():
        row: dict[str, Any] = {"method": method}
        for cat_name, metrics in categories.items():
            for k, v in metrics.items():
                key = f"{cat_name}.{k}"
                row[key] = (
                    float(v) if np.isscalar(v) and not isinstance(v, (bool,)) else v
                )
        rows.append(row)
    return rows


def _save_results(
    results_by_method: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
) -> None:
    """Write synthesis_quality_results.json and SYNTHESIS_QUALITY_RESULTS.md."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _flatten_for_json(results_by_method)
    json_path = output_dir / "synthesis_quality_results.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Results saved to {json_path}")

    md_lines = [
        "# Synthesis Quality Results",
        "",
        "Research question: Can L-GTA generate synthetic data with **greater privacy** "
        "(higher authenticity) than benchmarks? Privacy (authenticity): higher = more privacy.",
        "",
        "| Method | authenticity (privacy) | improved_precision | density | improved_recall | coverage | ... |",
        "|--------|-------------------------|---------------------|--------|-----------------|----------|-----|",
    ]
    for method, categories in results_by_method.items():
        auth = categories.get("privacy", {}).get("authenticity", float("nan"))
        fid = categories.get("fidelity", {})
        div = categories.get("diversity", {})
        prec = fid.get("improved_precision", float("nan"))
        dens = fid.get("density", float("nan"))
        rec = div.get("improved_recall", float("nan"))
        cov = div.get("coverage", float("nan"))
        md_lines.append(
            f"| {method} | {auth:.4f} | {prec:.4f} | {dens:.4f} | {rec:.4f} | {cov:.4f} | ... |"
        )
    md_path = output_dir / "SYNTHESIS_QUALITY_RESULTS.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Markdown saved to {md_path}")


def _plot_authenticity(
    results_by_method: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
) -> None:
    """Bar plot of authenticity (privacy) per method."""
    import matplotlib.pyplot as plt

    methods = list(results_by_method.keys())
    authenticity = [
        results_by_method[m].get("privacy", {}).get("authenticity", float("nan"))
        for m in methods
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(methods))
    bars = ax.bar(x, authenticity, color="steelblue", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Authenticity (privacy)")
    ax.set_title("Privacy: Authenticity by method (higher = more privacy)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / "authenticity_by_method.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


def run_synthesis_quality(cfg: Any) -> dict[str, dict[str, dict[str, float]]]:
    """Load synthetic data from downstream_forecasting cache and compute synthesis quality metrics.

    Returns results_by_method: method_name -> { fidelity, diversity, privacy }.
    """
    from lgta.experiments.downstream_forecasting import (
        ExperimentConfig,
        _cache_dir,
        _lgta_config_slug,
    )

    print("=" * 70)
    print("SYNTHESIS QUALITY EXPERIMENT (synthetic vs real metrics)")
    print(
        "Research question: Does L-GTA achieve greater privacy (authenticity) than benchmarks?"
    )
    print("=" * 70)

    results_by_method = _run_synthesis_quality_for_config(cfg)
    effective_out = (
        cfg.output_dir
        / cfg.eval_mode.value
        / cfg.dynamic_subdir
        / _lgta_config_slug(cfg)
    )
    _save_results(results_by_method, effective_out)
    _plot_authenticity(results_by_method, effective_out)

    auth_by_method = {
        m: results_by_method[m].get("privacy", {}).get("authenticity", float("nan"))
        for m in results_by_method
    }
    best = max(auth_by_method, key=lambda m: auth_by_method[m])
    print(f"\nPrivacy (authenticity) winner: {best} ({auth_by_method[best]:.4f})")
    print(f"Results written to: {effective_out}")
    return results_by_method


def main() -> None:
    from lgta.experiments.downstream_forecasting import (
        DEFAULT_DATASET_CONFIGS,
        ExperimentConfig,
        EvalMode,
        get_default_benchmark_generators,
    )

    import argparse

    parser = argparse.ArgumentParser(
        description="Compute synthesis quality metrics (privacy, fidelity, diversity, data quality) from downstream_forecasting cache."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g. tourism, wiki2). Default: tourism.",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default=None,
        help="Frequency (e.g. Q, M, D). Default: from dataset.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run for all supported datasets with their default frequencies.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/results/synthesis_quality"),
        help="Base directory for results.",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="TSTR",
        choices=[m.value for m in EvalMode],
        help="Must match the eval_mode used when populating the cache.",
    )
    parser.add_argument(
        "--no-dynamic-features",
        action="store_true",
        help="Match cache populated with --no-dynamic-features.",
    )
    parser.add_argument(
        "--variant-transformations",
        nargs="+",
        type=str,
        default=None,
        help="Match downstream_forecasting variant transformations if used.",
    )
    args = parser.parse_args()

    variant_transformations = args.variant_transformations or []
    eval_mode = EvalMode(args.eval_mode)
    dynamic_settings: list[bool] = (
        [True, False]
        if args.all_datasets and not args.no_dynamic_features
        else [not args.no_dynamic_features]
    )

    if args.all_datasets:
        for use_dyn in dynamic_settings:
            for dataset_name, freq in DEFAULT_DATASET_CONFIGS:
                cfg = ExperimentConfig(
                    dataset_name=dataset_name,
                    freq=freq,
                    output_dir=args.output_dir,
                    variant_transformations=variant_transformations,
                    eval_mode=eval_mode,
                    use_dynamic_features=use_dyn,
                    benchmark_generators=get_default_benchmark_generators(seed=42),
                )
                run_synthesis_quality(cfg)
    else:
        dataset_name = args.dataset if args.dataset is not None else "tourism"
        freq = args.freq
        if freq is None:
            freq = next(
                (f for d, f in DEFAULT_DATASET_CONFIGS if d == dataset_name), "Q"
            )
        cfg = ExperimentConfig(
            dataset_name=dataset_name,
            freq=freq,
            output_dir=args.output_dir,
            variant_transformations=variant_transformations,
            eval_mode=eval_mode,
            use_dynamic_features=not args.no_dynamic_features,
            benchmark_generators=get_default_benchmark_generators(seed=42),
        )
        run_synthesis_quality(cfg)


if __name__ == "__main__":
    main()
