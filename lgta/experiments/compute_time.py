"""
Compute-time experiment: measure training and generation separately.

For every method and dataset, the training phase (fresh fit, no cached
weights) and the generation phase (one timed call per synthetic variant)
are measured independently.  The reported score is a weighted combination

    weighted = w_train * train_seconds + w_gen * total_generation_seconds

with w_train = w_gen = 0.5 by default, plus each method's rate relative to
LGTA on the same dataset.

Every (dataset, method) measurement is checkpointed under --cache-dir:
trained weights plus a JSON record of the phase timings.  Interrupted runs
resume at the first incomplete measurement; TimeGAN additionally
checkpoints its three internal training phases (embedder / supervisor /
joint) so a stopped run resumes mid-training with accumulated phase times.
The summary (COMPUTE_TIME.md / .csv) is rewritten after every completed
measurement, so partial results are always available.

This experiment uses its own cache tree and never touches the caches of
the downstream forecasting experiment.

Usage:
    python -m lgta.experiments.compute_time --datasets tourism_Q labour_M
    python -m lgta.experiments.compute_time --summary-only
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lgta.benchmarks import (
    BaselineVAEGenerator,
    DiffusionTSGenerator,
    TimeGANGenerator,
    TimeVAEGenerator,
    TimeVAEWindowedGenerator,
)

SEED = 42
WINDOW_SIZE = 10
LGTA_TRANSFORMATIONS = ["jitter", "scaling", "magnitude_warp"]
LGTA_SIGMA = 2.0

DEFAULT_DATASETS = ["tourism_Q", "wiki2_D", "labour_M", "m3_Y", "m3_M"]

BENCHMARK_FACTORIES = {
    "BaselineVAE": lambda: BaselineVAEGenerator(seed=SEED),
    "TimeVAE": lambda: TimeVAEGenerator(seed=SEED),
    "TimeVAEWindowed": lambda: TimeVAEWindowedGenerator(
        seed=SEED, window_size=WINDOW_SIZE
    ),
    "DiffusionTS": lambda: DiffusionTSGenerator(seed=SEED),
    "TimeGAN": lambda: TimeGANGenerator(seed=SEED),
}
ALL_METHODS = ["LGTA"] + list(BENCHMARK_FACTORIES)


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _load_original_data(dataset: str, freq: str) -> np.ndarray:
    from lgta.preprocessing.pre_processing_datasets import PreprocessDatasets

    ppc = PreprocessDatasets(dataset=dataset, freq=freq)
    data = ppc.apply_preprocess()
    return np.nan_to_num(
        data["predict"]["data_matrix"].astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _read_record(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(path)


def measure_benchmark(
    name: str, X: np.ndarray, ckpt_dir: Path, n_variants: int
) -> dict:
    """Measure one benchmark generator; resumes from ckpt_dir if present."""
    record_path = ckpt_dir / f"{name}.json"
    weights_path = ckpt_dir / f"{name}_weights.pt"
    record = _read_record(record_path)

    gen = BENCHMARK_FACTORIES[name]()
    if name == "TimeGAN":
        gen.checkpoint_path = ckpt_dir / "TimeGAN_phases.pt"

    if "train_seconds" not in record or not weights_path.exists():
        print(f"    [{name}] training (fresh) ...", flush=True)
        t0 = time.perf_counter()
        gen.fit(X)
        wall = time.perf_counter() - t0
        # TimeGAN accumulates per-phase times across resumed runs; the sum
        # is the true training cost even when this process only ran the
        # remaining phases.
        train_seconds = (
            sum(gen.phase_seconds.values()) if getattr(gen, "phase_seconds", None)
            else wall
        )
        gen.save_weights(weights_path)
        record = {"train_seconds": train_seconds}
        if getattr(gen, "phase_seconds", None):
            record["phase_seconds"] = gen.phase_seconds
        _write_record(record_path, record)
    else:
        print(f"    [{name}] training cached, loading weights ...", flush=True)
        if not gen.load_weights(weights_path):
            raise RuntimeError(f"missing weights for {name} at {weights_path}")

    if "generate_seconds" not in record:
        gens = []
        for _ in range(n_variants):
            t0 = time.perf_counter()
            gen.generate()
            gens.append(time.perf_counter() - t0)
        record["generate_seconds"] = gens
        _write_record(record_path, record)
    del gen
    _release_memory()
    return record


def measure_lgta(dataset: str, freq: str, ckpt_dir: Path, n_variants: int) -> dict:
    """Measure LGTA: fresh CVAE training, posterior extraction, generation."""
    from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
    from lgta.model.generate_data import generate_synthetic_data
    from lgta.model.models import LatentMode

    record_path = ckpt_dir / "LGTA.json"
    record = _read_record(record_path)
    weights_dir = ckpt_dir / "lgta_model_weights"

    if "train_seconds" not in record:
        # A weights file without a timing record is a partially trained run;
        # retrain from scratch so the measured time covers full training.
        for stale in weights_dir.glob("*.pt"):
            stale.unlink()
        print("    [LGTA] training (fresh) ...", flush=True)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        creator = CreateTransformedVersionsCVAE(
            dataset_name=dataset,
            freq=freq,
            window_size=WINDOW_SIZE,
            weights_suffix="eq1_0_nodyn",
            weights_dir=weights_dir,
            use_dynamic_features=False,
        )
        t0 = time.perf_counter()
        model, _, _ = creator.fit(
            epochs=1000,
            latent_dim=4,
            equiv_weight=1.0,
            latent_mode=LatentMode.TEMPORAL,
        )
        record = {"train_seconds": time.perf_counter() - t0}
        _write_record(record_path, record)
    else:
        print("    [LGTA] training cached, loading weights ...", flush=True)
        creator = CreateTransformedVersionsCVAE(
            dataset_name=dataset,
            freq=freq,
            window_size=WINDOW_SIZE,
            weights_suffix="eq1_0_nodyn",
            weights_dir=weights_dir,
            use_dynamic_features=False,
        )
        model, _, _ = creator.fit(
            epochs=1000,
            latent_dim=4,
            equiv_weight=1.0,
            latent_mode=LatentMode.TEMPORAL,
        )

    if "generate_seconds" not in record:
        t0 = time.perf_counter()
        _, _, z_mean, _ = creator.predict(model)
        record["posterior_seconds"] = time.perf_counter() - t0
        gens = []
        transformations = (LGTA_TRANSFORMATIONS * n_variants)[:n_variants]
        for transformation in transformations:
            rng = np.random.default_rng(SEED)
            t0 = time.perf_counter()
            generate_synthetic_data(
                model,
                z_mean,
                creator,
                transformation,
                [LGTA_SIGMA],
                latent_mode=LatentMode.TEMPORAL,
                rng=rng,
            )
            gens.append(time.perf_counter() - t0)
        record["generate_seconds"] = gens
        _write_record(record_path, record)
    del creator, model
    _release_memory()
    return record


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _collect_records(cache_dir: Path) -> dict[str, dict[str, dict]]:
    """cache layout: <cache_dir>/<dataset_key>/<Method>.json"""
    collected: dict[str, dict[str, dict]] = {}
    if not cache_dir.exists():
        return collected
    for ds_dir in sorted(cache_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        for record_path in ds_dir.glob("*.json"):
            method = record_path.stem
            if method not in ALL_METHODS:
                continue
            record = _read_record(record_path)
            if "train_seconds" in record:
                collected.setdefault(ds_dir.name, {})[method] = record
    return collected


def _gen_totals(record: dict) -> tuple[float | None, float | None]:
    """(per-variant mean, total incl. one-time posterior extraction)."""
    gens = record.get("generate_seconds")
    if not gens:
        return None, None
    per_variant = float(np.mean(gens))
    total = float(np.sum(gens)) + float(record.get("posterior_seconds", 0.0))
    return per_variant, total


def write_summary(
    cache_dir: Path, output_dir: Path, w_train: float, w_gen: float
) -> None:
    collected = _collect_records(cache_dir)
    if not collected:
        print("No completed measurements found; nothing to summarise.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    header = [
        "Dataset",
        "Method",
        "Train (s)",
        "Gen/variant (s)",
        "Gen total (s)",
        "Weighted (s)",
        "Rate vs LGTA",
    ]
    md = [
        "# Computational Time: Training vs Generation",
        "",
        f"Weighted (s) = {w_train} * train + {w_gen} * total generation time "
        "(all variants; for LGTA the one-time posterior extraction is counted "
        "in the generation total). Rate vs LGTA compares the weighted score "
        "on the same dataset.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    csv_rows = [header]

    method_order = [m for m in ALL_METHODS]
    totals: dict[str, dict[str, float]] = {}
    n_datasets_by_method: dict[str, int] = {}

    for ds_key in sorted(collected):
        records = collected[ds_key]
        lgta_weighted = None
        rows = []
        for method in method_order:
            record = records.get(method)
            if record is None:
                continue
            train_s = float(record["train_seconds"])
            gen_pv, gen_total = _gen_totals(record)
            weighted = (
                w_train * train_s + w_gen * gen_total
                if gen_total is not None
                else None
            )
            if method == "LGTA" and weighted is not None:
                lgta_weighted = weighted
            rows.append((method, train_s, gen_pv, gen_total, weighted))
            if weighted is not None:
                agg = totals.setdefault(
                    method, {"train": 0.0, "gen": 0.0, "weighted": 0.0}
                )
                agg["train"] += train_s
                agg["gen"] += gen_total
                agg["weighted"] += weighted
                n_datasets_by_method[method] = n_datasets_by_method.get(method, 0) + 1

        for method, train_s, gen_pv, gen_total, weighted in rows:
            rate = (
                f"{weighted / lgta_weighted:.2f}x"
                if weighted is not None and lgta_weighted
                else ""
            )
            cells = [
                ds_key,
                method,
                f"{train_s:.2f}",
                f"{gen_pv:.4f}" if gen_pv is not None else "",
                f"{gen_total:.4f}" if gen_total is not None else "",
                f"{weighted:.2f}" if weighted is not None else "",
                rate,
            ]
            md.append("| " + " | ".join(cells) + " |")
            csv_rows.append(cells)

    md += ["", "## Totals (datasets measured so far)", ""]
    md.append("| Method | Datasets | Train (s) | Gen total (s) | Weighted (s) | Rate vs LGTA |")
    md.append("|---|---|---|---|---|---|")
    lgta_total = totals.get("LGTA", {}).get("weighted")
    for method in method_order:
        if method not in totals:
            continue
        agg = totals[method]
        rate = (
            f"{agg['weighted'] / lgta_total:.2f}x"
            if lgta_total and n_datasets_by_method.get(method)
            == n_datasets_by_method.get("LGTA")
            else "n/a"
        )
        md.append(
            f"| {method} | {n_datasets_by_method[method]} | {agg['train']:.2f} | "
            f"{agg['gen']:.4f} | {agg['weighted']:.2f} | {rate} |"
        )

    (output_dir / "COMPUTE_TIME.md").write_text("\n".join(md) + "\n")
    with (output_dir / "COMPUTE_TIME.csv").open("w") as f:
        for row in csv_rows:
            f.write(",".join(row) + "\n")
    print(f"Summary written to {output_dir / 'COMPUTE_TIME.md'}")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure training and generation time separately per method."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help="Dataset keys like tourism_Q wiki2_D labour_M m3_Y m3_M.")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS)
    parser.add_argument("--n-variants", type=int, default=3)
    parser.add_argument("--w-train", type=float, default=0.5)
    parser.add_argument("--w-gen", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("assets/cache/compute_time"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("assets/results/compute_time"))
    parser.add_argument("--summary-only", action="store_true",
                        help="Only rebuild the summary from existing checkpoints.")
    args = parser.parse_args()

    if args.summary_only:
        write_summary(args.cache_dir, args.output_dir, args.w_train, args.w_gen)
        return

    for ds_key in args.datasets:
        dataset, freq = ds_key.rsplit("_", 1)
        ckpt_dir = args.cache_dir / ds_key
        print(f"===== {ds_key} =====", flush=True)
        X = _load_original_data(dataset, freq)
        print(f"  data matrix: {X.shape}", flush=True)
        for method in args.methods:
            print(f"  --- {method} ---", flush=True)
            t_start = time.perf_counter()
            if method == "LGTA":
                record = measure_lgta(dataset, freq, ckpt_dir, args.n_variants)
            else:
                record = measure_benchmark(method, X, ckpt_dir, args.n_variants)
            gen_pv, gen_total = _gen_totals(record)
            print(
                f"  {method}: train={record['train_seconds']:.2f}s  "
                f"gen/variant={gen_pv:.4f}s  gen_total={gen_total:.4f}s  "
                f"(wall {time.perf_counter() - t_start:.1f}s)",
                flush=True,
            )
            write_summary(args.cache_dir, args.output_dir, args.w_train, args.w_gen)

    print("COMPUTE_TIME_DONE", flush=True)


if __name__ == "__main__":
    main()
