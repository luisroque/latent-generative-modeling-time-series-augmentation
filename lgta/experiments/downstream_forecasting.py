"""
Downstream forecasting experiment. Compares LGTA against all benchmark
generators by training forecasting models on original-only data vs.
original + synthetic data from each method.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lgta.benchmarks import (
    TimeSeriesGenerator,
    get_default_benchmark_generators,
)
from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from lgta.model.generate_data import generate_synthetic_data
from lgta.model.models import LatentMode
from lgta.preprocessing.pre_processing_datasets import PreprocessDatasets


SEED = 42


@dataclass
class ForecastResult:
    """Stores MSE statistics for one augmentation method + one forecaster."""

    method: str
    forecaster: str
    mse_median: float
    mse_std: float


@dataclass
class ExperimentConfig:
    """Top-level knobs for the downstream forecasting experiment."""

    dataset_name: str = "tourism_small"
    freq: str = "Q"
    window_size: int = 6
    forecast_epochs: int = 200
    forecast_batch_size: int = 32
    n_runs: int = 5
    lgta_transformation: str = "jitter"
    lgta_sigma: float = 0.5
    lgta_epochs: int = 1000
    lgta_latent_dim: int = 16
    benchmark_generators: list[TimeSeriesGenerator] = field(default_factory=list)
    output_dir: Path = Path("assets/results/downstream_forecasting")

    def _cache_key_dict(self) -> dict:
        """Fields that define cache identity (excludes generators and output_dir)."""
        return {
            "dataset_name": self.dataset_name,
            "freq": self.freq,
            "window_size": self.window_size,
            "forecast_epochs": self.forecast_epochs,
            "forecast_batch_size": self.forecast_batch_size,
            "n_runs": self.n_runs,
            "lgta_transformation": self.lgta_transformation,
            "lgta_sigma": self.lgta_sigma,
            "lgta_epochs": self.lgta_epochs,
            "lgta_latent_dim": self.lgta_latent_dim,
            "seed": SEED,
        }


# ---------------------------------------------------------------------------
# Forecasting models
# ---------------------------------------------------------------------------

class _LSTM(nn.Module):
    def __init__(self, n_features: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(torch.relu(out[:, -1, :]))


class _Linear(nn.Module):
    def __init__(self, n_features: int, window_size: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(n_features * window_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.reshape(x.size(0), -1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _prepare_windows(
    data: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _train_and_evaluate(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    ds = TensorDataset(
        torch.from_numpy(X_train).to(device),
        torch.from_numpy(y_train).to(device),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
    mse = float(np.mean((y_test - preds) ** 2))
    return mse, preds.astype(np.float32)


def _run_single(
    forecaster_type: str,
    n_features_in: int,
    n_features_out: int,
    window_size: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[float, np.ndarray]:
    device = _get_device()
    if forecaster_type == "LSTM":
        model = _LSTM(n_features_in, n_features_out).to(device)
    else:
        model = _Linear(n_features_in, window_size, n_features_out).to(device)
    mse, preds = _train_and_evaluate(
        model, X_train, y_train, X_test, y_test,
        cfg.forecast_epochs, cfg.forecast_batch_size, device,
    )
    return mse, preds


def _evaluate_method(
    method_name: str,
    X_data: np.ndarray,
    n_orig_features: int,
    window_size: int,
    X_test_windows: np.ndarray,
    y_test: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[list[ForecastResult], np.ndarray, np.ndarray]:
    """Returns (results, predictions_LSTM, predictions_Linear).

    predictions_* have shape (n_runs, n_test, n_orig_features).
    """
    X_win, y_win = _prepare_windows(X_data, window_size)
    n_test = X_test_windows.shape[0]
    n_total = X_win.shape[0]
    n_train = n_total - n_test

    X_train, y_train = X_win[:n_train], y_win[:n_train, :n_orig_features]

    if X_data.shape[1] > n_orig_features:
        X_test_aug = np.zeros(
            (n_test, window_size, X_data.shape[1]), dtype=np.float32,
        )
        X_test_aug[:, :, :n_orig_features] = X_test_windows
        for i in range(n_test):
            row_start = n_train + i
            X_test_aug[i, :, n_orig_features:] = X_data[
                row_start : row_start + window_size, n_orig_features:
            ]
        X_test_eval = X_test_aug
    else:
        X_test_eval = X_test_windows

    results: list[ForecastResult] = []
    preds_lstm_list: list[np.ndarray] = []
    preds_linear_list: list[np.ndarray] = []
    for forecaster_name in ("LSTM", "Linear"):
        mse_runs: list[float] = []
        for _ in range(cfg.n_runs):
            mse, preds = _run_single(
                forecaster_name, X_data.shape[1], n_orig_features,
                window_size, X_train, y_train, X_test_eval, y_test, cfg,
            )
            mse_runs.append(mse)
            if forecaster_name == "LSTM":
                preds_lstm_list.append(preds)
            else:
                preds_linear_list.append(preds)
        results.append(ForecastResult(
            method=method_name,
            forecaster=forecaster_name,
            mse_median=float(np.median(mse_runs)),
            mse_std=float(np.std(mse_runs)),
        ))
    preds_LSTM = np.stack(preds_lstm_list, axis=0)
    preds_Linear = np.stack(preds_linear_list, axis=0)
    return results, preds_LSTM, preds_Linear


# ---------------------------------------------------------------------------
# Cache (config id, save/load shared and per-method data)
# ---------------------------------------------------------------------------

CACHE_ROOT = Path("assets/cache/downstream_forecasting")


def _config_id(cfg: ExperimentConfig) -> str:
    """Stable hash of config for cache directory name."""
    key = json.dumps(cfg._cache_key_dict(), sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _cache_dir(cfg: ExperimentConfig) -> Path:
    """Cache is always under assets; output_dir only controls results file location."""
    return CACHE_ROOT / _config_id(cfg)


def _method_dir(cache_dir: Path, method_name: str) -> Path:
    return cache_dir / method_name.replace(" ", "_")


def _save_shared_test_data(
    cache_dir: Path,
    X_orig: np.ndarray,
    y_test: np.ndarray,
    X_test_windows: np.ndarray,
    n_orig_features: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "X_orig.npy", X_orig)
    np.save(cache_dir / "y_test.npy", y_test)
    np.save(cache_dir / "X_test_windows.npy", X_test_windows)
    (cache_dir / "n_orig_features.txt").write_text(str(n_orig_features))


def _load_shared_test_data(
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    X_orig = np.load(cache_dir / "X_orig.npy")
    y_test = np.load(cache_dir / "y_test.npy")
    X_test_windows = np.load(cache_dir / "X_test_windows.npy")
    n_orig_features = int((cache_dir / "n_orig_features.txt").read_text())
    return X_orig, y_test, X_test_windows, n_orig_features


def _has_shared_test_data(cache_dir: Path) -> bool:
    return (cache_dir / "y_test.npy").exists() and (cache_dir / "X_test_windows.npy").exists()


def _save_predictions(
    method_dir: Path,
    preds_LSTM: np.ndarray,
    preds_Linear: np.ndarray,
) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    np.save(method_dir / "predictions_LSTM.npy", preds_LSTM)
    np.save(method_dir / "predictions_Linear.npy", preds_Linear)


def _has_predictions(method_dir: Path) -> bool:
    return (method_dir / "predictions_LSTM.npy").exists() and (
        method_dir / "predictions_Linear.npy"
    ).exists()


def _load_predictions(method_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    preds_LSTM = np.load(method_dir / "predictions_LSTM.npy")
    preds_Linear = np.load(method_dir / "predictions_Linear.npy")
    return preds_LSTM, preds_Linear


def _results_from_predictions(
    method_name: str,
    y_test: np.ndarray,
    preds_LSTM: np.ndarray,
    preds_Linear: np.ndarray,
) -> list[ForecastResult]:
    """Compute ForecastResults from saved predictions (n_runs, n_test, n_out)."""
    mse_lstm = np.mean((y_test - preds_LSTM) ** 2, axis=(1, 2))
    mse_linear = np.mean((y_test - preds_Linear) ** 2, axis=(1, 2))
    return [
        ForecastResult(
            method=method_name,
            forecaster="LSTM",
            mse_median=float(np.median(mse_lstm)),
            mse_std=float(np.std(mse_lstm)),
        ),
        ForecastResult(
            method=method_name,
            forecaster="Linear",
            mse_median=float(np.median(mse_linear)),
            mse_std=float(np.std(mse_linear)),
        ),
    ]


def _save_synthetic(method_dir: Path, synthetic: np.ndarray) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    np.save(method_dir / "synthetic.npy", synthetic)


def _has_synthetic(method_dir: Path) -> bool:
    return (method_dir / "synthetic.npy").exists()


def _load_synthetic(method_dir: Path) -> np.ndarray:
    return np.load(method_dir / "synthetic.npy")


# ---------------------------------------------------------------------------
# Data loading and LGTA
# ---------------------------------------------------------------------------

def _load_original_data(cfg: ExperimentConfig) -> np.ndarray:
    """Load the (n_timesteps, n_series) data matrix without training LGTA."""
    ppc = PreprocessDatasets(dataset=cfg.dataset_name, freq=cfg.freq)
    data = ppc.apply_preprocess()
    return data["predict"]["data_matrix"].astype(np.float32)


def _generate_lgta(cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Train LGTA CVAE and generate one augmented dataset.

    Returns (X_orig, X_lgta) both of shape (n_timesteps, n_series).
    """
    creator = CreateTransformedVersionsCVAE(
        dataset_name=cfg.dataset_name, freq=cfg.freq,
    )
    model, _, _ = creator.fit(
        epochs=cfg.lgta_epochs, latent_dim=cfg.lgta_latent_dim,
    )
    _, _, z_mean, _ = creator.predict(model)
    X_orig = creator.X_train_raw

    X_lgta = generate_synthetic_data(
        model, z_mean, creator,
        cfg.lgta_transformation, [cfg.lgta_sigma],
        latent_mode=LatentMode.TEMPORAL,
    )
    return X_orig, X_lgta


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

_BENCHMARK_NAME_ALIASES: dict[str, str] = {
    "timegan": "TimeGANGenerator",
    "timevae": "TimeVAEGenerator",
    "direct": "DirectTransformGenerator",
    "tsdiff": "TSDiffGenerator",
}


def _resolve_method_arg(method: str) -> str:
    """Return canonical method name (original, lgta, or generator class name)."""
    key = method.strip().lower()
    if key in ("original", "lgta"):
        return key if key == "original" else "lgta"
    return _BENCHMARK_NAME_ALIASES.get(key, method)


def _benchmark_matches(gen: TimeSeriesGenerator, method: str) -> bool:
    canonical = _resolve_method_arg(method)
    if canonical in ("original", "lgta"):
        return False
    return gen.__class__.__name__ == canonical


def run_downstream_forecasting(
    cfg: ExperimentConfig | None = None,
    method: str | None = None,
    results_only: bool = False,
) -> list[ForecastResult]:
    """Run the downstream forecasting comparison.

    If method is set, only that method is run: 'original', 'lgta', or a
    benchmark name (e.g. 'timegan', 'TimeGANGenerator'). Otherwise all
    methods are run.

    When results_only is True, no training or inference is run; predictions
    are loaded from cache and results are computed and written. Use this
    after running methods separately to aggregate all cached runs into
    DOWNSTREAM_RESULTS.md.

    Cache: under assets/cache/downstream_forecasting/<config_id>/ we store
    shared test data and per-method predictions (and optionally synthetic).
    output_dir only controls where DOWNSTREAM_RESULTS.md is written. If predictions
    exist for a method, training/inference for that method is skipped.
    """
    if cfg is None:
        cfg = ExperimentConfig()

    cache = _cache_dir(cfg)

    if results_only:
        return _run_results_only(cfg, cache)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    single = method is not None
    if single:
        method_canonical = _resolve_method_arg(method)

    if not cfg.benchmark_generators:
        cfg.benchmark_generators = _default_generators()

    print("=" * 70)
    print("DOWNSTREAM FORECASTING EXPERIMENT")
    if single:
        print(f"  (single method: {method_canonical})")
    print("=" * 70)

    if single and method_canonical == "original":
        print("\n[1/4] Loading data ...")
        X_orig = _load_original_data(cfg)
        X_lgta = None
    elif single and method_canonical not in ("original", "lgta"):
        print("\n[1/4] Loading data ...")
        X_orig = _load_original_data(cfg)
        X_lgta = None
    else:
        print("\n[1/4] Training LGTA and generating synthetic data ...")
        X_orig, X_lgta = _generate_lgta(cfg)

    n_orig_features = X_orig.shape[1]
    X_orig_win, y_orig_win = _prepare_windows(X_orig, cfg.window_size)
    n_test = max(1, int(0.15 * X_orig_win.shape[0]))
    n_train = X_orig_win.shape[0] - n_test
    X_test_windows = X_orig_win[n_train:]
    y_test = y_orig_win[n_train:]

    if not _has_shared_test_data(cache):
        _save_shared_test_data(cache, X_orig, y_test, X_test_windows, n_orig_features)

    all_results: list[ForecastResult] = []

    run_original = not single or method_canonical == "original"
    if run_original:
        orig_dir = _method_dir(cache, "Original")
        if _has_predictions(orig_dir):
            print("\n[2/4] Original: loading from cache ...")
            _, y_test, _, _ = _load_shared_test_data(cache)
            p_lstm, p_lin = _load_predictions(orig_dir)
            all_results.extend(_results_from_predictions("Original", y_test, p_lstm, p_lin))
        else:
            print("\n[2/4] Evaluating Original (no augmentation) ...")
            results, p_lstm, p_lin = _evaluate_method(
                "Original", X_orig, n_orig_features, cfg.window_size,
                X_test_windows, y_test, cfg,
            )
            all_results.extend(results)
            _save_predictions(orig_dir, p_lstm, p_lin)

    run_lgta = (not single or method_canonical == "lgta") and X_lgta is not None
    if run_lgta:
        lgta_dir = _method_dir(cache, "LGTA")
        if _has_predictions(lgta_dir):
            print("\n[3/4] LGTA: loading from cache ...")
            _, y_test, _, _ = _load_shared_test_data(cache)
            p_lstm, p_lin = _load_predictions(lgta_dir)
            all_results.extend(_results_from_predictions("LGTA", y_test, p_lstm, p_lin))
        else:
            print("\n[3/4] Evaluating LGTA ...")
            X_lgta_aug = np.concatenate([X_orig, X_lgta], axis=1)
            results, p_lstm, p_lin = _evaluate_method(
                "LGTA", X_lgta_aug, n_orig_features, cfg.window_size,
                X_test_windows, y_test, cfg,
            )
            all_results.extend(results)
            _save_predictions(lgta_dir, p_lstm, p_lin)
            _save_synthetic(lgta_dir, X_lgta)

    benchmarks_to_run = cfg.benchmark_generators
    if single and method_canonical not in ("original", "lgta"):
        benchmarks_to_run = [g for g in cfg.benchmark_generators if _benchmark_matches(g, method)]
        if not benchmarks_to_run:
            raise ValueError(
                f"Unknown or unavailable method: {method}. "
                f"Choose from: original, lgta, timegan, timevae, direct, tsdiff (if installed)."
            )

    if benchmarks_to_run:
        print(f"\n[4/4] Evaluating {len(benchmarks_to_run)} benchmark method(s) ...")
        for gen in benchmarks_to_run:
            name = gen.name
            method_d = _method_dir(cache, name)
            if _has_predictions(method_d):
                print(f"  {name}: loading from cache ...")
                _, y_test, _, _ = _load_shared_test_data(cache)
                p_lstm, p_lin = _load_predictions(method_d)
                all_results.extend(_results_from_predictions(name, y_test, p_lstm, p_lin))
                continue
            X_synth: np.ndarray
            if _has_synthetic(method_d):
                print(f"  {name}: loading synthetic from cache, training forecasters ...")
                X_synth = _load_synthetic(method_d)
            else:
                print(f"  Training {name} ...")
                gen.fit(X_orig)
                X_synth = gen.generate()
                X_synth = np.clip(X_synth, a_min=0, a_max=None)
                _save_synthetic(method_d, X_synth)
            X_aug = np.concatenate([X_orig, X_synth], axis=1)
            results, p_lstm, p_lin = _evaluate_method(
                name, X_aug, n_orig_features, cfg.window_size,
                X_test_windows, y_test, cfg,
            )
            all_results.extend(results)
            _save_predictions(method_d, p_lstm, p_lin)

    _print_results(all_results)
    _save_results(all_results, cfg.output_dir)
    return all_results


def _run_results_only(cfg: ExperimentConfig, cache: Path) -> list[ForecastResult]:
    """Load all cached predictions and compute results; write DOWNSTREAM_RESULTS.md."""
    if not cache.exists():
        raise FileNotFoundError(
            f"Cache dir not found: {cache}. Run at least one method first so shared test data and predictions exist."
        )
    if not _has_shared_test_data(cache):
        raise FileNotFoundError(
            f"Shared test data missing in {cache}. Run at least one method first."
        )
    _, y_test, _, _ = _load_shared_test_data(cache)
    all_results: list[ForecastResult] = []
    for path in sorted(cache.iterdir()):
        if not path.is_dir():
            continue
        method_name = path.name.replace("_", " ")
        if _has_predictions(path):
            p_lstm, p_lin = _load_predictions(path)
            all_results.extend(_results_from_predictions(method_name, y_test, p_lstm, p_lin))
    if not all_results:
        raise FileNotFoundError(
            f"No cached predictions found under {cache}. Run at least one method first."
        )
    _print_results(all_results)
    _save_results(all_results, cfg.output_dir)
    return all_results


def _default_generators() -> list[TimeSeriesGenerator]:
    return get_default_benchmark_generators(seed=SEED)


def _print_results(results: list[ForecastResult]) -> None:
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    header = f"{'Method':<25s}  {'Forecaster':<10s}  {'MSE (median)':>12s}  {'± std':>10s}"
    print(header)
    print("-" * 70)
    for r in results:
        print(f"{r.method:<25s}  {r.forecaster:<10s}  {r.mse_median:12.4f}  {r.mse_std:10.4f}")
    print("=" * 70)


def _save_results(results: list[ForecastResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Downstream Forecasting Results\n"]
    lines.append("| Method | Forecaster | MSE (median) | ± std |")
    lines.append("|--------|------------|-------------|-------|")
    for r in results:
        lines.append(
            f"| {r.method} | {r.forecaster} | {r.mse_median:.4f} | {r.mse_std:.4f} |"
        )
    out = output_dir / "DOWNSTREAM_RESULTS.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run downstream forecasting experiment (optionally a single method)."
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Run only this method: original, lgta, timegan, timevae, direct, tsdiff (or class name e.g. TimeGANGenerator).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/results/downstream_forecasting"),
        help="Directory to write DOWNSTREAM_RESULTS.md (cache is always under assets/cache/downstream_forecasting).",
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Do not run training/inference; load cached predictions and write DOWNSTREAM_RESULTS.md for all available methods.",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig(output_dir=args.output_dir)
    run_downstream_forecasting(cfg, method=args.method, results_only=args.results_only)
