"""
Downstream forecasting experiment. Compares LGTA against all benchmark
generators by training forecasting models on original-only data vs.
original + synthetic data from each method.

Can be invoked directly (python lgta/experiments/downstream_forecasting.py)
or as a module (python -m lgta.experiments.downstream_forecasting) provided
the repo root is on PYTHONPATH or you run from the repo root.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lgta.benchmarks import (
    TimeSeriesGenerator,
    get_default_benchmark_generators,
)


SEED = 42

DEFAULT_DATASET_CONFIGS: list[tuple[str, str]] = [
    ("tourism_small", "Q"),
    ("tourism", "Q"),
    ("wiki2", "D"),
    ("labour", "M"),
    ("m3", "Q"),
]


@dataclass
class ForecastResult:
    """Stores MASE statistics for one augmentation method + one forecaster."""

    method: str
    forecaster: str
    mase_mean: float
    mase_std: float


@dataclass
class ExperimentConfig:
    """Top-level knobs for the downstream forecasting experiment."""

    dataset_name: str = "tourism_small"
    freq: str = "Q"
    window_size: int = 10
    forecast_epochs: int = 200
    forecast_batch_size: int = 32
    n_runs: int = 5
    lgta_transformation: str = "jitter"
    lgta_sigma: float = 2.0
    lgta_epochs: int = 1000
    lgta_latent_dim: int = 4
    lgta_equiv_weight: float = 1.0
    lgta_sample_from_posterior: bool = False
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
            "lgta_equiv_weight": self.lgta_equiv_weight,
            "lgta_sample_from_posterior": self.lgta_sample_from_posterior,
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


def _mase_scale(X_train: np.ndarray, y_train: np.ndarray) -> float:
    """In-sample MAE of naive (persistence) forecast. Used as MASE denominator."""
    n_out = y_train.shape[1]
    naive = X_train[:, -1, :n_out].astype(np.float64)
    y = y_train.astype(np.float64)
    scale = float(np.mean(np.abs(y - naive)))
    return scale if scale > 0 else 1.0


def _train_and_evaluate(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale: float,
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
        fitted = model(torch.from_numpy(X_train).to(device)).cpu().numpy()
    mae = float(np.mean(np.abs(y_test.astype(np.float64) - preds.astype(np.float64))))
    mase = mae / scale
    return mase, preds.astype(np.float32), fitted.astype(np.float32)


def _run_single(
    forecaster_type: str,
    n_features_in: int,
    n_features_out: int,
    window_size: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale: float,
    cfg: ExperimentConfig,
) -> tuple[float, np.ndarray]:
    device = _get_device()
    if forecaster_type == "LSTM":
        model = _LSTM(n_features_in, n_features_out).to(device)
    else:
        model = _Linear(n_features_in, window_size, n_features_out).to(device)
    mase, preds, fitted = _train_and_evaluate(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        scale,
        cfg.forecast_epochs,
        cfg.forecast_batch_size,
        device,
    )
    return mase, preds, fitted


def _evaluate_method(
    method_name: str,
    X_data: np.ndarray,
    n_orig_features: int,
    window_size: int,
    X_test_windows: np.ndarray,
    y_test: np.ndarray,
    scale: float,
    cfg: ExperimentConfig,
    y_train_mean: np.ndarray | None = None,
    y_train_std: np.ndarray | None = None,
) -> tuple[list[ForecastResult], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (results, predictions_LSTM, predictions_Linear, fitted_LSTM, fitted_Linear).

    predictions_* have shape (n_runs, n_test, n_orig_features).
    fitted_* have shape (n_runs, n_train, n_orig_features).
    If y_train_mean and y_train_std are provided, targets are scaled before training
    and predictions/fitted are unscaled before return (so outputs stay in original scale).
    """
    X_win, y_win = _prepare_windows(X_data, window_size)
    n_test = X_test_windows.shape[0]
    n_total = X_win.shape[0]
    n_train = n_total - n_test

    X_train, y_train = X_win[:n_train], y_win[:n_train, :n_orig_features].astype(
        np.float64
    )
    y_test_f = y_test.astype(np.float64)
    use_scale = y_train_mean is not None and y_train_std is not None
    if use_scale:
        y_train = (y_train - y_train_mean) / y_train_std
        y_test_f = (y_test_f - y_train_mean) / y_train_std

    if X_data.shape[1] > n_orig_features:
        X_test_aug = np.zeros(
            (n_test, window_size, X_data.shape[1]),
            dtype=np.float32,
        )
        X_test_aug[:, :, :n_orig_features] = X_test_windows
        for i in range(n_test):
            row_start = n_train + i
            X_test_aug[i, :, n_orig_features:] = X_data[
                row_start : row_start + window_size, n_orig_features:
            ]
        X_test_eval = X_test_aug
    else:
        X_test_eval = np.asarray(X_test_windows, dtype=np.float32).copy()

    if use_scale:
        # Scale original channels (0 .. n_orig_features-1) with per-series stats.
        X_train = X_train.astype(np.float64)
        X_train[:, :, :n_orig_features] = (
            X_train[:, :, :n_orig_features] - y_train_mean
        ) / y_train_std

        X_test_eval = X_test_eval.astype(np.float64)
        X_test_eval[:, :, :n_orig_features] = (
            X_test_eval[:, :, :n_orig_features] - y_train_mean
        ) / y_train_std

        # Scale synthetic channels (n_orig_features .. end) with their own per-series stats.
        if X_data.shape[1] > n_orig_features:
            syn_start = n_orig_features
            syn_end = X_data.shape[1]
            syn_slice = slice(syn_start, syn_end)

            syn_train = X_train[:, :, syn_slice]
            syn_mean = np.mean(syn_train, axis=(0, 1), dtype=np.float64)
            syn_std = np.std(syn_train, axis=(0, 1), dtype=np.float64)
            syn_std = np.maximum(syn_std, 1e-8)

            X_train[:, :, syn_slice] = (syn_train - syn_mean) / syn_std

            syn_test = X_test_eval[:, :, syn_slice]
            X_test_eval[:, :, syn_slice] = (syn_test - syn_mean) / syn_std

        X_train = X_train.astype(np.float32)
        X_test_eval = X_test_eval.astype(np.float32)

    y_test_for_train = y_test_f.astype(np.float32)
    results: list[ForecastResult] = []
    preds_lstm_list: list[np.ndarray] = []
    preds_linear_list: list[np.ndarray] = []
    fitted_lstm_list: list[np.ndarray] = []
    fitted_linear_list: list[np.ndarray] = []
    for forecaster_name in ("LSTM", "Linear"):
        mase_runs: list[float] = []
        for _ in range(cfg.n_runs):
            mase, preds, fitted = _run_single(
                forecaster_name,
                X_data.shape[1],
                n_orig_features,
                window_size,
                X_train,
                y_train.astype(np.float32),
                X_test_eval,
                y_test_for_train,
                scale,
                cfg,
            )
            mase_runs.append(mase)
            if forecaster_name == "LSTM":
                preds_lstm_list.append(preds)
                fitted_lstm_list.append(fitted)
            else:
                preds_linear_list.append(preds)
                fitted_linear_list.append(fitted)
        if not use_scale:
            results.append(
                ForecastResult(
                    method=method_name,
                    forecaster=forecaster_name,
                    mase_mean=float(np.mean(mase_runs)),
                    mase_std=float(np.std(mase_runs)),
                )
            )
    preds_LSTM = np.stack(preds_lstm_list, axis=0)
    preds_Linear = np.stack(preds_linear_list, axis=0)
    fitted_LSTM = np.stack(fitted_lstm_list, axis=0)
    fitted_Linear = np.stack(fitted_linear_list, axis=0)
    if use_scale:
        preds_LSTM = preds_LSTM * y_train_std + y_train_mean
        preds_Linear = preds_Linear * y_train_std + y_train_mean
        fitted_LSTM = fitted_LSTM * y_train_std + y_train_mean
        fitted_Linear = fitted_Linear * y_train_std + y_train_mean
        mase_lstm = np.mean(np.abs(y_test - preds_LSTM), axis=(1, 2)) / scale
        mase_linear = np.mean(np.abs(y_test - preds_Linear), axis=(1, 2)) / scale
        results = [
            ForecastResult(
                method=method_name,
                forecaster="LSTM",
                mase_mean=float(np.mean(mase_lstm)),
                mase_std=float(np.std(mase_lstm)),
            ),
            ForecastResult(
                method=method_name,
                forecaster="Linear",
                mase_mean=float(np.mean(mase_linear)),
                mase_std=float(np.std(mase_linear)),
            ),
        ]
    return results, preds_LSTM, preds_Linear, fitted_LSTM, fitted_Linear


# ---------------------------------------------------------------------------
# Cache (config id, save/load shared and per-method data)
# ---------------------------------------------------------------------------

CACHE_ROOT = Path("assets/cache/downstream_forecasting")


def _lgta_config_slug(cfg: ExperimentConfig) -> str:
    """Filesystem-safe folder name from LGTA-related config (dataset, freq, window, transformation, sigma, epochs, latent_dim, equiv_weight, sample_from_posterior)."""
    sigma_str = str(cfg.lgta_sigma).replace(".", "_")
    parts = [
        cfg.dataset_name,
        cfg.freq,
        f"w{cfg.window_size}",
        cfg.lgta_transformation,
        f"sig{sigma_str}",
        f"ep{cfg.lgta_epochs}",
        f"lat{cfg.lgta_latent_dim}",
        f"eq{cfg.lgta_equiv_weight}",
    ]
    if cfg.lgta_sample_from_posterior:
        parts.append("sample")
    return "_".join(str(p) for p in parts)


def _cache_dir(cfg: ExperimentConfig) -> Path:
    """Cache root per dataset so weights, test data, and predictions do not collide across datasets."""
    return CACHE_ROOT / cfg.dataset_name


def _method_dir(cache_dir: Path, method_name: str) -> Path:
    return cache_dir / method_name.replace(" ", "_")


def _lgta_method_dir(cache_dir: Path, cfg: ExperimentConfig) -> Path:
    """LGTA cache dir keyed by config so different sigma/transformation/latent_dim get separate cache."""
    return cache_dir / ("LGTA_" + _lgta_config_slug(cfg))


def _method_dir_for(
    cache_dir: Path, method_name: str, cfg: ExperimentConfig
) -> Path:
    """Resolve method name to cache dir; LGTA uses config-keyed subdir."""
    if method_name == "LGTA":
        return _lgta_method_dir(cache_dir, cfg)
    return _method_dir(cache_dir, method_name)


def _known_cache_method_names() -> list[str]:
    """Method names we may have in cache: Original, LGTA, and default benchmark names."""
    return ["Original", "LGTA"] + [
        g.name for g in get_default_benchmark_generators(seed=SEED)
    ]


def _save_shared_test_data(
    cache_dir: Path,
    X_orig: np.ndarray,
    y_test: np.ndarray,
    X_test_windows: np.ndarray,
    n_orig_features: int,
    scale: float,
    window_size: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "X_orig.npy", X_orig)
    np.save(cache_dir / "y_test.npy", y_test)
    np.save(cache_dir / "X_test_windows.npy", X_test_windows)
    np.save(cache_dir / "scale.npy", np.array(scale, dtype=np.float64))
    (cache_dir / "n_orig_features.txt").write_text(str(n_orig_features))
    (cache_dir / "window_size.txt").write_text(str(window_size))
    _, y_win = _prepare_windows(X_orig, window_size)
    n_test = X_test_windows.shape[0]
    y_train = y_win[: y_win.shape[0] - n_test, :n_orig_features]
    y_train_mean = np.mean(y_train, axis=0).astype(np.float64)
    y_train_std = np.std(y_train, axis=0).astype(np.float64)
    y_train_std = np.maximum(y_train_std, 1e-8)
    np.save(cache_dir / "y_train_mean.npy", y_train_mean)
    np.save(cache_dir / "y_train_std.npy", y_train_std)


def _load_shared_test_data(
    cache_dir: Path,
    window_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray | None, np.ndarray | None]:
    X_orig = np.load(cache_dir / "X_orig.npy")
    y_test = np.load(cache_dir / "y_test.npy")
    X_test_windows = np.load(cache_dir / "X_test_windows.npy")
    n_orig_features = int((cache_dir / "n_orig_features.txt").read_text())
    scale_path = cache_dir / "scale.npy"
    if scale_path.exists():
        scale = float(np.load(scale_path))
    else:
        w = int((cache_dir / "window_size.txt").read_text()) if (cache_dir / "window_size.txt").exists() else (window_size or 10)
        X_win, y_win = _prepare_windows(X_orig, w)
        n_test = X_test_windows.shape[0]
        n_train = X_win.shape[0] - n_test
        scale = _mase_scale(X_win[:n_train], y_win[:n_train, :n_orig_features])
    mean_path = cache_dir / "y_train_mean.npy"
    std_path = cache_dir / "y_train_std.npy"
    if mean_path.exists() and std_path.exists():
        y_train_mean = np.load(mean_path)
        y_train_std = np.load(std_path)
    else:
        w = int((cache_dir / "window_size.txt").read_text()) if (cache_dir / "window_size.txt").exists() else (window_size or 10)
        _, y_win = _prepare_windows(X_orig, w)
        n_test = X_test_windows.shape[0]
        y_train = y_win[: y_win.shape[0] - n_test, :n_orig_features]
        y_train_mean = np.mean(y_train, axis=0).astype(np.float64)
        y_train_std = np.std(y_train, axis=0).astype(np.float64)
        y_train_std = np.maximum(y_train_std, 1e-8)
        np.save(mean_path, y_train_mean)
        np.save(std_path, y_train_std)
    return X_orig, y_test, X_test_windows, n_orig_features, scale, y_train_mean, y_train_std


def _has_shared_test_data(cache_dir: Path) -> bool:
    return (cache_dir / "y_test.npy").exists() and (
        cache_dir / "X_test_windows.npy"
    ).exists()


def _save_predictions(
    method_dir: Path,
    preds_LSTM: np.ndarray,
    preds_Linear: np.ndarray,
    fitted_LSTM: np.ndarray | None = None,
    fitted_Linear: np.ndarray | None = None,
) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    np.save(method_dir / "predictions_LSTM.npy", preds_LSTM)
    np.save(method_dir / "predictions_Linear.npy", preds_Linear)
    if fitted_LSTM is not None:
        np.save(method_dir / "fitted_LSTM.npy", fitted_LSTM)
    if fitted_Linear is not None:
        np.save(method_dir / "fitted_Linear.npy", fitted_Linear)


def _has_predictions(method_dir: Path) -> bool:
    return (method_dir / "predictions_LSTM.npy").exists() and (
        method_dir / "predictions_Linear.npy"
    ).exists()


def _load_predictions(method_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    preds_LSTM = np.load(method_dir / "predictions_LSTM.npy")
    preds_Linear = np.load(method_dir / "predictions_Linear.npy")
    return preds_LSTM, preds_Linear


def _load_fitted(
    method_dir: Path,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load fitted (in-sample) predictions if present. Backward compatible when missing."""
    fitted_lstm_path = method_dir / "fitted_LSTM.npy"
    fitted_linear_path = method_dir / "fitted_Linear.npy"
    fitted_LSTM = (
        np.load(fitted_lstm_path) if fitted_lstm_path.exists() else None
    )
    fitted_Linear = (
        np.load(fitted_linear_path) if fitted_linear_path.exists() else None
    )
    return fitted_LSTM, fitted_Linear


def _results_from_predictions(
    method_name: str,
    y_test: np.ndarray,
    preds_LSTM: np.ndarray,
    preds_Linear: np.ndarray,
    scale: float,
) -> list[ForecastResult]:
    """Compute ForecastResults from saved predictions (n_runs, n_test, n_out)."""
    mae_lstm = np.mean(np.abs(y_test - preds_LSTM), axis=(1, 2))
    mae_linear = np.mean(np.abs(y_test - preds_Linear), axis=(1, 2))
    mase_lstm = mae_lstm / scale
    mase_linear = mae_linear / scale
    return [
        ForecastResult(
            method=method_name,
            forecaster="LSTM",
            mase_mean=float(np.mean(mase_lstm)),
            mase_std=float(np.std(mase_lstm)),
        ),
        ForecastResult(
            method=method_name,
            forecaster="Linear",
            mase_mean=float(np.mean(mase_linear)),
            mase_std=float(np.std(mase_linear)),
        ),
    ]


def _save_synthetic(method_dir: Path, synthetic: np.ndarray) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    np.save(method_dir / "synthetic.npy", synthetic)


def _has_synthetic(method_dir: Path) -> bool:
    return (method_dir / "synthetic.npy").exists()


def _load_synthetic(method_dir: Path) -> np.ndarray:
    return np.load(method_dir / "synthetic.npy")


def _methods_with_synthetic(
    cache_dir: Path, cfg: ExperimentConfig
) -> list[str]:
    """Return list of known method names that have cached synthetic data."""
    return [
        name
        for name in _known_cache_method_names()
        if _has_synthetic(_method_dir_for(cache_dir, name, cfg))
    ]


def _plot_original_vs_generated(
    cache_dir: Path,
    output_dir: Path,
    cfg: ExperimentConfig,
    n_series: int = 6,
    seed: int = SEED,
) -> None:
    """Plot original vs generated for n_series randomly selected series, one column per model."""
    methods = _methods_with_synthetic(cache_dir, cfg)
    if not methods:
        return
    X_orig, _, _, _, _, _, _ = _load_shared_test_data(cache_dir, cfg.window_size)
    n_timesteps, n_total_series = X_orig.shape
    n_plot = min(n_series, n_total_series)
    rng = np.random.RandomState(seed)
    series_indices: np.ndarray = rng.choice(n_total_series, size=n_plot, replace=False)

    synthetics: dict[str, np.ndarray] = {}
    for name in methods:
        method_d = _method_dir_for(cache_dir, name, cfg)
        synthetics[name] = _load_synthetic(method_d)

    n_cols = len(methods)
    fig, axes = plt.subplots(
        n_plot,
        n_cols,
        figsize=(3 * n_cols, 2.5 * n_plot),
        sharex=True,
        squeeze=False,
    )
    t = np.arange(n_timesteps)
    for row, series_idx in enumerate(series_indices):
        orig_curve = X_orig[:, series_idx]
        for col, method_name in enumerate(methods):
            ax = axes[row, col]
            ax.plot(t, orig_curve, label="Original", color="C0", alpha=0.9)
            ax.plot(
                t,
                synthetics[method_name][:, series_idx],
                label="Generated",
                color="C1",
                alpha=0.9,
            )
            if row == 0:
                title = (
                    f"{method_name} (σ={cfg.lgta_sigma})"
                    if method_name == "LGTA"
                    else method_name
                )
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"Series {series_idx}")
            ax.legend(loc="upper right", fontsize=7)
    for col in range(n_cols):
        axes[-1, col].set_xlabel("Time")
    plt.suptitle("Original vs generated (6 randomly selected series)", y=1.01)
    plt.tight_layout()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_file = plots_dir / "original_vs_generated_6series.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out_file}")


def _plot_predictions_by_method_forecaster(
    cache_dir: Path,
    output_dir: Path,
    cfg: ExperimentConfig,
    n_series: int = 3,
    seed: int = SEED,
) -> None:
    """Plot actual, fitted (train), and predictions (test): rows = method, columns = LSTM / Linear.

    Targets and predictions are in the same units (no scaling/unscaling in this pipeline).
    """
    method_names = [
        name
        for name in _known_cache_method_names()
        if _has_predictions(_method_dir_for(cache_dir, name, cfg))
    ]
    if not method_names:
        return
    X_orig, y_test, _, n_orig_features, _, _, _ = _load_shared_test_data(
        cache_dir, cfg.window_size
    )
    _, y_win = _prepare_windows(X_orig, cfg.window_size)
    n_test = y_test.shape[0]
    n_train = y_win.shape[0] - n_test
    y_train = y_win[:n_train]

    n_series_avail = min(n_series, n_orig_features)
    rng = np.random.RandomState(seed)
    series_indices: np.ndarray = rng.choice(
        n_orig_features, size=n_series_avail, replace=False
    )

    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fitted: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}
    for name in method_names:
        method_d = _method_dir_for(cache_dir, name, cfg)
        predictions[name] = _load_predictions(method_d)
        fitted[name] = _load_fitted(method_d)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(method_names)
    n_cols = 2
    t_full = np.arange(n_train + n_test)

    for plot_idx, series_idx in enumerate(series_indices):
        actual_full_s = np.concatenate(
            [y_train[:, series_idx], y_test[:, series_idx]], axis=0
        )
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4 * n_cols, 2.2 * n_rows),
            sharex=True,
            squeeze=False,
        )
        for row, method_name in enumerate(method_names):
            pred_lstm, pred_linear = predictions[method_name]
            f_lstm, f_linear = fitted[method_name]
            lstm_pred_mean = pred_lstm.mean(axis=0)[:, series_idx]
            linear_pred_mean = pred_linear.mean(axis=0)[:, series_idx]
            ax_lstm = axes[row, 0]
            ax_linear = axes[row, 1]
            ax_lstm.plot(
                t_full, actual_full_s, label="Actual", color="C0", alpha=0.9
            )
            if f_lstm is not None:
                lstm_fit_mean = f_lstm.mean(axis=0)[:n_train, series_idx]
                ax_lstm.plot(
                    np.arange(n_train),
                    lstm_fit_mean,
                    label="Fitted (train)",
                    color="C1",
                    alpha=0.9,
                    linestyle="--",
                )
            ax_lstm.plot(
                np.arange(n_train, n_train + n_test),
                lstm_pred_mean,
                label="Pred (test)",
                color="C2",
                alpha=0.9,
                linestyle="-.",
            )
            ax_linear.plot(
                t_full, actual_full_s, label="Actual", color="C0", alpha=0.9
            )
            if f_linear is not None:
                linear_fit_mean = f_linear.mean(axis=0)[:n_train, series_idx]
                ax_linear.plot(
                    np.arange(n_train),
                    linear_fit_mean,
                    label="Fitted (train)",
                    color="C1",
                    alpha=0.9,
                    linestyle="--",
                )
            ax_linear.plot(
                np.arange(n_train, n_train + n_test),
                linear_pred_mean,
                label="Pred (test)",
                color="C2",
                alpha=0.9,
                linestyle="-.",
            )
            if row == 0:
                ax_lstm.set_title("LSTM")
                ax_linear.set_title("Linear")
            ax_lstm.set_ylabel(method_name, fontsize=9)
            ax_lstm.legend(loc="upper right", fontsize=7)
            ax_linear.legend(loc="upper right", fontsize=7)
        axes[-1, 0].set_xlabel("Time step")
        axes[-1, 1].set_xlabel("Time step")
        plt.suptitle(
            f"Actual, fitted & predictions — series {series_idx} ({cfg.dataset_name})",
            y=1.01,
        )
        plt.tight_layout()
        out_file = plots_dir / f"predictions_by_method_series_{plot_idx}.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {out_file}")


# ---------------------------------------------------------------------------
# Data loading and LGTA
# ---------------------------------------------------------------------------


def _load_original_data(cfg: ExperimentConfig) -> np.ndarray:
    """Load the (n_timesteps, n_series) data matrix without training LGTA."""
    from lgta.preprocessing.pre_processing_datasets import PreprocessDatasets

    ppc = PreprocessDatasets(dataset=cfg.dataset_name, freq=cfg.freq)
    data = ppc.apply_preprocess()
    return data["predict"]["data_matrix"].astype(np.float32)


def _generate_lgta(cfg: ExperimentConfig, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    """Train LGTA CVAE and generate one augmented dataset.

    Returns (X_orig, X_lgta) both of shape (n_timesteps, n_series).
    Model weights are stored under cache/model_weights.
    """
    from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
    from lgta.model.generate_data import generate_synthetic_data
    from lgta.model.models import LatentMode

    eq_str = str(cfg.lgta_equiv_weight).replace(".", "_")
    weights_suffix_parts = [f"eq{eq_str}"]
    if cfg.lgta_sample_from_posterior:
        weights_suffix_parts.append("posterior")
    weights_suffix = "_".join(weights_suffix_parts)

    creator = CreateTransformedVersionsCVAE(
        dataset_name=cfg.dataset_name,
        freq=cfg.freq,
        window_size=cfg.window_size,
        weights_suffix=weights_suffix,
        weights_dir=cache / "model_weights",
    )
    model, _, _ = creator.fit(
        epochs=cfg.lgta_epochs,
        latent_dim=cfg.lgta_latent_dim,
        equiv_weight=cfg.lgta_equiv_weight,
        latent_mode=LatentMode.TEMPORAL,
    )
    _, _, z_mean, z_log_var = creator.predict(model)
    X_orig = creator.X_train_raw

    print(
        f"  LGTA generation: transformation={cfg.lgta_transformation!r}, "
        f"sigma={cfg.lgta_sigma} (applied in latent space)"
    )
    rng = np.random.default_rng(SEED)
    X_lgta = generate_synthetic_data(
        model,
        z_mean,
        creator,
        cfg.lgta_transformation,
        [cfg.lgta_sigma],
        latent_mode=LatentMode.TEMPORAL,
        z_log_var=z_log_var if cfg.lgta_sample_from_posterior else None,
        sample_from_posterior=cfg.lgta_sample_from_posterior,
        rng=rng,
    )
    return X_orig, X_lgta


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

_BENCHMARK_NAME_ALIASES: dict[str, str] = {
    "timegan": "TimeGANGenerator",
    "timevae": "TimeVAEGenerator",
    "diffusion_ts": "DiffusionTSGenerator",
    "diffusiontsgenerator": "DiffusionTSGenerator",
    "direct": "DirectTransformGenerator",
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
    are loaded from cache and results are computed and written.

    Cache: per-dataset folder assets/cache/downstream_forecasting/<dataset_name>/
    containing shared test data, model_weights/, and per-method subdirs.
    Results (DOWNSTREAM_RESULTS.md, DOWNSTREAM_RESULTS_LSTM.md, DOWNSTREAM_RESULTS_Linear.md, and plots/) are written to
    output_dir / <lgta_config_slug>, so each dataset and LGTA configuration gets its own folder.
    """
    if cfg is None:
        cfg = ExperimentConfig()

    cache = _cache_dir(cfg)  # single cache root, no per-config subfolders

    if results_only:
        return _run_results_only(cfg, cache)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    single = method is not None
    if single:
        method_canonical = _resolve_method_arg(method)

    if not cfg.benchmark_generators:
        cfg.benchmark_generators = get_default_benchmark_generators(seed=SEED)

    print("=" * 70)
    print("DOWNSTREAM FORECASTING EXPERIMENT")
    if single:
        print(f"  (single method: {method_canonical})")
    print("=" * 70)

    if single and method_canonical not in ("original", "lgta"):
        print("\n[1/4] Loading data ...")
        X_orig = _load_original_data(cfg)
        X_lgta = None
    else:
        print("\n[1/4] Training LGTA and generating synthetic data ...")
        X_orig, X_lgta = _generate_lgta(cfg, cache)

    n_orig_features = X_orig.shape[1]
    X_orig_win, y_orig_win = _prepare_windows(X_orig, cfg.window_size)
    n_test = max(1, int(0.15 * X_orig_win.shape[0]))
    n_train = X_orig_win.shape[0] - n_test
    X_test_windows = X_orig_win[n_train:]
    y_test = y_orig_win[n_train:]

    if not _has_shared_test_data(cache):
        scale = _mase_scale(X_orig_win[:n_train], y_orig_win[:n_train, :n_orig_features])
        _save_shared_test_data(
            cache, X_orig, y_test, X_test_windows, n_orig_features, scale, cfg.window_size
        )
    else:
        _, _, _, _, scale, _, _ = _load_shared_test_data(cache, cfg.window_size)

    all_results: list[ForecastResult] = []

    run_original = not single or method_canonical == "original"
    if run_original:
        orig_dir = _method_dir(cache, "Original")
        if _has_predictions(orig_dir):
            print("\n[2/4] Original: loading from cache ...")
            _, y_test, _, _, scale, _, _ = _load_shared_test_data(
                cache, cfg.window_size
            )
            p_lstm, p_lin = _load_predictions(orig_dir)
            all_results.extend(
                _results_from_predictions("Original", y_test, p_lstm, p_lin, scale)
            )
        else:
            print("\n[2/4] Evaluating Original (no augmentation) ...")
            _, _, _, _, scale, y_mean, y_std = _load_shared_test_data(
                cache, cfg.window_size
            )
            results, p_lstm, p_lin, f_lstm, f_lin = _evaluate_method(
                "Original",
                X_orig,
                n_orig_features,
                cfg.window_size,
                X_test_windows,
                y_test,
                scale,
                cfg,
                y_mean,
                y_std,
            )
            all_results.extend(results)
            _save_predictions(orig_dir, p_lstm, p_lin, f_lstm, f_lin)

    run_lgta = (not single or method_canonical == "lgta") and X_lgta is not None
    if run_lgta:
        lgta_dir = _lgta_method_dir(cache, cfg)
        if _has_predictions(lgta_dir):
            print("\n[3/4] LGTA: loading from cache ...")
            _, y_test, _, _, scale, _, _ = _load_shared_test_data(
                cache, cfg.window_size
            )
            p_lstm, p_lin = _load_predictions(lgta_dir)
            all_results.extend(
                _results_from_predictions("LGTA", y_test, p_lstm, p_lin, scale)
            )
        else:
            print("\n[3/4] Evaluating LGTA ...")
            _, _, _, _, scale, y_mean, y_std = _load_shared_test_data(
                cache, cfg.window_size
            )
            X_lgta_aug = np.concatenate([X_orig, X_lgta], axis=1)
            results, p_lstm, p_lin, f_lstm, f_lin = _evaluate_method(
                "LGTA",
                X_lgta_aug,
                n_orig_features,
                cfg.window_size,
                X_test_windows,
                y_test,
                scale,
                cfg,
                y_mean,
                y_std,
            )
            all_results.extend(results)
            _save_predictions(lgta_dir, p_lstm, p_lin, f_lstm, f_lin)
            _save_synthetic(lgta_dir, X_lgta)

    benchmarks_to_run = cfg.benchmark_generators
    if single and method_canonical not in ("original", "lgta"):
        benchmarks_to_run = [
            g for g in cfg.benchmark_generators if _benchmark_matches(g, method)
        ]
        if not benchmarks_to_run:
            raise ValueError(
                f"Unknown or unavailable method: {method}. "
                f"Choose from: original, lgta, timegan, timevae, direct."
            )

    if benchmarks_to_run:
        print(f"\n[4/4] Evaluating {len(benchmarks_to_run)} benchmark method(s) ...")
        _, y_test, _, _, scale, y_mean, y_std = _load_shared_test_data(
            cache, cfg.window_size
        )
        for gen in benchmarks_to_run:
            name = gen.name
            method_d = _method_dir(cache, name)
            if _has_predictions(method_d):
                print(f"  {name}: loading from cache ...")
                p_lstm, p_lin = _load_predictions(method_d)
                all_results.extend(
                    _results_from_predictions(name, y_test, p_lstm, p_lin, scale)
                )
                continue
            X_synth: np.ndarray
            if _has_synthetic(method_d):
                print(
                    f"  {name}: loading synthetic from cache, training forecasters ..."
                )
                X_synth = _load_synthetic(method_d)
            else:
                print(f"  Training {name} ...")
                gen.fit(X_orig)
                X_synth = gen.generate()
                X_synth = np.clip(X_synth, a_min=0, a_max=None)
                _save_synthetic(method_d, X_synth)
            X_aug = np.concatenate([X_orig, X_synth], axis=1)
            results, p_lstm, p_lin, f_lstm, f_lin = _evaluate_method(
                name,
                X_aug,
                n_orig_features,
                cfg.window_size,
                X_test_windows,
                y_test,
                scale,
                cfg,
                y_mean,
                y_std,
            )
            all_results.extend(results)
            _save_predictions(method_d, p_lstm, p_lin, f_lstm, f_lin)

    _print_results(all_results)
    effective_out = cfg.output_dir / _lgta_config_slug(cfg)
    print(f"\nResults written to: {effective_out}")
    _save_results(all_results, effective_out)
    _plot_original_vs_generated(cache, effective_out, cfg, n_series=6, seed=SEED)
    _plot_predictions_by_method_forecaster(
        cache, effective_out, cfg, n_series=3, seed=SEED
    )
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
    _, y_test, _, _, scale, _, _ = _load_shared_test_data(cache, cfg.window_size)
    all_results: list[ForecastResult] = []
    for method_name in _known_cache_method_names():
        method_dir = _method_dir_for(cache, method_name, cfg)
        if _has_predictions(method_dir):
            p_lstm, p_lin = _load_predictions(method_dir)
            all_results.extend(
                _results_from_predictions(method_name, y_test, p_lstm, p_lin, scale)
            )
    if not all_results:
        raise FileNotFoundError(
            f"No cached predictions found under {cache}. Run at least one method first."
        )
    _print_results(all_results)
    effective_out = cfg.output_dir / _lgta_config_slug(cfg)
    print(f"\nResults written to: {effective_out}")
    _save_results(all_results, effective_out)
    _plot_original_vs_generated(cache, effective_out, cfg, n_series=6, seed=SEED)
    _plot_predictions_by_method_forecaster(
        cache, effective_out, cfg, n_series=3, seed=SEED
    )
    return all_results


def _print_results(results: list[ForecastResult]) -> None:
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    header = (
        f"{'Method':<25s}  {'Forecaster':<10s}  {'MASE (mean)':>12s}  {'± std':>10s}"
    )
    print(header)
    print("-" * 70)
    for r in results:
        print(
            f"{r.method:<25s}  {r.forecaster:<10s}  {r.mase_mean:12.4f}  {r.mase_std:10.4f}"
        )
    print("=" * 70)


def _save_results(results: list[ForecastResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Downstream Forecasting Results\n"]
    lines.append("| Method | Forecaster | MASE (mean) | ± std |")
    lines.append("|--------|------------|-------------|-------|")
    for r in results:
        lines.append(
            f"| {r.method} | {r.forecaster} | {r.mase_mean:.4f} | {r.mase_std:.4f} |"
        )
    out = output_dir / "DOWNSTREAM_RESULTS.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nResults saved to {out}")
    _save_results_by_forecaster(results, output_dir)


def _save_results_by_forecaster(
    results: list[ForecastResult], output_dir: Path
) -> None:
    """Write one markdown file per forecaster (LSTM, Linear) with methods ordered by % change vs Original (ascending)."""
    for forecaster in ("LSTM", "Linear"):
        subset = [r for r in results if r.forecaster == forecaster]
        if not subset:
            continue
        original = next((r for r in subset if r.method == "Original"), None)
        lines = [f"# Downstream Forecasting Results — {forecaster}\n"]
        if original is not None:
            original_mase = original.mase_mean
            with_pct = [
                (r, (r.mase_mean - original_mase) / original_mase * 100.0)
                for r in subset
            ]
            ordered = sorted(with_pct, key=lambda x: x[1])
            lines.append("| Method | MASE (mean) | ± std | % change vs Original |")
            lines.append("|--------|-------------|-------|----------------------|")
            for r, pct in ordered:
                lines.append(
                    f"| {r.method} | {r.mase_mean:.4f} | {r.mase_std:.4f} | {pct:+.2f}% |"
                )
        else:
            ordered = sorted(subset, key=lambda r: r.mase_mean)
            lines.append("| Method | MASE (mean) | ± std |")
            lines.append("|--------|-------------|-------|")
            for r in ordered:
                lines.append(f"| {r.method} | {r.mase_mean:.4f} | {r.mase_std:.4f} |")
        out = output_dir / f"DOWNSTREAM_RESULTS_{forecaster}.md"
        out.write_text("\n".join(lines) + "\n")
        print(f"Results by forecaster saved to {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run downstream forecasting experiment (optionally a single method)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g. tourism_small, tourism, wiki2, labour, m3). Default: tourism_small.",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default=None,
        help="Time series frequency (e.g. Q, D, M). Default: Q for tourism, D for wiki2, M for labour, Q for m3.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run the experiment for all supported datasets (tourism_small, tourism, wiki2, labour, m3) with their default frequencies.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Run only this method: original, lgta, timegan, timevae, direct (or class name e.g. TimeGANGenerator).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/results/downstream_forecasting"),
        help="Base directory for results; actual output is written to a subfolder named by LGTA config (dataset, freq, window, transformation, sigma, epochs, latent_dim).",
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Do not run training/inference; load cached predictions and write DOWNSTREAM_RESULTS.md for all available methods.",
    )
    parser.add_argument(
        "--lgta-sample-from-posterior",
        action="store_true",
        help="Sample z from the CVAE posterior N(z_mean, z_std) before applying the transformation (increases diversity).",
    )
    args = parser.parse_args()

    if args.all_datasets:
        for i, (dataset_name, freq) in enumerate(DEFAULT_DATASET_CONFIGS):
            print(f"\n{'='*70}")
            print(f"Dataset {i+1}/{len(DEFAULT_DATASET_CONFIGS)}: {dataset_name} (freq={freq})")
            print("="*70)
            cfg = ExperimentConfig(
                dataset_name=dataset_name,
                freq=freq,
                output_dir=args.output_dir,
                lgta_sample_from_posterior=args.lgta_sample_from_posterior,
            )
            run_downstream_forecasting(
                cfg,
                method=args.method,
                results_only=args.results_only,
            )
    else:
        dataset_name = args.dataset if args.dataset is not None else "tourism_small"
        freq = args.freq
        if freq is None:
            freq = next(
                (f for d, f in DEFAULT_DATASET_CONFIGS if d == dataset_name),
                "Q",
            )
        cfg = ExperimentConfig(
            dataset_name=dataset_name,
            freq=freq,
            output_dir=args.output_dir,
            lgta_sample_from_posterior=args.lgta_sample_from_posterior,
        )
        run_downstream_forecasting(
            cfg,
            method=args.method,
            results_only=args.results_only,
        )
