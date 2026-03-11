"""
Downstream forecasting experiment with two evaluation modes:

  TSTR  — Train on Synthetic, Test on Real.  Forecasters are trained
          exclusively on synthetic windows and evaluated on real data.
  downstream_task — Train on Original + Synthetic, Test on Real.
          Synthetic variants are stacked with the original data so the
          forecaster sees both real and generated windows.

Compares LGTA against benchmark generators.  The "Original" baseline
trains and tests on real data.  Select the mode with ``--eval-mode``.
Dynamic time features can be disabled with ``--no-dynamic-features``.
When ``--all-datasets`` is used, both with- and without-dynamic runs
are performed unless ``--no-dynamic-features`` is explicitly passed.

Can be invoked directly (python lgta/experiments/downstream_forecasting.py)
or as a module (python -m lgta.experiments.downstream_forecasting) provided
the repo root is on PYTHONPATH or you run from the repo root.
"""

import gc
import json
import re
import sys
import time
from enum import Enum
from pathlib import Path

import resource

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataclasses import dataclass, field


class EvalMode(Enum):
    """Training strategy for downstream evaluation.

    TSTR — Train on Synthetic, Test on Real (pure TSTR).
    DOWNSTREAM — Train on Original + Synthetic, Test on Real (augmentation).
    """

    TSTR = "TSTR"
    DOWNSTREAM = "downstream_task"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lgta.benchmarks import (
    DirectTransformGenerator,
    TimeSeriesGenerator,
    get_default_benchmark_generators,
)


SEED = 42

DEFAULT_DATASET_CONFIGS: list[tuple[str, str]] = [
    ("tourism", "Q"),
    ("wiki2", "D"),
    ("labour", "M"),
    ("m3", "Y"),
    ("m3", "Q"),
    ("m3", "M"),
    # ("m4", "W"),
    # ("m4", "H"),
]


@dataclass
class ForecastResult:
    """Stores MASE statistics for one augmentation method + one forecaster."""

    method: str
    forecaster: str
    mase_mean: float
    mase_std: float


@dataclass
class ResourceUsage:
    """Stores wall-clock time and peak memory for one augmentation method's generation."""

    method: str
    time_seconds: float
    memory_mb: float


def _current_rss_mb() -> float:
    """Return current process peak resident set size in MB (bytes on macOS, KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


@dataclass
class ExperimentConfig:
    """Top-level knobs for the downstream forecasting experiment."""

    dataset_name: str = "tourism"
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
    variant_transformations: list[str] = field(default_factory=list)
    benchmark_generators: list[TimeSeriesGenerator] = field(default_factory=list)
    output_dir: Path = Path("assets/results/downstream_forecasting")
    eval_mode: EvalMode = EvalMode.TSTR
    use_dynamic_features: bool = True

    @property
    def dynamic_subdir(self) -> str:
        return "with_dynamic" if self.use_dynamic_features else "without_dynamic"

    @property
    def effective_transformations(self) -> list[str]:
        return self.variant_transformations if self.variant_transformations else [self.lgta_transformation]

    @property
    def n_variants(self) -> int:
        return len(self.effective_transformations)

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
            "variant_transformations": self.variant_transformations,
            "use_dynamic_features": self.use_dynamic_features,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_device() -> torch.device:
    """Device for the downstream forecasting models.

    Forces CPU because the data and models are tiny (hundreds of samples,
    small LSTM).  MPS has memory-pool fragmentation that leaks tens
    of GBs on repeated alloc/dealloc cycles; CUDA would be fine but is not
    needed for this scale.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _release_memory() -> None:
    """Run GC and clear device caches to reduce OOM risk between heavy steps."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _prepare_windows(
    data: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _y_train_mask_from_valid_mask(
    valid_mask: np.ndarray | None,
    window_size: int,
    n_train: int,
    n_orig_features: int,
) -> np.ndarray | None:
    """Mask for y_train windows: True where the predicted timestep was observed. Shape (n_train, n_orig_features)."""
    if valid_mask is None:
        return None
    return valid_mask[window_size : window_size + n_train, :n_orig_features].copy()


def _y_test_mask_from_valid_mask(
    valid_mask: np.ndarray | None,
    window_size: int,
    n_train: int,
    n_test: int,
    n_orig_features: int,
) -> np.ndarray | None:
    """Mask for y_test windows: True where the predicted timestep was observed. Shape (n_test, n_orig_features)."""
    if valid_mask is None:
        return None
    return valid_mask[
        window_size + n_train : window_size + n_train + n_test, :n_orig_features
    ].copy()


def _mase_scale(
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_train_mask: np.ndarray | None = None,
) -> float:
    """In-sample MAE of naive (persistence) forecast. Used as MASE denominator.
    If y_train_mask is provided (True = observed), scale is computed only over valid positions.
    """
    n_out = y_train.shape[1]
    naive = X_train[:, -1, :n_out].astype(np.float64)
    y = y_train.astype(np.float64)
    diff = np.abs(y - naive)
    if y_train_mask is not None:
        n_valid = np.sum(y_train_mask)
        if n_valid == 0:
            return 1.0
        scale = float(np.sum(diff * y_train_mask) / n_valid)
    else:
        scale = float(np.mean(diff))
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
    y_train_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_dev = torch.from_numpy(X_train).to(device)
    y_dev = torch.from_numpy(y_train).to(device)
    mask_dev = (
        torch.from_numpy(y_train_mask.astype(np.float32)).to(device)
        if y_train_mask is not None
        else None
    )
    train_tensors = [X_dev, y_dev] + ([mask_dev] if mask_dev is not None else [])
    ds = TensorDataset(*train_tensors)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch in loader:
            bx, by = batch[0], batch[1]
            bm = batch[2] if len(batch) > 2 else None
            optimizer.zero_grad()
            pred = model(bx)
            if bm is not None:
                loss = ((pred - by) ** 2 * bm).sum() / bm.sum().clamp(min=1)
            else:
                loss = nn.functional.mse_loss(pred, by)
            loss.backward()
            optimizer.step()

    del ds, loader, optimizer, mask_dev

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
        fitted = model(X_dev).cpu().numpy()
    del X_dev, y_dev
    _release_memory()

    mae = float(np.mean(np.abs(y_test.astype(np.float64) - preds.astype(np.float64))))
    mase = mae / scale
    return mase, preds.astype(np.float32), fitted.astype(np.float32)


def _run_single(
    n_features_in: int,
    n_features_out: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale: float,
    cfg: ExperimentConfig,
    y_train_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    device = _get_device()
    model = _LSTM(n_features_in, n_features_out).to(device)
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
        y_train_mask,
    )
    del model
    return mase, preds, fitted


def _evaluate_method(
    method_name: str,
    variants: list[np.ndarray],
    n_orig_features: int,
    window_size: int,
    n_train: int,
    X_test_windows: np.ndarray,
    y_test: np.ndarray,
    scale: float,
    cfg: ExperimentConfig,
    y_train_mean: np.ndarray | None = None,
    y_train_std: np.ndarray | None = None,
    y_test_mask: np.ndarray | None = None,
    y_train_mask: np.ndarray | None = None,
) -> tuple[list[ForecastResult], np.ndarray, np.ndarray]:
    """Proper TSTR: train on synthetic windows, test on purely real windows.

    Each variant is windowed independently and training windows are stacked
    along axis 0.  The model architecture (n_orig_features input) and the
    test data (purely real) are identical across all methods.

    preds_LSTM has shape (n_runs, n_test, n_orig_features).
    fitted_LSTM has shape (n_runs, n_stacked_train, n_orig_features).
    If y_train_mean and y_train_std are provided, targets are scaled before
    training and predictions/fitted are unscaled before return.
    If y_test_mask is provided (True = observed), MASE is computed only over
    valid positions.
    """
    X_train_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []
    for variant in variants:
        X_win, y_win = _prepare_windows(variant, window_size)
        X_train_parts.append(X_win[:n_train].copy())
        y_train_parts.append(y_win[:n_train, :n_orig_features].copy())
        del X_win, y_win

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    del X_train_parts, y_train_parts

    X_test_eval = np.asarray(X_test_windows, dtype=np.float32).copy()
    y_test_f = y_test.astype(np.float32)

    if y_train_mask is not None:
        y_train_mask = np.tile(y_train_mask, (len(variants), 1))

    use_scale = y_train_mean is not None and y_train_std is not None
    if use_scale:
        mean_f32 = y_train_mean.astype(np.float32)
        std_f32 = y_train_std.astype(np.float32)
        y_train = (y_train - mean_f32) / std_f32
        y_test_f = (y_test_f - mean_f32) / std_f32
        X_train[:, :, :n_orig_features] = (
            X_train[:, :, :n_orig_features] - mean_f32
        ) / std_f32
        X_test_eval[:, :, :n_orig_features] = (
            X_test_eval[:, :, :n_orig_features] - mean_f32
        ) / std_f32

    preds_lstm_list: list[np.ndarray] = []
    fitted_lstm_list: list[np.ndarray] = []
    for _ in range(cfg.n_runs):
        mase, preds, fitted = _run_single(
            n_orig_features,
            n_orig_features,
            X_train,
            y_train,
            X_test_eval,
            y_test_f,
            scale,
            cfg,
            y_train_mask,
        )
        preds_lstm_list.append(preds)
        fitted_lstm_list.append(fitted)
    preds_LSTM = np.stack(preds_lstm_list, axis=0)
    fitted_LSTM = np.stack(fitted_lstm_list, axis=0)
    if use_scale:
        preds_LSTM = preds_LSTM * std_f32 + mean_f32
        fitted_LSTM = fitted_LSTM * std_f32 + mean_f32
    results = _results_from_predictions(
        method_name, y_test, preds_LSTM, scale, y_test_mask
    )
    return results, preds_LSTM, fitted_LSTM


# ---------------------------------------------------------------------------
# Cache (config id, save/load shared and per-method data)
# ---------------------------------------------------------------------------

CACHE_ROOT = Path("assets/cache/downstream_forecasting")


def _lgta_config_slug(cfg: ExperimentConfig) -> str:
    """Filesystem-safe folder name from LGTA-related config."""
    sigma_str = str(cfg.lgta_sigma).replace(".", "_")
    if cfg.n_variants > 1:
        transf_str = "+".join(cfg.effective_transformations)
        parts: list[str | float] = [
            cfg.dataset_name,
            cfg.freq,
            f"w{cfg.window_size}",
            f"{cfg.n_variants}var",
            transf_str,
            f"sig{sigma_str}",
            f"ep{cfg.lgta_epochs}",
            f"lat{cfg.lgta_latent_dim}",
            f"eq{cfg.lgta_equiv_weight}",
        ]
    else:
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
    if not cfg.use_dynamic_features:
        parts.append("nodyn")
    return "_".join(str(p) for p in parts)


def _cache_dir(cfg: ExperimentConfig) -> Path:
    """Cache root per dataset+freq so weights, test data, and predictions do not collide across datasets or frequencies."""
    return CACHE_ROOT / f"{cfg.dataset_name}_{cfg.freq}"


def _method_dir(cache_dir: Path, method_name: str, n_variants: int = 1) -> Path:
    suffix = f"_{n_variants}var" if n_variants > 1 else ""
    return cache_dir / (method_name + suffix).replace(" ", "_")


def _lgta_method_dir(cache_dir: Path, cfg: ExperimentConfig) -> Path:
    """LGTA cache dir keyed by config so different sigma/transformation/latent_dim get separate cache."""
    return cache_dir / ("LGTA_" + _lgta_config_slug(cfg))


def _method_dir_for(
    cache_dir: Path, method_name: str, cfg: ExperimentConfig
) -> Path:
    """Resolve method name to cache dir; LGTA uses config-keyed subdir."""
    if method_name == "LGTA":
        return _lgta_method_dir(cache_dir, cfg)
    if method_name == "Original":
        return _method_dir(cache_dir, method_name, n_variants=1)
    return _method_dir(cache_dir, method_name, n_variants=cfg.n_variants)


def _pred_dir(method_dir: Path, eval_mode: EvalMode) -> Path:
    """Prediction cache subdirectory, keyed by evaluation mode.

    Synthetic data lives directly in *method_dir* (shared across modes);
    predictions and fitted arrays live inside the mode-specific subdir.
    """
    return method_dir / eval_mode.value


def _benchmark_display_name(gen: TimeSeriesGenerator, n_variants: int) -> str:
    """Display name for a benchmark generator, accounting for multi-variant Direct."""
    if n_variants > 1 and isinstance(gen, DirectTransformGenerator):
        return "Direct"
    return gen.name


def _known_cache_method_names(n_variants: int = 1) -> list[str]:
    """Method names we may have in cache: Original, LGTA, and default benchmark names."""
    return ["Original", "LGTA"] + [
        _benchmark_display_name(g, n_variants)
        for g in get_default_benchmark_generators(seed=SEED)
    ]


def _save_shared_test_data(
    cache_dir: Path,
    X_orig: np.ndarray,
    y_test: np.ndarray,
    X_test_windows: np.ndarray,
    n_orig_features: int,
    scale: float,
    window_size: int,
    valid_mask: np.ndarray | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "X_orig.npy", X_orig)
    np.save(cache_dir / "y_test.npy", y_test)
    np.save(cache_dir / "X_test_windows.npy", X_test_windows)
    np.save(cache_dir / "scale.npy", np.array(scale, dtype=np.float64))
    (cache_dir / "n_orig_features.txt").write_text(str(n_orig_features))
    (cache_dir / "window_size.txt").write_text(str(window_size))
    if valid_mask is not None:
        np.save(cache_dir / "valid_mask.npy", valid_mask.astype(np.uint8))
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    X_orig = np.load(cache_dir / "X_orig.npy")
    y_test = np.load(cache_dir / "y_test.npy")
    X_test_windows = np.load(cache_dir / "X_test_windows.npy")
    n_orig_features = int((cache_dir / "n_orig_features.txt").read_text())
    valid_mask_path = cache_dir / "valid_mask.npy"
    valid_mask = np.load(valid_mask_path).astype(bool) if valid_mask_path.exists() else None
    scale_path = cache_dir / "scale.npy"
    if scale_path.exists():
        scale = float(np.load(scale_path))
    else:
        w = int((cache_dir / "window_size.txt").read_text()) if (cache_dir / "window_size.txt").exists() else (window_size or 10)
        X_win, y_win = _prepare_windows(X_orig, w)
        n_test = X_test_windows.shape[0]
        n_train = X_win.shape[0] - n_test
        y_train_mask = _y_train_mask_from_valid_mask(valid_mask, w, n_train, n_orig_features)
        scale = _mase_scale(X_win[:n_train], y_win[:n_train, :n_orig_features], y_train_mask)
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
    return X_orig, y_test, X_test_windows, n_orig_features, scale, y_train_mean, y_train_std, valid_mask


def _has_shared_test_data(cache_dir: Path) -> bool:
    return (cache_dir / "y_test.npy").exists() and (
        cache_dir / "X_test_windows.npy"
    ).exists()


def _save_predictions(
    method_dir: Path,
    preds_LSTM: np.ndarray,
    fitted_LSTM: np.ndarray | None = None,
) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    np.save(method_dir / "predictions_LSTM.npy", preds_LSTM)
    if fitted_LSTM is not None:
        np.save(method_dir / "fitted_LSTM.npy", fitted_LSTM)


def _has_predictions(method_dir: Path) -> bool:
    return (method_dir / "predictions_LSTM.npy").exists()


def _load_predictions(method_dir: Path) -> np.ndarray:
    return np.load(method_dir / "predictions_LSTM.npy")


def _load_fitted(method_dir: Path) -> np.ndarray | None:
    """Load fitted (in-sample) predictions if present. Backward compatible when missing."""
    fitted_path = method_dir / "fitted_LSTM.npy"
    return np.load(fitted_path) if fitted_path.exists() else None


def _results_from_predictions(
    method_name: str,
    y_test: np.ndarray,
    preds_LSTM: np.ndarray,
    scale: float,
    y_test_mask: np.ndarray | None = None,
) -> list[ForecastResult]:
    """Compute ForecastResults from saved LSTM predictions (n_runs, n_test, n_out).
    If y_test_mask is provided (True = observed), MAE/MASE are computed only over valid positions.
    """
    abs_lstm = np.abs(y_test - preds_LSTM)
    if y_test_mask is not None:
        n_valid = np.sum(y_test_mask)
        mae_lstm = (
            np.full(abs_lstm.shape[0], np.nan)
            if n_valid == 0
            else np.sum(abs_lstm * y_test_mask, axis=(1, 2)) / n_valid
        )
    else:
        mae_lstm = np.mean(abs_lstm, axis=(1, 2))
    mase_lstm = mae_lstm / scale
    return [
        ForecastResult(
            method=method_name,
            forecaster="LSTM",
            mase_mean=float(np.mean(mase_lstm)),
            mase_std=float(np.std(mase_lstm)),
        ),
    ]


def _save_synthetic(method_dir: Path, synthetic: np.ndarray) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    np.save(method_dir / "synthetic.npy", synthetic)


def _has_synthetic(method_dir: Path) -> bool:
    return (method_dir / "synthetic.npy").exists()


def _load_synthetic(method_dir: Path) -> np.ndarray:
    return np.load(method_dir / "synthetic.npy")


def _save_synthetic_variants(method_dir: Path, variants: list[np.ndarray]) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    if len(variants) == 1:
        np.save(method_dir / "synthetic.npy", variants[0])
    else:
        for i, v in enumerate(variants):
            np.save(method_dir / f"synthetic_v{i}.npy", v)


def _has_synthetic_variants(method_dir: Path, n_variants: int) -> bool:
    if n_variants == 1:
        return _has_synthetic(method_dir)
    return all(
        (method_dir / f"synthetic_v{i}.npy").exists() for i in range(n_variants)
    )


def _load_synthetic_variants(method_dir: Path, n_variants: int) -> list[np.ndarray]:
    if n_variants == 1:
        return [_load_synthetic(method_dir)]
    return [np.load(method_dir / f"synthetic_v{i}.npy") for i in range(n_variants)]


def _methods_with_synthetic(
    cache_dir: Path, cfg: ExperimentConfig
) -> list[str]:
    """Return list of known method names that have cached synthetic data."""
    return [
        name
        for name in _known_cache_method_names(cfg.n_variants)
        if _has_synthetic_variants(
            _method_dir_for(cache_dir, name, cfg), cfg.n_variants
        )
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
    X_orig, _, _, _, _, _, _, _ = _load_shared_test_data(cache_dir, cfg.window_size)
    n_timesteps, n_total_series = X_orig.shape
    n_plot = min(n_series, n_total_series)
    rng = np.random.RandomState(seed)
    series_indices: np.ndarray = rng.choice(n_total_series, size=n_plot, replace=False)

    synthetics: dict[str, np.ndarray] = {}
    for name in methods:
        method_d = _method_dir_for(cache_dir, name, cfg)
        synthetics[name] = _load_synthetic_variants(method_d, cfg.n_variants)[0]

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
    """Plot actual, fitted (train), and LSTM predictions (test): one row per method.

    Targets and predictions are in the same units (no scaling/unscaling in this pipeline).
    """
    method_names = [
        name
        for name in _known_cache_method_names(cfg.n_variants)
        if _has_predictions(_pred_dir(_method_dir_for(cache_dir, name, cfg), cfg.eval_mode))
    ]
    if not method_names:
        return
    X_orig, y_test, _, n_orig_features, _, _, _, valid_mask = _load_shared_test_data(
        cache_dir, cfg.window_size
    )
    _, y_win = _prepare_windows(X_orig, cfg.window_size)
    n_test = y_test.shape[0]
    n_train = y_win.shape[0] - n_test
    y_train = y_win[:n_train].copy().astype(np.float64)

    y_train_vmask: np.ndarray | None = None
    y_test_vmask: np.ndarray | None = None
    if valid_mask is not None:
        y_train_vmask = _y_train_mask_from_valid_mask(
            valid_mask, cfg.window_size, n_train, n_orig_features
        )
        y_test_vmask = _y_test_mask_from_valid_mask(
            valid_mask, cfg.window_size, n_train, n_test, n_orig_features
        )
        y_train[~y_train_vmask] = np.nan

    y_test_plot = y_test.copy().astype(np.float64)
    if y_test_vmask is not None:
        y_test_plot[~y_test_vmask] = np.nan

    n_series_avail = min(n_series, n_orig_features)
    rng = np.random.RandomState(seed)
    series_indices: np.ndarray = rng.choice(
        n_orig_features, size=n_series_avail, replace=False
    )

    predictions: dict[str, np.ndarray] = {}
    fitted: dict[str, np.ndarray | None] = {}
    for name in method_names:
        pd = _pred_dir(_method_dir_for(cache_dir, name, cfg), cfg.eval_mode)
        predictions[name] = _load_predictions(pd)
        fitted[name] = _load_fitted(pd)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(method_names)
    t_full = np.arange(n_train + n_test)

    for plot_idx, series_idx in enumerate(series_indices):
        actual_full_s = np.concatenate(
            [y_train[:, series_idx], y_test_plot[:, series_idx]], axis=0
        )
        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(6, 2.2 * n_rows),
            sharex=True,
            squeeze=True,
        )
        if n_rows == 1:
            axes = [axes]
        for row, method_name in enumerate(method_names):
            pred_lstm = predictions[method_name]
            f_lstm = fitted[method_name]
            lstm_pred_mean = pred_lstm.mean(axis=0)[:, series_idx].copy()
            if y_test_vmask is not None:
                lstm_pred_mean[~y_test_vmask[:, series_idx]] = np.nan
            ax = axes[row]
            ax.plot(
                t_full, actual_full_s, label="Actual", color="C0", alpha=0.9
            )
            if f_lstm is not None:
                lstm_fit_mean = f_lstm.mean(axis=0)[:n_train, series_idx].copy()
                if y_train_vmask is not None:
                    lstm_fit_mean[~y_train_vmask[:, series_idx]] = np.nan
                ax.plot(
                    np.arange(n_train),
                    lstm_fit_mean,
                    label="Fitted (train)",
                    color="C1",
                    alpha=0.9,
                    linestyle="--",
                )
            ax.plot(
                np.arange(n_train, n_train + n_test),
                lstm_pred_mean,
                label="Pred (test)",
                color="C2",
                alpha=0.9,
                linestyle="-.",
            )
            ax.set_ylabel(method_name, fontsize=9)
            ax.legend(loc="upper right", fontsize=7)
        axes[-1].set_xlabel("Time step")
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


def _load_original_data(
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load the (n_timesteps, n_series) data matrix and optional valid_mask (True = observed)."""
    from lgta.preprocessing.pre_processing_datasets import PreprocessDatasets

    ppc = PreprocessDatasets(dataset=cfg.dataset_name, freq=cfg.freq)
    data = ppc.apply_preprocess()
    X = np.nan_to_num(
        data["predict"]["data_matrix"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    mask = data["predict"].get("valid_mask")
    valid_mask = mask.astype(bool) if mask is not None else None
    return X, valid_mask


def _generate_lgta(
    cfg: ExperimentConfig, cache: Path
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray | None]:
    """Train LGTA CVAE and generate synthetic dataset(s).

    Returns (X_orig, lgta_variants, valid_mask).
    Each element of lgta_variants has shape (n_timesteps, n_series).
    When cfg.n_variants > 1, one variant is generated per transformation in
    cfg.effective_transformations; otherwise a single variant is produced.
    """
    from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
    from lgta.model.generate_data import generate_synthetic_data
    from lgta.model.models import LatentMode

    eq_str = str(cfg.lgta_equiv_weight).replace(".", "_")
    weights_suffix_parts = [f"eq{eq_str}"]
    if cfg.lgta_sample_from_posterior:
        weights_suffix_parts.append("posterior")
    if not cfg.use_dynamic_features:
        weights_suffix_parts.append("nodyn")
    weights_suffix = "_".join(weights_suffix_parts)

    creator = CreateTransformedVersionsCVAE(
        dataset_name=cfg.dataset_name,
        freq=cfg.freq,
        window_size=cfg.window_size,
        weights_suffix=weights_suffix,
        weights_dir=cache / "model_weights",
        use_dynamic_features=cfg.use_dynamic_features,
    )
    model, _, _ = creator.fit(
        epochs=cfg.lgta_epochs,
        latent_dim=cfg.lgta_latent_dim,
        equiv_weight=cfg.lgta_equiv_weight,
        latent_mode=LatentMode.TEMPORAL,
    )
    _, _, z_mean, z_log_var = creator.predict(model)
    X_orig = creator.X_train_raw
    valid_mask = getattr(creator, "valid_mask", None)

    transformations = cfg.effective_transformations
    print(
        f"  LGTA generation: transformations={transformations!r}, "
        f"sigma={cfg.lgta_sigma} (applied in latent space)"
    )

    lgta_variants: list[np.ndarray] = []
    for transformation in transformations:
        rng = np.random.default_rng(SEED)
        X_lgta = generate_synthetic_data(
            model,
            z_mean,
            creator,
            transformation,
            [cfg.lgta_sigma],
            latent_mode=LatentMode.TEMPORAL,
            z_log_var=z_log_var if cfg.lgta_sample_from_posterior else None,
            sample_from_posterior=cfg.lgta_sample_from_posterior,
            rng=rng,
        )
        if valid_mask is not None:
            X_lgta = X_lgta * valid_mask.astype(np.float32)
        lgta_variants.append(X_lgta)

    del model, z_mean, z_log_var, creator
    _release_memory()

    return X_orig, lgta_variants, valid_mask


def _weights_path(weights_dir: Path, gen: TimeSeriesGenerator) -> Path:
    """Variant-agnostic path for a generator's trained model weights."""
    return weights_dir / f"{gen.__class__.__name__}_weights.pt"


def _fit_or_load(
    gen: TimeSeriesGenerator,
    X_orig: np.ndarray,
    weights_dir: Path | None,
) -> None:
    """Load cached weights if available, otherwise fit and save."""
    if weights_dir is not None:
        wp = _weights_path(weights_dir, gen)
        if gen.load_weights(wp):
            print(f"    Loaded {gen.name} weights from cache")
            return
    gen.fit(X_orig)
    if weights_dir is not None:
        gen.save_weights(_weights_path(weights_dir, gen))


def _generate_benchmark_variants(
    gen: TimeSeriesGenerator,
    X_orig: np.ndarray,
    transformations: list[str],
    valid_mask: np.ndarray | None,
    weights_dir: Path | None = None,
) -> list[np.ndarray]:
    """Generate synthetic variants for a benchmark generator.

    For DirectTransformGenerator: one variant per transformation in *transformations*.
    For other generators: calls generate() len(transformations) times (stochastic samples).
    Weights are loaded from / saved to *weights_dir* (variant-agnostic) so
    that expensive training is never repeated across single- and multi-variant runs.
    """
    variants: list[np.ndarray] = []
    if isinstance(gen, DirectTransformGenerator):
        for t in transformations:
            direct = DirectTransformGenerator(
                transformation=t, sigma=gen.sigma, seed=gen.seed
            )
            _fit_or_load(direct, X_orig, weights_dir)
            synth = direct.generate()
            synth = np.clip(synth, a_min=0, a_max=None)
            if valid_mask is not None:
                synth = synth * valid_mask.astype(np.float32)
            variants.append(synth)
    else:
        _fit_or_load(gen, X_orig, weights_dir)
        for _ in transformations:
            synth = gen.generate()
            synth = np.clip(synth, a_min=0, a_max=None)
            if valid_mask is not None:
                synth = synth * valid_mask.astype(np.float32)
            variants.append(synth)
    return variants


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
    """Run downstream forecasting comparison in the configured evaluation mode.

    EvalMode.TSTR — Train on Synthetic, Test on Real.  Each augmentation
    method generates synthetic data; a forecaster is trained exclusively on
    synthetic windows and evaluated on held-out real windows.

    EvalMode.DOWNSTREAM — Train on Original + Synthetic, Test on Real.
    Synthetic variants are stacked with the original training data so the
    forecaster sees both real and generated windows.

    Results are written to output_dir / <eval_mode> / <dynamic_subdir> / <lgta_config_slug>.
    Prediction caches are stored per eval-mode inside each method's cache
    directory so both modes can coexist.
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

    dyn_label = "WITH" if cfg.use_dynamic_features else "WITHOUT"
    print("=" * 70)
    print(f"DOWNSTREAM FORECASTING EXPERIMENT  [{cfg.eval_mode.value}]  [{dyn_label} dynamic features]")
    if single:
        print(f"  (single method: {method_canonical})")
    print("=" * 70)

    resource_usages: list[ResourceUsage] = []
    lgta_variants: list[np.ndarray] | None = None
    if single and method_canonical not in ("original", "lgta"):
        print("\n[1/4] Loading data ...")
        X_orig, valid_mask = _load_original_data(cfg)
    else:
        print("\n[1/4] Training LGTA and generating synthetic data ...")
        t0 = time.perf_counter()
        rss_before = _current_rss_mb()
        X_orig, lgta_variants, valid_mask = _generate_lgta(cfg, cache)
        rss_after = _current_rss_mb()
        resource_usages.append(
            ResourceUsage("LGTA", time.perf_counter() - t0, max(rss_before, rss_after))
        )

    n_orig_features = X_orig.shape[1]
    X_orig_win, y_orig_win = _prepare_windows(X_orig, cfg.window_size)
    n_test = max(1, int(0.15 * X_orig_win.shape[0]))
    n_train = X_orig_win.shape[0] - n_test
    X_test_windows = X_orig_win[n_train:]
    y_test = y_orig_win[n_train:]

    if not _has_shared_test_data(cache):
        y_train_mask = _y_train_mask_from_valid_mask(
            valid_mask, cfg.window_size, n_train, n_orig_features
        )
        scale = _mase_scale(
            X_orig_win[:n_train], y_orig_win[:n_train, :n_orig_features], y_train_mask
        )
        _save_shared_test_data(
            cache,
            X_orig,
            y_test,
            X_test_windows,
            n_orig_features,
            scale,
            cfg.window_size,
            valid_mask,
        )
    else:
        _, _, _, _, scale, _, _, _ = _load_shared_test_data(cache, cfg.window_size)

    all_results: list[ForecastResult] = []

    is_tstr = cfg.eval_mode is EvalMode.TSTR

    run_original = not single or method_canonical == "original"
    if run_original:
        orig_method_dir = _method_dir(cache, "Original")
        orig_pred = _pred_dir(orig_method_dir, cfg.eval_mode)
        if _has_predictions(orig_pred):
            print("\n[2/4] Original: loading from cache ...")
            X_orig, y_test, _, n_orig_features, scale, _, _, valid_mask = _load_shared_test_data(
                cache, cfg.window_size
            )
            n_train_win = X_orig.shape[0] - cfg.window_size - y_test.shape[0]
            y_test_mask = _y_test_mask_from_valid_mask(
                valid_mask, cfg.window_size, n_train_win, y_test.shape[0], n_orig_features
            )
            p_lstm = _load_predictions(orig_pred)
            all_results.extend(
                _results_from_predictions(
                    "Original", y_test, p_lstm, scale, y_test_mask
                )
            )
        else:
            print("\n[2/4] Evaluating Original (no augmentation) ...")
            _, _, _, _, scale, y_mean, y_std, _ = _load_shared_test_data(
                cache, cfg.window_size
            )
            y_test_mask = _y_test_mask_from_valid_mask(
                valid_mask, cfg.window_size, n_train, y_test.shape[0], n_orig_features
            )
            y_train_mask = _y_train_mask_from_valid_mask(
                valid_mask, cfg.window_size, n_train, n_orig_features
            )
            n_orig_copies = max(1, cfg.n_variants) if is_tstr else 1
            results, p_lstm, f_lstm = _evaluate_method(
                "Original",
                [X_orig] * n_orig_copies,
                n_orig_features,
                cfg.window_size,
                n_train,
                X_test_windows,
                y_test,
                scale,
                cfg,
                y_mean,
                y_std,
                y_test_mask,
                y_train_mask,
            )
            all_results.extend(results)
            _save_predictions(orig_pred, p_lstm, f_lstm)

    _release_memory()

    run_lgta = (not single or method_canonical == "lgta") and lgta_variants is not None
    if run_lgta:
        lgta_method_d = _lgta_method_dir(cache, cfg)
        lgta_pred = _pred_dir(lgta_method_d, cfg.eval_mode)
        if _has_predictions(lgta_pred):
            print("\n[3/4] LGTA: loading from cache ...")
            X_orig, y_test, _, n_orig_features, scale, _, _, valid_mask = _load_shared_test_data(
                cache, cfg.window_size
            )
            n_train_win = X_orig.shape[0] - cfg.window_size - y_test.shape[0]
            y_test_mask = _y_test_mask_from_valid_mask(
                valid_mask, cfg.window_size, n_train_win, y_test.shape[0], n_orig_features
            )
            p_lstm = _load_predictions(lgta_pred)
            all_results.extend(
                _results_from_predictions(
                    "LGTA", y_test, p_lstm, scale, y_test_mask
                )
            )
        else:
            print("\n[3/4] Evaluating LGTA ...")
            _, _, _, _, scale, y_mean, y_std, _ = _load_shared_test_data(
                cache, cfg.window_size
            )
            y_test_mask = _y_test_mask_from_valid_mask(
                valid_mask, cfg.window_size, n_train, y_test.shape[0], n_orig_features
            )
            y_train_mask = _y_train_mask_from_valid_mask(
                valid_mask, cfg.window_size, n_train, n_orig_features
            )
            lgta_train_variants = lgta_variants if is_tstr else [X_orig] + lgta_variants
            results, p_lstm, f_lstm = _evaluate_method(
                "LGTA",
                lgta_train_variants,
                n_orig_features,
                cfg.window_size,
                n_train,
                X_test_windows,
                y_test,
                scale,
                cfg,
                y_mean,
                y_std,
                y_test_mask,
                y_train_mask,
            )
            all_results.extend(results)
            _save_predictions(lgta_pred, p_lstm, f_lstm)
            _save_synthetic_variants(lgta_method_d, lgta_variants)

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
        _release_memory()
        X_orig, y_test, _, n_orig_features, scale, y_mean, y_std, valid_mask = _load_shared_test_data(
            cache, cfg.window_size
        )
        n_train_win = X_orig.shape[0] - cfg.window_size - y_test.shape[0]
        y_test_mask = _y_test_mask_from_valid_mask(
            valid_mask, cfg.window_size, n_train_win, y_test.shape[0], n_orig_features
        )
        y_train_mask = _y_train_mask_from_valid_mask(
            valid_mask, cfg.window_size, n_train_win, n_orig_features
        )
        n_v = cfg.n_variants
        for gen in benchmarks_to_run:
            name = _benchmark_display_name(gen, n_v)
            method_d = _method_dir(cache, name, n_v)
            bench_pred = _pred_dir(method_d, cfg.eval_mode)
            if _has_predictions(bench_pred):
                print(f"  {name}: loading from cache ...")
                p_lstm = _load_predictions(bench_pred)
                all_results.extend(
                    _results_from_predictions(
                        name, y_test, p_lstm, scale, y_test_mask
                    )
                )
                continue
            if _has_synthetic_variants(method_d, n_v):
                print(
                    f"  {name}: loading synthetic from cache, training forecasters ..."
                )
                variants = _load_synthetic_variants(method_d, n_v)
            else:
                print(f"  Generating {name} ({n_v} variant(s)) ...")
                t0 = time.perf_counter()
                rss_before = _current_rss_mb()
                variants = _generate_benchmark_variants(
                    gen, X_orig, cfg.effective_transformations, valid_mask,
                    weights_dir=cache,
                )
                rss_after = _current_rss_mb()
                resource_usages.append(
                    ResourceUsage(
                        name,
                        time.perf_counter() - t0,
                        max(rss_before, rss_after),
                    )
                )
                _save_synthetic_variants(method_d, variants)
            if valid_mask is not None:
                variants = [
                    v * valid_mask.astype(np.float32) for v in variants
                ]
            train_variants = variants if is_tstr else [X_orig] + variants
            results, p_lstm, f_lstm = _evaluate_method(
                name,
                train_variants,
                n_orig_features,
                cfg.window_size,
                n_train_win,
                X_test_windows,
                y_test,
                scale,
                cfg,
                y_mean,
                y_std,
                y_test_mask,
                y_train_mask,
            )
            all_results.extend(results)
            _save_predictions(bench_pred, p_lstm, f_lstm)
            _release_memory()

    _print_results(all_results)
    effective_out = cfg.output_dir / cfg.eval_mode.value / cfg.dynamic_subdir / _lgta_config_slug(cfg)
    print(f"\nResults written to: {effective_out}")
    _save_results(all_results, effective_out)
    if resource_usages:
        _save_resource_usage(resource_usages, effective_out)
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
    X_orig, y_test, _, n_orig_features, scale, _, _, valid_mask = _load_shared_test_data(
        cache, cfg.window_size
    )
    n_train_win = X_orig.shape[0] - cfg.window_size - y_test.shape[0]
    y_test_mask = _y_test_mask_from_valid_mask(
        valid_mask, cfg.window_size, n_train_win, y_test.shape[0], n_orig_features
    )
    all_results: list[ForecastResult] = []
    for method_name in _known_cache_method_names(cfg.n_variants):
        method_d = _method_dir_for(cache, method_name, cfg)
        pd = _pred_dir(method_d, cfg.eval_mode)
        if _has_predictions(pd):
            p_lstm = _load_predictions(pd)
            all_results.extend(
                _results_from_predictions(
                    method_name, y_test, p_lstm, scale, y_test_mask
                )
            )
    if not all_results:
        raise FileNotFoundError(
            f"No cached predictions found under {cache} for eval_mode={cfg.eval_mode.value}. "
            "Run at least one method first."
        )
    _print_results(all_results)
    effective_out = cfg.output_dir / cfg.eval_mode.value / cfg.dynamic_subdir / _lgta_config_slug(cfg)
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
    json_path = output_dir / "downstream_results.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "method": r.method,
                    "forecaster": r.forecaster,
                    "mase_mean": r.mase_mean,
                    "mase_std": r.mase_std,
                }
                for r in results
            ],
            indent=2,
        )
    )


def _save_resource_usage(
    resource_usages: list[ResourceUsage], output_dir: Path
) -> None:
    """Write resource_usage.json with time_seconds and memory_mb per method."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "resource_usage.json"
    path.write_text(
        json.dumps(
            [
                {
                    "method": u.method,
                    "time_seconds": u.time_seconds,
                    "memory_mb": u.memory_mb,
                }
                for u in resource_usages
            ],
            indent=2,
        )
    )


def _save_results_by_forecaster(
    results: list[ForecastResult], output_dir: Path
) -> None:
    """Write LSTM results markdown with methods ordered by % change vs Original (ascending)."""
    subset = [r for r in results if r.forecaster == "LSTM"]
    if not subset:
        return
    original = next((r for r in subset if r.method == "Original"), None)
    lines = ["# Downstream Forecasting Results — LSTM\n"]
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
    out = output_dir / "DOWNSTREAM_RESULTS_LSTM.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Results saved to {out}")


def _load_results_json(path: Path) -> list[ForecastResult]:
    """Load ForecastResult list from a downstream_results.json file."""
    raw = json.loads(path.read_text())
    return [
        ForecastResult(
            method=item["method"],
            forecaster=item["forecaster"],
            mase_mean=float(item["mase_mean"]),
            mase_std=float(item["mase_std"]),
        )
        for item in raw
    ]


def _parse_results_from_md(content: str) -> list[ForecastResult]:
    """Parse ForecastResult list from DOWNSTREAM_RESULTS.md table (Method | Forecaster | MASE (mean) | ± std)."""
    results: list[ForecastResult] = []
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("|") or line == "|" or "---" in line:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 4:
            continue
        method, forecaster, mase_str, std_str = parts[0], parts[1], parts[2], parts[3]
        if method == "Method" or forecaster == "Forecaster":
            continue
        try:
            mase_mean = float(mase_str)
            mase_std = float(std_str)
        except ValueError:
            continue
        results.append(
            ForecastResult(
                method=method,
                forecaster=forecaster,
                mase_mean=mase_mean,
                mase_std=mase_std,
            )
        )
    return results


def _load_results_from_dir(result_dir: Path) -> list[ForecastResult] | None:
    """Load results from result dir: prefer downstream_results.json, else parse DOWNSTREAM_RESULTS.md."""
    json_path = result_dir / "downstream_results.json"
    if json_path.exists():
        return _load_results_json(json_path)
    md_path = result_dir / "DOWNSTREAM_RESULTS.md"
    if md_path.exists():
        return _parse_results_from_md(md_path.read_text())
    return None


def _load_resource_usage_json(path: Path) -> list[ResourceUsage] | None:
    """Load ResourceUsage list from resource_usage.json; returns None if missing or invalid."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        return [
            ResourceUsage(
                method=item["method"],
                time_seconds=float(item["time_seconds"]),
                memory_mb=float(item["memory_mb"]),
            )
            for item in raw
        ]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _has_result_file(result_dir: Path) -> bool:
    return (result_dir / "downstream_results.json").exists() or (
        result_dir / "DOWNSTREAM_RESULTS.md"
    ).exists()


_DYNAMIC_SUBDIRS = ("with_dynamic", "without_dynamic")


def _discover_result_dirs(
    output_dir: Path,
) -> list[tuple[str, str, str, Path]]:
    """Find all result directories that contain downstream results.

    Returns list of (eval_mode, dynamic_setting, config_slug, dir_path).
    Scans output_dir/<mode>/<dynamic_subdir>/<slug> (new layout) and
    output_dir/<mode>/<slug> (legacy layout without dynamic subdir).
    """
    discovered: list[tuple[str, str, str, Path]] = []
    for mode in ("TSTR", "downstream_task"):
        mode_dir = output_dir / mode
        if not mode_dir.is_dir():
            continue
        for dyn in _DYNAMIC_SUBDIRS:
            dyn_dir = mode_dir / dyn
            if not dyn_dir.is_dir():
                continue
            for slug_dir in dyn_dir.iterdir():
                if slug_dir.is_dir() and _has_result_file(slug_dir):
                    discovered.append((mode, dyn, slug_dir.name, slug_dir))
        for slug_dir in mode_dir.iterdir():
            if (
                slug_dir.is_dir()
                and slug_dir.name not in _DYNAMIC_SUBDIRS
                and _has_result_file(slug_dir)
            ):
                discovered.append((mode, "with_dynamic", slug_dir.name, slug_dir))
    for slug_dir in output_dir.iterdir():
        if slug_dir.is_dir() and slug_dir.name not in ("TSTR", "downstream_task"):
            if _has_result_file(slug_dir):
                discovered.append(("legacy", "with_dynamic", slug_dir.name, slug_dir))
    return discovered


def _slug_to_freq(config_slug: str) -> str:
    """Extract frequency from config slug. Slug format: {dataset}_{freq}_w{N}_...
    Returns empty string if slug has fewer than two parts (e.g. legacy)."""
    parts = config_slug.split("_")
    return parts[1] if len(parts) >= 2 else ""


def _slug_to_dataset_and_variants(config_slug: str) -> tuple[str, str]:
    """Parse config slug into short dataset name and variant label (e.g. '3var' or '1var').

    Slug format: {dataset}_{freq}_w{N}_... or {dataset}_{freq}_w{N}_{n}var_...
    """
    parts = config_slug.split("_")
    dataset_name = parts[0] if parts else config_slug
    match = re.search(r"(\d+)var", config_slug)
    variants = f"{match.group(1)}var" if match else "1var"
    return dataset_name, variants


def _write_combined_summary(output_dir: Path) -> None:
    """Combine all per-dataset results (TSTR and downstream_task) into one summary.

    Reads downstream_results.json from each discovered result dir, computes
    % change vs Original per (dataset, eval_mode, dynamic_setting, forecaster),
    and writes COMBINED_RESULTS.md and COMBINED_RESULTS.csv with columns:
    Dataset | Freq | Method | Variants | Dynamic | TSTR MASE | TSTR % | downstream_task MASE | downstream_task %.
    """
    discovered = _discover_result_dirs(output_dir)
    if not discovered:
        print(
            "No result directories (downstream_results.json or DOWNSTREAM_RESULTS.md) found. "
            "Run experiments first (e.g. --all-datasets for both TSTR and downstream_task)."
        )
        return

    by_key: dict[tuple[str, str, str], list[ForecastResult]] = {}
    for eval_mode, dyn_setting, config_slug, dir_path in discovered:
        key = (config_slug, dyn_setting, eval_mode)
        loaded = _load_results_from_dir(dir_path)
        if loaded:
            by_key[key] = loaded

    slug_dyn_pairs = sorted({(s, d) for s, d, _ in by_key})
    methods_per_pair: dict[tuple[str, str], set[str]] = {}
    for (slug, dyn, _mode), results in by_key.items():
        methods_per_pair.setdefault((slug, dyn), set()).update(
            r.method for r in results
        )
    all_methods = sorted(
        set().union(*(s for s in methods_per_pair.values())) if methods_per_pair else set()
    )
    eval_modes = ["TSTR", "downstream_task"]
    if any(m == "legacy" for m, _, _, _ in discovered):
        eval_modes.append("legacy")

    _DYN_DISPLAY = {"with_dynamic": "Yes", "without_dynamic": "No"}

    def original_mase(
        slug: str, dyn: str, mode: str, forecaster: str
    ) -> float | None:
        key = (slug, dyn, mode)
        if key not in by_key:
            return None
        for r in by_key[key]:
            if r.method == "Original" and r.forecaster == forecaster:
                return r.mase_mean
        return None

    def mase_and_pct(
        slug: str, dyn: str, mode: str, method: str, forecaster: str
    ) -> tuple[str, str]:
        key = (slug, dyn, mode)
        if key not in by_key:
            return "", ""
        orig = original_mase(slug, dyn, mode, forecaster)
        for r in by_key[key]:
            if r.method == method and r.forecaster == forecaster:
                mase_str = f"{r.mase_mean:.4f}"
                if orig is not None and orig > 0:
                    pct = (r.mase_mean - orig) / orig * 100.0
                    pct_str = f"{pct:+.2f}%"
                else:
                    pct_str = ""
                return mase_str, pct_str
        return "", ""

    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "COMBINED_RESULTS.md"
    csv_path = output_dir / "COMBINED_RESULTS.csv"

    md_lines = [
        "# Combined Downstream Forecasting Results",
        "",
        "All datasets, TSTR and downstream_task (LSTM), with % change vs Original.",
        "",
    ]
    csv_rows: list[list[str]] = []
    header = ["Dataset", "Freq", "Method", "Variants", "Dynamic"]
    for mode in eval_modes:
        header.append(f"{mode} MASE")
        header.append(f"{mode} %")

    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "|".join(["--------"] * len(header)) + "|")
    csv_rows.append(header)

    for slug, dyn in slug_dyn_pairs:
        dataset_short, variants_label = _slug_to_dataset_and_variants(slug)
        freq_label = _slug_to_freq(slug)
        dyn_label = _DYN_DISPLAY.get(dyn, dyn)
        for method in all_methods:
            row_cells: list[str] = [dataset_short, freq_label, method, variants_label, dyn_label]
            has_any = False
            for mode in eval_modes:
                mase_str, pct_str = mase_and_pct(slug, dyn, mode, method, "LSTM")
                row_cells.extend([mase_str, pct_str])
                if mase_str or pct_str:
                    has_any = True
            if not has_any:
                continue
            md_lines.append("| " + " | ".join(row_cells) + " |")
            csv_rows.append(row_cells)

    md_path.write_text("\n".join(md_lines) + "\n")
    with csv_path.open("w") as f:
        f.write(",".join(csv_rows[0]) + "\n")
        for row in csv_rows[1:]:
            f.write(",".join(row) + "\n")

    print(f"Combined summary written to {md_path} and {csv_path}")


def _write_resource_usage_summary(output_dir: Path) -> None:
    """Write RESOURCE_USAGE.md and RESOURCE_USAGE.csv from resource_usage.json in each result dir.

    One row per (dataset, method, variants, dynamic) with Time (s) and Memory (MB).
    Uses one result dir per (config_slug, dynamic) to avoid duplicate rows across eval modes.
    """
    discovered = _discover_result_dirs(output_dir)
    slug_dyn_pairs = sorted({(slug, dyn) for _mode, dyn, slug, _ in discovered})
    dir_by_slug_dyn: dict[tuple[str, str], Path] = {}
    for _mode, dyn, slug, dir_path in discovered:
        key = (slug, dyn)
        if key not in dir_by_slug_dyn:
            dir_by_slug_dyn[key] = dir_path

    _DYN_DISPLAY = {"with_dynamic": "Yes", "without_dynamic": "No"}
    header = ["Dataset", "Freq", "Method", "Variants", "Dynamic", "Time (s)", "Memory (MB)"]
    md_lines = [
        "# Resource Usage by Dataset and Augmentation Method",
        "",
        "Time (s): wall-clock generation time. Memory (MB): peak RSS after generation.",
        "",
    ]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "|".join(["--------"] * len(header)) + "|")
    csv_rows: list[list[str]] = [header]

    for slug, dyn in slug_dyn_pairs:
        dir_path = dir_by_slug_dyn.get((slug, dyn))
        if dir_path is None:
            continue
        usages = _load_resource_usage_json(dir_path / "resource_usage.json")
        if not usages:
            continue
        dataset_short, variants_label = _slug_to_dataset_and_variants(slug)
        freq_label = _slug_to_freq(slug)
        dyn_label = _DYN_DISPLAY.get(dyn, dyn)
        for u in usages:
            row = [
                dataset_short,
                freq_label,
                u.method,
                variants_label,
                dyn_label,
                f"{u.time_seconds:.2f}",
                f"{u.memory_mb:.2f}",
            ]
            md_lines.append("| " + " | ".join(row) + " |")
            csv_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "RESOURCE_USAGE.md"
    csv_path = output_dir / "RESOURCE_USAGE.csv"
    md_path.write_text("\n".join(md_lines) + "\n")
    with csv_path.open("w") as f:
        f.write(",".join(csv_rows[0]) + "\n")
        for row in csv_rows[1:]:
            f.write(",".join(row) + "\n")
    print(f"Resource usage summary written to {md_path} and {csv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run downstream forecasting experiment (optionally a single method)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g. tourism, wiki2, labour, m3, m4). Default: tourism.",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default=None,
        help="Time series frequency (e.g. Y, Q, M, D, W, H). Y/Q/M for m3 and m4, W/H for m4.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run the experiment for all supported datasets (tourism, wiki2, labour, m3, m4) with their default frequencies.",
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
    parser.add_argument(
        "--variant-transformations",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Generate multiple synthetic variants per method (e.g. --variant-transformations jitter scaling magnitude_warp). "
            "For LGTA and Direct each variant uses a different transformation; "
            "for other generators each variant is a different random sample."
        ),
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="TSTR",
        choices=[m.value for m in EvalMode],
        help=(
            "Evaluation strategy: TSTR trains only on synthetic data; "
            "downstream_task trains on original + synthetic data."
        ),
    )
    parser.add_argument(
        "--no-dynamic-features",
        action="store_true",
        help="Disable dynamic time features for the LGTA CVAE model.",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Only write COMBINED_RESULTS.md/csv from existing downstream_results.json files (no training).",
    )
    args = parser.parse_args()

    if args.combine_only:
        _write_combined_summary(args.output_dir)
        _write_resource_usage_summary(args.output_dir)
        sys.exit(0)

    variant_transformations = args.variant_transformations or []
    eval_mode = EvalMode(args.eval_mode)
    dynamic_settings: list[bool] = (
        [True, False] if args.all_datasets and not args.no_dynamic_features else
        [not args.no_dynamic_features]
    )

    if args.all_datasets:
        for use_dyn in dynamic_settings:
            dyn_label = "WITH" if use_dyn else "WITHOUT"
            print(f"\n{'#'*70}")
            print(f"# Dynamic features: {dyn_label}")
            print(f"{'#'*70}")
            for i, (dataset_name, freq) in enumerate(DEFAULT_DATASET_CONFIGS):
                _release_memory()
                print(f"\n{'='*70}")
                print(f"Dataset {i+1}/{len(DEFAULT_DATASET_CONFIGS)}: {dataset_name} (freq={freq})")
                print("="*70)
                cfg = ExperimentConfig(
                    dataset_name=dataset_name,
                    freq=freq,
                    output_dir=args.output_dir,
                    lgta_sample_from_posterior=args.lgta_sample_from_posterior,
                    variant_transformations=variant_transformations,
                    eval_mode=eval_mode,
                    use_dynamic_features=use_dyn,
                )
                run_downstream_forecasting(
                    cfg,
                    method=args.method,
                    results_only=args.results_only,
                )
        _write_combined_summary(args.output_dir)
        _write_resource_usage_summary(args.output_dir)
    else:
        dataset_name = args.dataset if args.dataset is not None else "tourism"
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
            variant_transformations=variant_transformations,
            eval_mode=eval_mode,
            use_dynamic_features=not args.no_dynamic_features,
        )
        run_downstream_forecasting(
            cfg,
            method=args.method,
            results_only=args.results_only,
        )
