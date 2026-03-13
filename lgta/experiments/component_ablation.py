"""
Component ablation study for L-GTA. Isolates the contribution of the
two key innovations (temporal latent space, equivariant decoder training)
using a 2x2 matrix plus an encoder-type axis. Measures controllability,
reconstruction quality, and transformation signature preservation.
"""

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr, wasserstein_distance

from lgta.experiments.transformation_signatures import (
    TransformationFingerprint,
    compute_fingerprint,
)
from lgta.benchmarks import get_default_benchmark_generators
from lgta.benchmarks.base import TimeSeriesGenerator
from lgta.benchmarks.direct import DirectTransformGenerator
from lgta.evaluation.metrics import MetricsAggregator
from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from lgta.model.generate_data import generate_synthetic_data
from lgta.model.models import EncoderType, LatentMode
from lgta.transformations.manipulate_data import ManipulateData

FREQ_TO_SAMPLING_FREQ: dict[str, int] = {
    "Y": 1,
    "A": 1,
    "Q": 4,
    "M": 12,
    "W": 52,
    "D": 365,
    "H": 8760,
}

DEFAULT_ABLATION_DATASETS: list[tuple[str, str]] = [
    ("tourism", "Q"),
    ("wiki2", "D"),
    ("labour", "M"),
    ("m3", "Y"),
    ("m3", "M"),
]

ABLATION_SIGMA_VALUES: list[float] = [0.5, 1.0, 2.0]

CACHE_ABLATION_ROOT = Path("assets/cache/component_ablation")


def _results_cache_dir(dataset_name: str, freq: str) -> Path:
    return CACHE_ABLATION_ROOT / dataset_name / freq / "results"


def _load_cached_transformation_result(
    cache_dir: Path,
    model_key: str,
    transformation: str,
    sigma_values: list[float],
    n_repetitions: int,
) -> TransformationResult | None:
    path = cache_dir / f"{_variant_slug(model_key)}_{transformation}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if (
            data.get("sigma_values") == sigma_values
            and data.get("n_repetitions") == n_repetitions
        ):
            return data["result"]
    except (pickle.PickleError, KeyError):
        pass
    return None


def _save_cached_transformation_result(
    cache_dir: Path,
    model_key: str,
    transformation: str,
    sigma_values: list[float],
    n_repetitions: int,
    result: TransformationResult,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_variant_slug(model_key)}_{transformation}.pkl"
    with open(path, "wb") as f:
        pickle.dump(
            {
                "sigma_values": sigma_values,
                "n_repetitions": n_repetitions,
                "result": result,
            },
            f,
        )


def _variant_slug(name: str) -> str:
    """Safe filename slug from ablation variant name."""
    slug = "".join(
        c if c.isalnum() or c in " _-" else "_" for c in name
    ).replace(" ", "_").replace("-", "_").strip("_")
    return slug or "variant"


def _sigma_filename(sigma: float) -> str:
    """Filename-safe sigma string (e.g. 0.1 -> 0_1, 2.0 -> 2_0)."""
    return str(sigma).replace(".", "_")


def _plot_original_vs_synthetic_ablation(
    X_orig: np.ndarray,
    X_synthetic: np.ndarray,
    variant_name: str,
    output_path: Path,
    n_series: int = 6,
    seed: int = 42,
    sigma_label: str | None = None,
    valid_mask: np.ndarray | None = None,
) -> None:
    """Plot original vs synthetic for one ablation variant (one column, n_series rows).
    If valid_mask is set, only observed positions are shown; missing regions are masked (gaps in lines)."""
    n_timesteps, n_total_series = X_orig.shape
    n_plot = min(n_series, n_total_series)
    rng = np.random.default_rng(seed)
    series_indices: np.ndarray = rng.choice(
        n_total_series, size=n_plot, replace=False
    )
    t = np.arange(n_timesteps)
    fig, axes = plt.subplots(
        n_plot, 1, figsize=(6, 2.5 * n_plot), sharex=True, squeeze=False
    )
    for row, series_idx in enumerate(series_indices):
        ax = axes[row, 0]
        y_orig = X_orig[:, series_idx].astype(float)
        y_syn = X_synthetic[:, series_idx].astype(float)
        if valid_mask is not None:
            inv = ~np.asarray(valid_mask[:, series_idx], dtype=bool)
            y_orig = y_orig.copy()
            y_syn = y_syn.copy()
            y_orig[inv] = np.nan
            y_syn[inv] = np.nan
        ax.plot(t, y_orig, label="Original", color="C0", alpha=0.9)
        ax.plot(t, y_syn, label="Generated", color="C1", alpha=0.9)
        ax.set_ylabel(f"Series {series_idx}")
        ax.legend(loc="upper right", fontsize=7)
    axes[-1, 0].set_xlabel("Time")
    title = f"{variant_name}: original vs generated"
    if sigma_label is not None:
        title += f" (σ={sigma_label})"
    plt.suptitle(title, y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def _plot_merged_original_vs_synthetic(
    X_orig: np.ndarray,
    accumulated: list[tuple[str, float, np.ndarray]],
    variant_names: list[str],
    sigma_values: list[float],
    series_indices: np.ndarray,
    output_dir: Path,
    transformation: str | None = None,
    valid_mask: np.ndarray | None = None,
) -> None:
    """One figure per series: rows=variants, cols=sigmas; each cell original vs generated.
    If valid_mask is set, only observed positions are shown; missing regions are masked."""
    lookup: dict[tuple[str, float], np.ndarray] = {
        (name, sigma): X for name, sigma, X in accumulated
    }
    n_sigmas = len(sigma_values)
    n_variants = len(variant_names)
    t = np.arange(X_orig.shape[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    trans_suffix = f"_{transformation}" if transformation else ""
    for series_idx in series_indices:
        fig, axes = plt.subplots(
            n_variants,
            n_sigmas,
            figsize=(2.5 * n_sigmas, 1.8 * n_variants),
            sharex=True,
            squeeze=False,
        )
        y_orig = X_orig[:, series_idx].astype(float)
        if valid_mask is not None:
            inv = ~np.asarray(valid_mask[:, series_idx], dtype=bool)
            y_orig_plot = y_orig.copy()
            y_orig_plot[inv] = np.nan
        else:
            y_orig_plot = y_orig
        for i_var, name in enumerate(variant_names):
            for i_sigma, sigma in enumerate(sigma_values):
                ax = axes[i_var, i_sigma]
                key = (name, sigma)
                if key not in lookup:
                    continue
                X_syn = lookup[key]
                y_syn = X_syn[:, series_idx].astype(float).copy()
                if valid_mask is not None:
                    y_syn[inv] = np.nan
                ax.plot(t, y_orig_plot, label="Original", color="C0", alpha=0.9)
                ax.plot(t, y_syn, label="Generated", color="C1", alpha=0.9)
                if i_var == 0:
                    ax.set_title(f"σ={sigma}", fontsize=9)
                if i_sigma == 0:
                    ax.set_ylabel(name, fontsize=8)
                ax.legend(loc="upper right", fontsize=6)
        for i_sigma in range(n_sigmas):
            axes[-1, i_sigma].set_xlabel("Time")
        title = f"Series {series_idx}: original vs generated (rows=variants, cols=σ)"
        if transformation:
            title += f" — {transformation}"
        fig.suptitle(title, y=1.002)
        plt.tight_layout()
        out_path = output_dir / f"ablation_merged_series_{series_idx}{trans_suffix}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


ALL_TRANSFORMATIONS: list[str] = [
    "jitter",
    "scaling",
    "magnitude_warp",
    "drift",
    "trend",
]


@dataclass
class ComponentAblationConfig:
    """Single ablation variant configuration."""

    name: str
    latent_mode: LatentMode = LatentMode.TEMPORAL
    equiv_weight: float = 0.0
    encoder_type: EncoderType = EncoderType.FULL
    use_channel_attention: bool = False
    use_dynamic_features: bool = True
    latent_dim: int = 4
    kl_weight_max: float = 0.1
    kl_anneal_epochs: int = 30
    epochs: int = 1000
    sigma_values: list[float] = field(
        default_factory=lambda: list(ABLATION_SIGMA_VALUES)
    )
    n_repetitions: int = 5

    @property
    def model_key(self) -> str:
        ch = "_chattn" if self.use_channel_attention else ""
        dyn = "_dyn" if self.use_dynamic_features else "_noDyn"
        return (
            f"{self.latent_mode.value}_enc{self.encoder_type.value}{ch}"
            f"_eq{self.equiv_weight}_lat{self.latent_dim}"
            f"_kl{self.kl_weight_max}{dyn}"
        )


@dataclass
class TransformationResult:
    """Metrics for one transformation under one ablation variant."""

    transformation: str
    spearman_rho: float
    is_monotonic: bool
    mean_distances: np.ndarray
    std_distances: np.ndarray
    fingerprint: TransformationFingerprint
    direct_fingerprint: TransformationFingerprint
    fingerprint_distance: float
    direct_spearman_rho: float
    direct_is_monotonic: bool
    direct_mse: float


@dataclass
class ComponentAblationResult:
    """Full result for one ablation variant across all transformations."""

    name: str
    latent_mode: LatentMode
    equiv_weight: float
    encoder_type: EncoderType
    use_channel_attention: bool
    use_dynamic_features: bool
    recon_mse: float
    transformation_results: dict[str, TransformationResult]


@dataclass
class AblationRunResult:
    """Result of run_component_ablation: results plus data needed for comparison report."""

    results: list[ComponentAblationResult]
    X_orig: np.ndarray
    valid_mask: np.ndarray | None
    trained_models: dict[str, tuple]


def _fingerprint_distance(
    lgta: TransformationFingerprint,
    direct: TransformationFingerprint,
) -> float:
    """Euclidean distance between LGTA and direct fingerprint vectors.

    Fingerprints have three components (see transformation_signatures.TransformationFingerprint):
    - autocorrelation: lag-1 autocorrelation of residuals (smooth vs i.i.d. transforms).
    - linearity: R² of residuals vs time (trend vs non-trend).
    - amplitude_dependence: |residuals| vs |original| (multiplicative vs additive).
    Small distance means LGTA output matches the transformation character of direct augmentation.
    """
    v_lgta = np.array([
        lgta.autocorrelation, lgta.linearity, lgta.amplitude_dependence,
    ])
    v_direct = np.array([
        direct.autocorrelation, direct.linearity, direct.amplitude_dependence,
    ])
    return float(np.linalg.norm(v_lgta - v_direct))


def _compute_wasserstein(
    X_ref: np.ndarray,
    X_aug: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Mean Wasserstein distance per series. If valid_mask is set, only observed positions are used."""
    distances: list[float] = []
    for i in range(X_ref.shape[1]):
        a = X_ref[:, i]
        b = X_aug[:, i]
        if valid_mask is not None:
            valid = np.asarray(valid_mask[:, i], dtype=bool)
            if np.sum(valid) == 0:
                continue
            a = a[valid]
            b = b[valid]
        distances.append(float(wasserstein_distance(a, b)))
    return float(np.mean(distances)) if distances else 0.0


def _evaluate_transformation(
    config: ComponentAblationConfig,
    model,
    z_mean: np.ndarray,
    X_orig: np.ndarray,
    vae_creator: CreateTransformedVersionsCVAE,
    transformation: str,
    valid_mask: np.ndarray | None = None,
) -> TransformationResult:
    """Sweep sigma for one transformation, compute controllability + fingerprint.

    Sigma usage: Mean rho and Monotonic % use all config.sigma_values (sweep).
    Fingerprint, fingerprint_distance, and direct_mse use only the last sigma
    (max value in the sweep), since we keep a single LGTA/direct output per transformation.
    If valid_mask is set, comparisons use only observed positions.
    """
    n_sigma = len(config.sigma_values)
    lgta_dists = np.zeros((n_sigma, config.n_repetitions))
    direct_dists = np.zeros((n_sigma, config.n_repetitions))

    last_X_lgta: np.ndarray | None = None
    last_X_direct: np.ndarray | None = None

    for i, sigma in enumerate(config.sigma_values):
        for rep in range(config.n_repetitions):
            X_lgta = generate_synthetic_data(
                model, z_mean, vae_creator,
                transformation, [sigma],
                latent_mode=config.latent_mode,
            )
            X_orig_scaled = vae_creator.scaler_target.transform(X_orig)
            X_direct_scaled = ManipulateData(
                x=X_orig_scaled,
                transformation=transformation,
                parameters=[sigma],
            ).apply_transf()
            X_direct = vae_creator.scaler_target.inverse_transform(X_direct_scaled)

            lgta_dists[i, rep] = _compute_wasserstein(
                X_orig, X_lgta, valid_mask=valid_mask
            )
            direct_dists[i, rep] = _compute_wasserstein(
                X_orig, X_direct, valid_mask=valid_mask
            )

            if i == n_sigma - 1 and rep == config.n_repetitions - 1:
                last_X_lgta = X_lgta
                last_X_direct = X_direct

    lgta_means = lgta_dists.mean(axis=1)
    lgta_rho, _ = spearmanr(config.sigma_values, lgta_means)
    lgta_mono = all(
        lgta_means[j] <= lgta_means[j + 1] for j in range(len(lgta_means) - 1)
    )

    direct_means = direct_dists.mean(axis=1)
    direct_rho, _ = spearmanr(config.sigma_values, direct_means)
    direct_mono = all(
        direct_means[j] <= direct_means[j + 1]
        for j in range(len(direct_means) - 1)
    )

    assert last_X_lgta is not None and last_X_direct is not None
    lgta_fp = compute_fingerprint(X_orig, last_X_lgta, valid_mask=valid_mask)
    direct_fp = compute_fingerprint(X_orig, last_X_direct, valid_mask=valid_mask)
    fp_dist = _fingerprint_distance(lgta_fp, direct_fp)

    residual_direct = last_X_direct - X_orig
    if valid_mask is not None:
        residual_direct = residual_direct[np.asarray(valid_mask, dtype=bool)]
    direct_mse = float(np.mean(residual_direct ** 2))

    return TransformationResult(
        transformation=transformation,
        spearman_rho=float(lgta_rho),
        is_monotonic=lgta_mono,
        mean_distances=lgta_means,
        std_distances=lgta_dists.std(axis=1),
        fingerprint=lgta_fp,
        direct_fingerprint=direct_fp,
        fingerprint_distance=fp_dist,
        direct_spearman_rho=float(direct_rho) if not np.isnan(direct_rho) else 0.0,
        direct_is_monotonic=direct_mono,
        direct_mse=direct_mse,
    )


def run_component_ablation(
    configs: list[ComponentAblationConfig],
    dataset_name: str = "tourism",
    freq: str = "Q",
    scaler_type: str = "standard",
    plots_output_dir: Path | None = None,
    weights_dir: Path | None = None,
    plot_synthetic_transformation: str = "jitter",
    plot_synthetic_sigma: float = 0.5,
    plot_n_series: int = 6,
    plot_seed: int = 42,
    plot_per_sigma: bool = False,
    use_cache: bool = True,
) -> AblationRunResult:
    """Train each variant and evaluate across all transformations.

    Models are saved under weights_dir with a unique suffix per config (model_key)
    and loaded on subsequent runs when use_cache is True. Transformation
    evaluation results are cached under assets/cache/component_ablation/<dataset>/<freq>/results.
    If plots_output_dir is set, for each variant generates synthetic data
    and saves original-vs-synthetic plot(s) under plots_output_dir (no subfolders).
    If plot_per_sigma is False: one plot per variant using plot_synthetic_sigma.
    If plot_per_sigma is True: one plot per (variant, sigma) for each sigma in
    that variant's config.sigma_values, using plot_synthetic_transformation.
    """
    vae_creators: dict[bool, CreateTransformedVersionsCVAE] = {}
    trained_models: dict[str, tuple] = {}
    results: list[ComponentAblationResult] = []
    X_orig: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    results_cache_dir = _results_cache_dir(dataset_name, freq) if use_cache else None

    for config in configs:
        if config.model_key not in trained_models:
            use_dyn = config.use_dynamic_features
            if use_dyn not in vae_creators:
                vae_creators[use_dyn] = CreateTransformedVersionsCVAE(
                    dataset_name=dataset_name,
                    freq=freq,
                    scaler_type=scaler_type,
                    weights_dir=weights_dir,
                    device=torch.device("cpu"),
                    use_dynamic_features=use_dyn,
                )
            vae_creator = vae_creators[use_dyn]

            print(f"\n{'='*60}")
            print(f"Training: {config.name}")
            print(f"  latent_mode={config.latent_mode.value}, "
                  f"equiv={config.equiv_weight}, "
                  f"encoder={config.encoder_type.value}, "
                  f"channel_attn={config.use_channel_attention}, "
                  f"dynamic={config.use_dynamic_features}")
            print(f"{'='*60}")

            model, _, _ = vae_creator.fit(
                epochs=config.epochs,
                latent_dim=config.latent_dim,
                kl_anneal_epochs=config.kl_anneal_epochs,
                kl_weight_max=config.kl_weight_max,
                load_weights=use_cache,
                encoder_type=config.encoder_type,
                equiv_weight=config.equiv_weight,
                latent_mode=config.latent_mode,
                use_channel_attention=config.use_channel_attention,
                weights_suffix_override=config.model_key,
            )
            if X_orig is None:
                X_orig = vae_creator.X_train_raw
                valid_mask = getattr(vae_creator, "valid_mask", None)
            X_recon, _, z_mean, _ = vae_creator.predict(
                model, detemporalize_method="mean",
            )
            recon_mse = float(np.mean((X_orig - X_recon) ** 2))
            trained_models[config.model_key] = (model, z_mean, recon_mse, vae_creator)

        model, z_mean, recon_mse, vae_creator = trained_models[config.model_key]

        transf_results: dict[str, TransformationResult] = {}
        for transf in ALL_TRANSFORMATIONS:
            cached = (
                _load_cached_transformation_result(
                    results_cache_dir,
                    config.model_key,
                    transf,
                    config.sigma_values,
                    config.n_repetitions,
                )
                if results_cache_dir is not None
                else None
            )
            if cached is not None:
                print(f"\n  Evaluating {config.name} / {transf}... (from cache)")
                transf_results[transf] = cached
            else:
                print(f"\n  Evaluating {config.name} / {transf}...")
                transf_results[transf] = _evaluate_transformation(
                    config, model, z_mean, X_orig, vae_creator, transf,
                    valid_mask=valid_mask,
                )
                if results_cache_dir is not None:
                    _save_cached_transformation_result(
                        results_cache_dir,
                        config.model_key,
                        transf,
                        config.sigma_values,
                        config.n_repetitions,
                        transf_results[transf],
                    )
            tr = transf_results[transf]
            print(f"    rho={tr.spearman_rho:.3f}  "
                  f"mono={'Y' if tr.is_monotonic else 'N'}  "
                  f"fp_dist={tr.fingerprint_distance:.3f}")

        results.append(ComponentAblationResult(
            name=config.name,
            latent_mode=config.latent_mode,
            equiv_weight=config.equiv_weight,
            encoder_type=config.encoder_type,
            use_channel_attention=config.use_channel_attention,
            use_dynamic_features=config.use_dynamic_features,
            recon_mse=recon_mse,
            transformation_results=transf_results,
        ))

        if plots_output_dir is not None and X_orig is not None:
            slug = _variant_slug(config.name)
            rng = np.random.default_rng(plot_seed)
            if not plot_per_sigma:
                X_synthetic = generate_synthetic_data(
                    model,
                    z_mean,
                    vae_creator,
                    plot_synthetic_transformation,
                    [plot_synthetic_sigma],
                    latent_mode=config.latent_mode,
                    rng=rng,
                )
                out_path = plots_output_dir / (
                    f"ablation_original_vs_synthetic_{slug}.png"
                )
                _plot_original_vs_synthetic_ablation(
                    X_orig,
                    X_synthetic,
                    config.name,
                    out_path,
                    n_series=plot_n_series,
                    seed=plot_seed,
                    valid_mask=valid_mask,
                )

    if (
        plots_output_dir is not None
        and X_orig is not None
        and plot_per_sigma
    ):
        variant_names = [c.name for c in configs]
        sigma_values = configs[0].sigma_values
        rng = np.random.default_rng(plot_seed)
        n_total_series = X_orig.shape[1]
        n_merged_series = min(3, n_total_series)
        series_indices = rng.choice(
            n_total_series, size=n_merged_series, replace=False
        )
        for transformation in ALL_TRANSFORMATIONS:
            accumulated_synthetics = []
            for config in configs:
                model, z_mean, _, vae_creator_cfg = trained_models[config.model_key]
                for sigma in sigma_values:
                    X_synthetic = generate_synthetic_data(
                        model,
                        z_mean,
                        vae_creator_cfg,
                        transformation,
                        [sigma],
                        latent_mode=config.latent_mode,
                        rng=rng,
                    )
                    accumulated_synthetics.append(
                        (config.name, sigma, X_synthetic.copy())
                    )
            _plot_merged_original_vs_synthetic(
                X_orig,
                accumulated_synthetics,
                variant_names,
                sigma_values,
                series_indices,
                plots_output_dir,
                transformation=transformation,
                valid_mask=valid_mask,
            )

    assert X_orig is not None
    return AblationRunResult(
        results=results,
        X_orig=X_orig,
        valid_mask=valid_mask,
        trained_models=trained_models,
    )


ABLATION_CSV_COLUMNS: tuple[str, ...] = (
    "Variant",
    "Mean ρ",
    "Monotonic %",
    "Recon MSE",
    "Mean FP dist",
)


def write_ablation_table_md(
    results: list[ComponentAblationResult],
    sigma_values: list[float],
    output_dir: Path,
) -> None:
    """Write ablation_results.csv and ablation_results.md with a table of all L-GTA variants (no benchmarks)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sigma_str = ", ".join(str(s) for s in sigma_values)
    csv_path = output_dir / "ablation_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ABLATION_CSV_COLUMNS))
        writer.writeheader()
        for r in results:
            rhos = [tr.spearman_rho for tr in r.transformation_results.values()]
            monos = [tr.is_monotonic for tr in r.transformation_results.values()]
            fp_dists = [tr.fingerprint_distance for tr in r.transformation_results.values()]
            mean_rho = float(np.mean(rhos))
            mono_pct = float(np.mean(monos)) * 100.0
            mean_fp = float(np.mean(fp_dists))
            writer.writerow({
                "Variant": r.name,
                "Mean ρ": f"{mean_rho:.4f}",
                "Monotonic %": f"{mono_pct:.4f}",
                "Recon MSE": f"{r.recon_mse:.4f}",
                "Mean FP dist": f"{mean_fp:.4f}",
            })
    print(f"Saved: {csv_path}")
    lines = [
        "# Component Ablation Results (L-GTA variants)",
        "",
        f"σ ∈ {{{sigma_str}}}.",
        "",
        "| Variant | Mean ρ | Monotonic % | Recon MSE | Mean FP dist |",
        "|----------|--------|-------------|-----------|--------------|",
    ]
    for r in results:
        rhos = [tr.spearman_rho for tr in r.transformation_results.values()]
        monos = [tr.is_monotonic for tr in r.transformation_results.values()]
        fp_dists = [tr.fingerprint_distance for tr in r.transformation_results.values()]
        mean_rho = float(np.mean(rhos))
        mono_pct = float(np.mean(monos)) * 100.0
        mean_fp = float(np.mean(fp_dists))
        lines.append(
            f"| {r.name} | {mean_rho:.4f} | {mono_pct:.1f}% | {r.recon_mse:.4f} | {mean_fp:.4f} |"
        )
    md_path = output_dir / "ablation_results.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {md_path}")


FINGERPRINT_CSV_COLUMNS: tuple[str, ...] = (
    "variant",
    "transformation",
    "autocorrelation",
    "linearity",
    "amplitude_dependence",
    "spearman_rho",
    "monotonic",
)

FINGERPRINT_CSV_COLUMNS_LEGACY: tuple[str, ...] = (
    "variant",
    "transformation",
    "autocorrelation",
    "linearity",
    "amplitude_dependence",
)


def _save_fingerprint_csv(
    results: list[ComponentAblationResult],
    output_dir: Path,
) -> None:
    """Persist per-transformation fingerprint values for every ablation variant
    plus the Direct baseline so that cross-dataset summary plots can be
    reconstructed without re-running experiments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fingerprint_data.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(FINGERPRINT_CSV_COLUMNS))
        writer.writeheader()
        for r in results:
            for t in ALL_TRANSFORMATIONS:
                tr = r.transformation_results[t]
                fp = tr.fingerprint
                writer.writerow({
                    "variant": r.name,
                    "transformation": t,
                    "autocorrelation": f"{fp.autocorrelation:.6f}",
                    "linearity": f"{fp.linearity:.6f}",
                    "amplitude_dependence": f"{fp.amplitude_dependence:.6f}",
                    "spearman_rho": f"{tr.spearman_rho:.6f}",
                    "monotonic": "1" if tr.is_monotonic else "0",
                })
        first_result = results[0]
        for t in ALL_TRANSFORMATIONS:
            tr0 = first_result.transformation_results[t]
            dfp = tr0.direct_fingerprint
            writer.writerow({
                "variant": "Direct",
                "transformation": t,
                "autocorrelation": f"{dfp.autocorrelation:.6f}",
                "linearity": f"{dfp.linearity:.6f}",
                "amplitude_dependence": f"{dfp.amplitude_dependence:.6f}",
                "spearman_rho": f"{tr0.direct_spearman_rho:.6f}",
                "monotonic": "1" if tr0.direct_is_monotonic else "0",
            })
    print(f"Saved: {csv_path}")


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load rows from a CSV file; returns empty list if file is missing or empty."""
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _to_float(value: str) -> float | None:
    """Best-effort float parse: handles '—' and percent suffix."""
    s = (value or "").strip()
    if not s or s == "—":
        return None
    if s.endswith("%"):
        s = s[:-1].strip()
    try:
        return float(s)
    except ValueError:
        return None


def write_cross_dataset_ablation_summary(
    base_output_dir: Path,
    datasets: list[tuple[str, str]],
) -> None:
    """Aggregate ablation_results.csv across datasets and write a cross-dataset summary.

    Averages metrics per variant over all (dataset, freq) pairs where that variant appears.
    """
    variant_to_metrics: dict[str, dict[str, list[float]]] = {}
    for dataset_name, freq in datasets:
        dir_path = base_output_dir / dataset_name / freq
        rows = _load_csv_rows(dir_path / "ablation_results.csv")
        if not rows:
            continue
        for row in rows:
            variant = row.get("Variant")
            if not variant:
                continue
            metrics = variant_to_metrics.setdefault(variant, {})
            for key in ("Mean ρ", "Monotonic %", "Recon MSE", "Mean FP dist"):
                val = _to_float(row.get(key, ""))
                if val is None:
                    continue
                metrics.setdefault(key, []).append(val)

    if not variant_to_metrics:
        return

    def fmt(x: float | None) -> str:
        return f"{x:.4f}" if x is not None else "—"

    cross_headers = ["Variant", "Mean ρ (avg)", "Monotonic % (avg)", "Recon MSE (avg)", "Mean FP dist (avg)", "Cells"]
    summary_rows: list[tuple[str, float | None, float | None, float | None, float | None, int]] = []
    for variant in sorted(variant_to_metrics.keys()):
        m = variant_to_metrics[variant]
        cells_count = max(len(v) for v in m.values()) if m else 0
        mean_rho = (sum(m["Mean ρ"]) / len(m["Mean ρ"])) if m.get("Mean ρ") else None
        mono = (sum(m["Monotonic %"]) / len(m["Monotonic %"])) if m.get("Monotonic %") else None
        recon = (sum(m["Recon MSE"]) / len(m["Recon MSE"])) if m.get("Recon MSE") else None
        fp = (sum(m["Mean FP dist"]) / len(m["Mean FP dist"])) if m.get("Mean FP dist") else None
        summary_rows.append((variant, mean_rho, mono, recon, fp, cells_count))

    csv_out = base_output_dir / "CROSS_DATASET_ABLATION.csv"
    with csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cross_headers)
        w.writeheader()
        for variant, mean_rho, mono, recon, fp, cells_count in summary_rows:
            w.writerow({
                "Variant": variant,
                "Mean ρ (avg)": fmt(mean_rho) if mean_rho is not None else "",
                "Monotonic % (avg)": fmt(mono) if mono is not None else "",
                "Recon MSE (avg)": fmt(recon) if recon is not None else "",
                "Mean FP dist (avg)": fmt(fp) if fp is not None else "",
                "Cells": str(cells_count),
            })
    print(f"Saved: {csv_out}")

    lines: list[str] = [
        "# Cross-dataset Component Ablation Summary",
        "",
        "Metrics are averaged over all (dataset, freq) pairs where a variant appears.",
        "",
        "| " + " | ".join(cross_headers) + " |",
        "|---------|--------------|-------------------|-----------------|---------------------|-------|",
    ]
    for variant, mean_rho, mono, recon, fp, cells_count in summary_rows:
        lines.append(
            f"| {variant} | {fmt(mean_rho)} | {fmt(mono)} | {fmt(recon)} | {fmt(fp)} | {cells_count} |"
        )
    md_out = base_output_dir / "CROSS_DATASET_ABLATION.md"
    md_out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {md_out}")


@dataclass
class _FingerprintRow:
    variant: str
    transformation: str
    autocorrelation: float
    linearity: float
    amplitude_dependence: float
    spearman_rho: float | None = None
    monotonic: float | None = None


def _load_fingerprint_rows(csv_path: Path) -> list[_FingerprintRow]:
    """Load fingerprint CSV into typed rows; returns empty list if missing.
    Supports legacy CSVs without spearman_rho/monotonic columns."""
    if not csv_path.exists():
        return []
    rows: list[_FingerprintRow] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_rho = "spearman_rho" in fieldnames
        has_mono = "monotonic" in fieldnames
        for raw in reader:
            rho = float(raw["spearman_rho"]) if has_rho and raw.get("spearman_rho") else None
            mono = float(raw["monotonic"]) if has_mono and raw.get("monotonic") else None
            rows.append(_FingerprintRow(
                variant=raw["variant"],
                transformation=raw["transformation"],
                autocorrelation=float(raw["autocorrelation"]),
                linearity=float(raw["linearity"]),
                amplitude_dependence=float(raw["amplitude_dependence"]),
                spearman_rho=rho,
                monotonic=mono,
            ))
    return rows


_SIGNATURE_METRICS: list[tuple[str, str]] = [
    ("autocorrelation", "Autocorrelation"),
    ("linearity", r"Linearity ($R^2$)"),
    ("amplitude_dependence", "Amplitude Dependence"),
]

_PRETTY_TRANSFORM: dict[str, str] = {
    "jitter": "Jitter",
    "scaling": "Scaling",
    "magnitude_warp": "Mag. Warp",
    "drift": "Drift",
    "trend": "Trend",
}


def _compute_metric_mean_std(
    dataset_fingerprints: list[tuple[str, str, list[_FingerprintRow]]],
    lgta_variant: str,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a fingerprint metric across datasets for L-GTA vs Direct.

    Returns (mean_lgta, std_lgta, mean_direct, std_direct) as arrays ordered
    according to ALL_TRANSFORMATIONS.
    """
    transformations = list(ALL_TRANSFORMATIONS)
    lgta_means: list[float] = []
    lgta_stds: list[float] = []
    direct_means: list[float] = []
    direct_stds: list[float] = []

    for t in transformations:
        lgta_vals: list[float] = []
        direct_vals: list[float] = []
        for _, _, rows in dataset_fingerprints:
            by_key = {(r.variant, r.transformation): r for r in rows}
            lgta_row = by_key.get((lgta_variant, t))
            direct_row = by_key.get(("Direct", t))
            if lgta_row is not None:
                val = getattr(lgta_row, metric_key)
                if val is not None:
                    lgta_vals.append(float(val))
            if direct_row is not None:
                val = getattr(direct_row, metric_key)
                if val is not None:
                    direct_vals.append(float(val))
        if lgta_vals:
            lgta_arr = np.array(lgta_vals, dtype=float)
            lgta_means.append(float(lgta_arr.mean()))
            lgta_stds.append(float(lgta_arr.std(ddof=0)))
        else:
            lgta_means.append(0.0)
            lgta_stds.append(0.0)
        if direct_vals:
            direct_arr = np.array(direct_vals, dtype=float)
            direct_means.append(float(direct_arr.mean()))
            direct_stds.append(float(direct_arr.std(ddof=0)))
        else:
            direct_means.append(0.0)
            direct_stds.append(0.0)

    return (
        np.array(lgta_means, dtype=float),
        np.array(lgta_stds, dtype=float),
        np.array(direct_means, dtype=float),
        np.array(direct_stds, dtype=float),
    )


def plot_cross_dataset_signature_summary(
    base_output_dir: Path,
    datasets: list[tuple[str, str]],
    lgta_variant: str = "G: Full L-GTA, no dynamic",
    output_filename: str = "CROSS_DATASET_SIGNATURE_SUMMARY.png",
) -> None:
    """Create a single figure comparing L-GTA vs Direct across all datasets.

    Layout: 3 rows (metrics) x N columns (datasets).  Each cell shows paired
    bars per transformation.
    """
    dataset_fingerprints: list[tuple[str, str, list[_FingerprintRow]]] = []
    for dataset_name, freq in datasets:
        csv_path = base_output_dir / dataset_name / freq / "fingerprint_data.csv"
        rows = _load_fingerprint_rows(csv_path)
        if rows:
            dataset_fingerprints.append((dataset_name, freq, rows))

    if not dataset_fingerprints:
        print("No fingerprint CSVs found; skipping cross-dataset signature summary.")
        return

    n_datasets = len(dataset_fingerprints)
    n_metrics = len(_SIGNATURE_METRICS)
    fig, axes = plt.subplots(
        n_metrics, n_datasets,
        figsize=(4.0 * n_datasets, 3.0 * n_metrics),
        sharey="row",
        squeeze=False,
    )

    bar_width = 0.32
    transformations = list(ALL_TRANSFORMATIONS)
    x = np.arange(len(transformations))

    for col_idx, (ds_name, freq, rows) in enumerate(dataset_fingerprints):
        lgta_rows = [r for r in rows if r.variant == lgta_variant]
        direct_rows = [r for r in rows if r.variant == "Direct"]

        lgta_by_t = {r.transformation: r for r in lgta_rows}
        direct_by_t = {r.transformation: r for r in direct_rows}

        for row_idx, (metric_key, metric_label) in enumerate(_SIGNATURE_METRICS):
            ax = axes[row_idx][col_idx]

            lgta_vals = [
                getattr(lgta_by_t[t], metric_key, 0.0)
                if t in lgta_by_t else 0.0
                for t in transformations
            ]
            direct_vals = [
                getattr(direct_by_t[t], metric_key, 0.0)
                if t in direct_by_t else 0.0
                for t in transformations
            ]

            ax.bar(
                x - bar_width / 2, lgta_vals, bar_width,
                label="L-GTA", color="#2196F3", edgecolor="white", linewidth=0.5,
            )
            ax.bar(
                x + bar_width / 2, direct_vals, bar_width,
                label="Direct", color="#9E9E9E", alpha=0.75,
                edgecolor="white", linewidth=0.5,
            )

            tick_labels = [_PRETTY_TRANSFORM.get(t, t) for t in transformations]
            ax.set_xticks(x)
            ax.set_xticklabels(
                tick_labels, rotation=30, ha="right", fontsize=8,
            )
            ax.grid(True, alpha=0.15, axis="y")
            ax.tick_params(axis="y", labelsize=8)

            if row_idx == 0:
                ax.set_title(
                    f"{ds_name.upper()} ({freq})",
                    fontsize=11, fontweight="bold", pad=6,
                )
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=10)
            if row_idx == 0 and col_idx == n_datasets - 1:
                ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Transformation Signature: L-GTA vs Direct",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = base_output_dir / output_filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def _draw_paired_bars(
    ax: plt.Axes,
    x: np.ndarray,
    lgta_mean: np.ndarray,
    lgta_std: np.ndarray,
    direct_mean: np.ndarray,
    direct_std: np.ndarray,
    tick_labels: list[str],
    ylabel: str,
    bar_width: float = 0.35,
    show_legend: bool = False,
) -> None:
    """Draw paired L-GTA vs Direct bars with error bars on a given axes."""
    ax.bar(
        x - bar_width / 2, lgta_mean, bar_width,
        yerr=lgta_std, label="L-GTA", color="#2196F3",
        edgecolor="white", linewidth=0.5, alpha=0.9, capsize=3,
    )
    ax.bar(
        x + bar_width / 2, direct_mean, bar_width,
        yerr=direct_std, label="Direct", color="#9E9E9E",
        edgecolor="white", linewidth=0.5, alpha=0.8, capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")
    if show_legend:
        ax.legend(fontsize=8)


def plot_cross_dataset_signature_summary_averaged(
    base_output_dir: Path,
    datasets: list[tuple[str, str]],
    lgta_variant: str = "G: Full L-GTA, no dynamic",
    output_filename: str = "CROSS_DATASET_SIGNATURE_MONO_SUMMARY.png",
) -> None:
    """Create a 5-panel figure (3 top + 2 bottom) with dataset-averaged
    signatures and controllability.

    Top row: autocorrelation, linearity, amplitude dependence.
    Bottom row: Spearman rho, Monotonic %.
    All averaged over datasets with standard deviation error bars.
    """
    from matplotlib.gridspec import GridSpec

    dataset_fingerprints: list[tuple[str, str, list[_FingerprintRow]]] = []
    for dataset_name, freq in datasets:
        csv_path = base_output_dir / dataset_name / freq / "fingerprint_data.csv"
        rows = _load_fingerprint_rows(csv_path)
        if rows:
            dataset_fingerprints.append((dataset_name, freq, rows))

    if not dataset_fingerprints:
        print("No fingerprint CSVs found; skipping averaged cross-dataset signature summary.")
        return

    transformations = list(ALL_TRANSFORMATIONS)
    x = np.arange(len(transformations), dtype=float)
    tick_labels = [_PRETTY_TRANSFORM.get(t, t) for t in transformations]

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 6, figure=fig, hspace=0.45, wspace=0.55)
    ax_auto = fig.add_subplot(gs[0, 0:2])
    ax_lin = fig.add_subplot(gs[0, 2:4])
    ax_amp = fig.add_subplot(gs[0, 4:6])
    ax_rho = fig.add_subplot(gs[1, 1:3])
    ax_mono = fig.add_subplot(gs[1, 3:5])

    panels: list[tuple[plt.Axes, str, str]] = [
        (ax_auto, "autocorrelation", "Autocorrelation"),
        (ax_lin, "linearity", r"Linearity ($R^2$)"),
        (ax_amp, "amplitude_dependence", "Amplitude Dependence"),
    ]
    for i, (ax, metric_key, ylabel) in enumerate(panels):
        lm, ls, dm, ds = _compute_metric_mean_std(
            dataset_fingerprints, lgta_variant, metric_key,
        )
        _draw_paired_bars(
            ax, x, lm, ls, dm, ds, tick_labels, ylabel,
            show_legend=(i == 0),
        )

    # Spearman rho
    lm_rho, ls_rho, dm_rho, ds_rho = _compute_metric_mean_std(
        dataset_fingerprints, lgta_variant, "spearman_rho",
    )
    has_rho = np.any(lm_rho != 0) or np.any(dm_rho != 0)
    if has_rho:
        _draw_paired_bars(
            ax_rho, x, lm_rho, ls_rho, dm_rho, ds_rho,
            tick_labels, r"Spearman $\rho$",
        )
    else:
        ax_rho.text(
            0.5, 0.5,
            "No ρ data.\nRe-run per-dataset\nablation to populate.",
            ha="center", va="center", fontsize=9, color="grey",
            transform=ax_rho.transAxes,
        )
        ax_rho.set_ylabel(r"Spearman $\rho$", fontsize=10)

    # Monotonic %
    lm_mono, ls_mono, dm_mono, ds_mono = _compute_metric_mean_std(
        dataset_fingerprints, lgta_variant, "monotonic",
    )
    has_mono = np.any(lm_mono != 0) or np.any(dm_mono != 0)
    if has_mono:
        _draw_paired_bars(
            ax_mono, x,
            lm_mono * 100.0, ls_mono * 100.0,
            dm_mono * 100.0, ds_mono * 100.0,
            tick_labels, "Monotonic (%)",
        )
        ax_mono.set_ylim(0.0, 105.0)
    else:
        ax_mono.text(
            0.5, 0.5,
            "No monotonic data.\nRe-run per-dataset\nablation to populate.",
            ha="center", va="center", fontsize=9, color="grey",
            transform=ax_mono.transAxes,
        )
        ax_mono.set_ylabel("Monotonic (%)", fontsize=10)

    fig.suptitle(
        "Transformation Signature & Controllability (avg ± std across datasets)",
        fontsize=13, fontweight="bold",
    )
    out = base_output_dir / output_filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


FULL_LGTA_CONFIG_NAME = "D: Full L-GTA"
FULL_LGTA_NODYN_CONFIG_NAME = "G: Full L-GTA, no dynamic"
COMPARISON_SIGMA = 2.0
BENCHMARK_SEED = 42

# 3var: same as downstream_forecasting with --variant-transformations jitter scaling magnitude_warp.
# One variant per transformation for L-GTA and Direct; 3 samples for other benchmarks.
COMPARISON_N_VARIANTS = 3
COMPARISON_VARIANT_TRANSFORMATIONS: list[str] = [
    "jitter",
    "scaling",
    "magnitude_warp",
]

# Reuse benchmark weights from downstream_forecasting when present (same path convention).
DOWNSTREAM_CACHE_ROOT = Path("assets/cache/downstream_forecasting")
DOWNSTREAM_LGTA_WEIGHTS_SUFFIX = "eq1_0_nodyn"
DOWNSTREAM_LGTA_WINDOW_SIZE = 10
DOWNSTREAM_LGTA_LATENT_DIM = 4


def _flatten_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    """Flatten nested {category: {k: v}} to {category.k: v} for table columns."""
    out: dict[str, float] = {}
    for cat, vals in metrics.items():
        for k, v in vals.items():
            out[f"{cat}.{k}"] = float(v) if np.isscalar(v) else float("nan")
    return out


def _downstream_benchmark_weights_dir(dataset_name: str, freq: str) -> Path:
    """Weights dir for benchmarks; same convention as downstream_forecasting (load/save here)."""
    return DOWNSTREAM_CACHE_ROOT / f"{dataset_name}_{freq}"


def _downstream_lgta_weights_dir(dataset_name: str, freq: str) -> Path:
    """Weights dir for LGTA model; same as downstream_forecasting cache/model_weights."""
    return DOWNSTREAM_CACHE_ROOT / f"{dataset_name}_{freq}" / "model_weights"


def _try_load_downstream_lgta(
    dataset_name: str,
    freq: str,
) -> tuple | None:
    """Load full L-GTA (no dynamic, 3var config) from downstream_forecasting cache if present.

    Returns (model, z_mean, vae_creator) or None if weights are not found.
    """
    weights_dir = _downstream_lgta_weights_dir(dataset_name, freq)
    if not weights_dir.exists():
        return None
    creator = CreateTransformedVersionsCVAE(
        dataset_name=dataset_name,
        freq=freq,
        window_size=DOWNSTREAM_LGTA_WINDOW_SIZE,
        weights_suffix=DOWNSTREAM_LGTA_WEIGHTS_SUFFIX,
        weights_dir=weights_dir,
        use_dynamic_features=False,
        device=torch.device("cpu"),
    )
    dynamic_features_np, X_inp_np = creator._feature_engineering(creator.n_train)
    n_main_features = X_inp_np.shape[-1]
    weights_file = Path(
        creator.weights_dir
        / f"{creator.dataset_name}_n{n_main_features}_w{creator.window_size}_l{DOWNSTREAM_LGTA_LATENT_DIM}_vae_weights_{DOWNSTREAM_LGTA_WEIGHTS_SUFFIX}.pt"
    )
    if not weights_file.exists():
        return None
    model, _, _ = creator.fit(
        epochs=1,
        latent_dim=DOWNSTREAM_LGTA_LATENT_DIM,
        load_weights=True,
        equiv_weight=1.0,
        latent_mode=LatentMode.TEMPORAL,
    )
    _, _, z_mean, _ = creator.predict(model, detemporalize_method="mean")
    return (model, z_mean, creator)


def _benchmark_weights_path(weights_dir: Path, gen: TimeSeriesGenerator) -> Path:
    """Path for one generator's weights; matches downstream_forecasting._weights_path."""
    return weights_dir / f"{gen.__class__.__name__}_weights.pt"


def write_comparison_table_md(
    run_result: AblationRunResult,
    configs: list[ComponentAblationConfig],
    output_dir: Path,
    dataset_name: str,
    freq: str,
    n_benchmark_samples: int | None = None,
) -> None:
    """Write comparison_results.md: Full L-GTA (3var), Direct (3var), TimeGAN, TimeVAE, Diffusion-TS.

    Uses 3var setup aligned with downstream_forecasting: one variant per transformation
    (jitter, scaling, magnitude_warp) at σ=2 for L-GTA and Direct; 3 samples for benchmarks.
    Benchmark models are loaded from downstream_forecasting cache when present (so run
    downstream_forecasting first to train them, or they will be fit here and saved to that cache).
    Mean ρ and Monotonic % from ablation (L-GTA and Direct only). Benchmarks are evaluated
    by generating samples and comparing to original only (no transformation/σ-sweep), so
    those columns show — for benchmarks. All methods get pymdma metrics from metrics.py.
    """
    if n_benchmark_samples is None:
        n_benchmark_samples = COMPARISON_N_VARIANTS
    output_dir.mkdir(parents=True, exist_ok=True)
    X_orig = run_result.X_orig
    valid_mask = run_result.valid_mask
    results = run_result.results
    trained_models = run_result.trained_models

    full_lgta_result = next(
        (r for r in results if r.name == FULL_LGTA_CONFIG_NAME),
        None,
    )
    if full_lgta_result is None:
        print(
            f"Warning: {FULL_LGTA_CONFIG_NAME} not in results; skipping comparison table."
        )
        return

    full_lgta_config = next(
        (c for c in configs if c.name == FULL_LGTA_CONFIG_NAME),
        None,
    )
    if full_lgta_config is None:
        print(
            f"Warning: {FULL_LGTA_CONFIG_NAME} not in configs; skipping comparison table."
        )
        return

    full_lgta_nodyn_result = next(
        (r for r in results if r.name == FULL_LGTA_NODYN_CONFIG_NAME),
        None,
    )
    full_lgta_nodyn_config = next(
        (c for c in configs if c.name == FULL_LGTA_NODYN_CONFIG_NAME),
        None,
    )

    sampling_freq = FREQ_TO_SAMPLING_FREQ.get(freq.upper().strip(), 1)
    feature_cache = DOWNSTREAM_CACHE_ROOT / f"{dataset_name}_{freq}" / "tsfel_features"
    aggregator = MetricsAggregator(
        sampling_freq=sampling_freq, cache_dir=feature_cache
    )

    downstream_lgta = _try_load_downstream_lgta(dataset_name, freq)
    if downstream_lgta is not None:
        print("  Using L-GTA weights from downstream cache (no dynamic, 3var)")

    def _lgta_model_for_dynamic() -> tuple[torch.nn.Module, np.ndarray, object]:
        if full_lgta_config.model_key in trained_models:
            model, z_mean, _, vae_creator = trained_models[full_lgta_config.model_key]
            return (model, z_mean, vae_creator)
        if downstream_lgta is not None:
            return downstream_lgta
        model, z_mean, _, vae_creator = trained_models[full_lgta_config.model_key]
        return (model, z_mean, vae_creator)

    def _lgta_model_for_nodyn() -> tuple[torch.nn.Module, np.ndarray, object] | None:
        if full_lgta_nodyn_config is not None and full_lgta_nodyn_config.model_key in trained_models:
            model, z_mean, _, vae_creator = trained_models[full_lgta_nodyn_config.model_key]
            return (model, z_mean, vae_creator)
        if downstream_lgta is not None:
            return downstream_lgta
        return None

    rng = np.random.default_rng(BENCHMARK_SEED)
    model_dyn, z_mean_dyn, vae_creator_dyn = _lgta_model_for_dynamic()
    lgta_flat_list: list[dict[str, float]] = []
    for transformation in COMPARISON_VARIANT_TRANSFORMATIONS:
        X_lgta = generate_synthetic_data(
            model_dyn,
            z_mean_dyn,
            vae_creator_dyn,
            transformation,
            [COMPARISON_SIGMA],
            latent_mode=LatentMode.TEMPORAL,
            rng=rng,
        )
        if valid_mask is not None:
            X_lgta = X_lgta * valid_mask.astype(np.float32)
        metrics = aggregator.compute_metrics_single(X_orig, X_lgta)
        lgta_flat_list.append(_flatten_metrics(metrics))
    lgta_avg_flat: dict[str, float] = {}
    if lgta_flat_list:
        for k in lgta_flat_list[0]:
            lgta_avg_flat[k] = float(np.nanmean([f[k] for f in lgta_flat_list]))

    lgta_nodyn_avg_flat: dict[str, float] = {}
    nodyn_source = _lgta_model_for_nodyn()
    if nodyn_source is not None:
        model_nodyn, z_mean_nodyn, vae_creator_nodyn = nodyn_source
        rng_nodyn = np.random.default_rng(BENCHMARK_SEED)
        lgta_nodyn_flat_list: list[dict[str, float]] = []
        for transformation in COMPARISON_VARIANT_TRANSFORMATIONS:
            X_lgta = generate_synthetic_data(
                model_nodyn,
                z_mean_nodyn,
                vae_creator_nodyn,
                transformation,
                [COMPARISON_SIGMA],
                latent_mode=LatentMode.TEMPORAL,
                rng=rng_nodyn,
            )
            if valid_mask is not None:
                X_lgta = X_lgta * valid_mask.astype(np.float32)
            metrics = aggregator.compute_metrics_single(X_orig, X_lgta)
            lgta_nodyn_flat_list.append(_flatten_metrics(metrics))
        if lgta_nodyn_flat_list:
            for k in lgta_nodyn_flat_list[0]:
                lgta_nodyn_avg_flat[k] = float(np.nanmean([f[k] for f in lgta_nodyn_flat_list]))

    X_orig_scaled = vae_creator_dyn.scaler_target.transform(X_orig)
    direct_flat_list: list[dict[str, float]] = []
    for transformation in COMPARISON_VARIANT_TRANSFORMATIONS:
        X_direct_scaled = ManipulateData(
            x=X_orig_scaled,
            transformation=transformation,
            parameters=[COMPARISON_SIGMA],
        ).apply_transf()
        X_direct = vae_creator_dyn.scaler_target.inverse_transform(X_direct_scaled)
        if valid_mask is not None:
            X_direct = X_direct * valid_mask.astype(np.float32)
        metrics = aggregator.compute_metrics_single(X_orig, X_direct)
        direct_flat_list.append(_flatten_metrics(metrics))
    direct_avg_flat: dict[str, float] = {}
    if direct_flat_list:
        for k in direct_flat_list[0]:
            direct_avg_flat[k] = float(np.nanmean([f[k] for f in direct_flat_list]))

    synthetic_by_method: dict[str, dict[str, float]] = {}
    if lgta_avg_flat:
        synthetic_by_method[f"Full L-GTA (3var, σ={COMPARISON_SIGMA})"] = lgta_avg_flat
    if lgta_nodyn_avg_flat:
        synthetic_by_method[f"Full L-GTA, no dynamic (3var, σ={COMPARISON_SIGMA})"] = lgta_nodyn_avg_flat
    if direct_avg_flat:
        synthetic_by_method[f"Direct (3var, σ={COMPARISON_SIGMA})"] = direct_avg_flat

    benchmark_weights_dir = _downstream_benchmark_weights_dir(dataset_name, freq)
    benchmark_metrics_avg: dict[str, dict[str, float]] = {}
    for gen in get_default_benchmark_generators(seed=BENCHMARK_SEED):
        if isinstance(gen, DirectTransformGenerator):
            continue
        try:
            wp = _benchmark_weights_path(benchmark_weights_dir, gen)
            if gen.load_weights(wp):
                print(f"  Loaded {gen.name} weights from downstream cache")
            else:
                gen.fit(X_orig)
                benchmark_weights_dir.mkdir(parents=True, exist_ok=True)
                gen.save_weights(wp)
            flat_list: list[dict[str, float]] = []
            for _ in range(n_benchmark_samples):
                X_syn = gen.generate()
                metrics = aggregator.compute_metrics_single(X_orig, X_syn)
                flat_list.append(_flatten_metrics(metrics))
            keys = flat_list[0].keys() if flat_list else []
            avg_flat: dict[str, float] = {}
            for k in keys:
                vals = [f[k] for f in flat_list]
                avg_flat[k] = float(np.nanmean(vals))
            benchmark_metrics_avg[f"{gen.name} (3var)"] = avg_flat
        except Exception as e:
            print(f"Warning: {gen.name} fit/generate failed: {e}")

    direct_rho_mean = float(
        np.mean([
            tr.direct_spearman_rho
            for tr in results[0].transformation_results.values()
        ])
    )
    direct_mono_pct = float(
        np.mean([
            tr.direct_is_monotonic
            for tr in results[0].transformation_results.values()
        ])
    ) * 100.0
    full_lgta_rho_mean = float(
        np.mean([tr.spearman_rho for tr in full_lgta_result.transformation_results.values()])
    )
    full_lgta_mono_pct = float(
        np.mean([
            tr.is_monotonic
            for tr in full_lgta_result.transformation_results.values()
        ])
    ) * 100.0
    if full_lgta_nodyn_result is not None:
        full_lgta_nodyn_rho_mean = float(
            np.mean([tr.spearman_rho for tr in full_lgta_nodyn_result.transformation_results.values()])
        )
        full_lgta_nodyn_mono_pct = float(
            np.mean([
                tr.is_monotonic
                for tr in full_lgta_nodyn_result.transformation_results.values()
            ])
        ) * 100.0
    else:
        full_lgta_nodyn_rho_mean = float("nan")
        full_lgta_nodyn_mono_pct = float("nan")

    method_order = [
        f"Full L-GTA (3var, σ={COMPARISON_SIGMA})",
        f"Full L-GTA, no dynamic (3var, σ={COMPARISON_SIGMA})",
        f"Direct (3var, σ={COMPARISON_SIGMA})",
        "TimeGANGenerator (3var)",
        "TimeVAEGenerator (3var)",
        "DiffusionTSGenerator (3var)",
    ]
    rows: list[dict[str, str | float]] = []
    all_metric_keys: list[str] = []
    for method in method_order:
        if method in synthetic_by_method:
            flat = synthetic_by_method[method]
            if not all_metric_keys:
                all_metric_keys = sorted(flat.keys())
            if method == f"Full L-GTA, no dynamic (3var, σ={COMPARISON_SIGMA})":
                mean_rho = full_lgta_nodyn_rho_mean if not np.isnan(full_lgta_nodyn_rho_mean) else "—"
                mono_pct = full_lgta_nodyn_mono_pct if not np.isnan(full_lgta_nodyn_mono_pct) else "—"
            elif "Full L-GTA" in method:
                mean_rho = full_lgta_rho_mean
                mono_pct = full_lgta_mono_pct
            else:
                mean_rho = direct_rho_mean
                mono_pct = direct_mono_pct
        elif method in benchmark_metrics_avg:
            flat = benchmark_metrics_avg[method]
            if not all_metric_keys:
                all_metric_keys = sorted(flat.keys())
            mean_rho: str | float = "—"
            mono_pct: str | float = "—"
        else:
            continue
        row: dict[str, str | float] = {
            "Method": method,
            "Mean ρ": mean_rho,
            "Monotonic %": mono_pct,
        }
        for k in all_metric_keys:
            row[k] = flat.get(k, float("nan"))
        rows.append(row)

    DISTANCE_METRIC_KEYS = frozenset({
        "fidelity.frechet_distance",
        "fidelity.wasserstein_distance",
        "fidelity.mmd",
    })
    distance_keys_in_table = [k for k in all_metric_keys if k in DISTANCE_METRIC_KEYS]
    min_max: dict[str, tuple[float, float]] = {}
    for k in distance_keys_in_table:
        values = [
            row[k] for row in rows
            if isinstance(row.get(k), (int, float))
            and not np.isnan(float(row[k]))
        ]
        if values:
            min_max[k] = (min(values), max(values))

    header = ["Method", "Mean ρ", "Monotonic %"] + all_metric_keys
    csv_path = output_dir / "comparison_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            csv_row: dict[str, str] = {}
            for h in header:
                v = row.get(h, "")
                if v == "—":
                    csv_row[h] = ""
                elif isinstance(v, float):
                    if np.isnan(v):
                        csv_row[h] = ""
                    elif h == "Monotonic %":
                        csv_row[h] = f"{v:.4f}"
                    elif h in min_max:
                        lo, hi = min_max[h]
                        if hi <= lo:
                            norm = 0.0
                        else:
                            norm = (float(v) - lo) / (hi - lo)
                        csv_row[h] = f"{norm:.4f}"
                    else:
                        csv_row[h] = f"{v:.4f}"
                else:
                    csv_row[h] = str(v)
            writer.writerow(csv_row)
    print(f"Saved: {csv_path}")
    lines = [
        "# Comparison: Full L-GTA vs benchmarks (all methods 3var)",
        "",
        "**All methods 3var** (aligned with downstream_forecasting): "
        "L-GTA and Direct use 3 variants (one per transformation: "
        f"{', '.join(COMPARISON_VARIANT_TRANSFORMATIONS)}) at σ={COMPARISON_SIGMA}; "
        "TimeGAN, TimeVAE, Diffusion-TS use 3 samples each. "
        "Mean ρ / Monotonic % from ablation (L-GTA and Direct only); benchmarks have no σ-sweep (generate vs original only), so —. "
        "Metrics averaged over variants/samples. "
        "**Distance metrics** (Fr\u00e9chet, Wasserstein, MMD) are min\u2013max normalized per dataset (0 = best, 1 = worst).",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        cells: list[str] = []
        for h in header:
            v = row.get(h, "")
            if v == "—":
                cells.append("—")
            elif isinstance(v, float):
                if np.isnan(v):
                    cells.append("—")
                elif h == "Monotonic %":
                    cells.append(f"{v:.1f}%")
                elif h in min_max:
                    lo, hi = min_max[h]
                    if hi <= lo:
                        norm = 0.0
                    else:
                        norm = (float(v) - lo) / (hi - lo)
                    cells.append(f"{norm:.4f}")
                else:
                    cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    md_path = output_dir / "comparison_results.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {md_path}")


def write_cross_dataset_comparison_summary(
    base_output_dir: Path,
    datasets: list[tuple[str, str]],
) -> None:
    """Aggregate comparison_results.csv across datasets and write a cross-dataset summary.

    Averages each metric per method over all (dataset, freq) pairs where it appears.
    """
    method_to_metrics: dict[str, dict[str, list[float]]] = {}
    all_metric_keys: set[str] = set()

    for dataset_name, freq in datasets:
        dir_path = base_output_dir / dataset_name / freq
        rows = _load_csv_rows(dir_path / "comparison_results.csv")
        if not rows:
            continue
        # Header inferred from first data row; keys we care about are numeric.
        for row in rows:
            method = row.get("Method")
            if not method:
                continue
            metrics = method_to_metrics.setdefault(method, {})
            for key, value in row.items():
                if key in {"Method"}:
                    continue
                val = _to_float(value)
                if val is None:
                    continue
                metrics.setdefault(key, []).append(val)
                all_metric_keys.add(key)

    if not method_to_metrics:
        return

    ordered_methods = sorted(method_to_metrics.keys())
    ordered_metrics = sorted(all_metric_keys)
    csv_headers = ["Method"] + [f"{k} (avg)" for k in ordered_metrics]

    csv_out = base_output_dir / "CROSS_DATASET_COMPARISON.csv"
    with csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_headers)
        w.writeheader()
        for method in ordered_methods:
            metrics = method_to_metrics[method]
            row: dict[str, str] = {"Method": method}
            for key in ordered_metrics:
                vals = metrics.get(key, [])
                if not vals:
                    row[f"{key} (avg)"] = ""
                else:
                    avg_val = sum(vals) / len(vals)
                    row[f"{key} (avg)"] = f"{avg_val:.4f}"
            w.writerow(row)
    print(f"Saved: {csv_out}")

    normalized_metrics = (
        "fidelity.frechet_distance",
        "fidelity.wasserstein_distance",
        "fidelity.mmd",
    )
    higher_better = (
        "Mean ρ",
        "Monotonic %",
        "diversity.coverage",
        "diversity.improved_recall",
        "fidelity.cosine_similarity",
        "fidelity.density",
        "fidelity.improved_precision",
        "privacy.authenticity",
    )
    lines: list[str] = [
        "# Cross-dataset Metrics Comparison Summary",
        "",
        "Metrics are averaged over all (dataset, freq) pairs where a method appears.",
        "",
        "**Normalized metrics (min–max per dataset, 0 = best, 1 = worst):** "
        + f"`{'`, `'.join(normalized_metrics)}`. All other columns are raw (pymdma or computed values).",
        "",
        "**Direction (what is better):**",
        "- **Higher is better:** " + ", ".join(higher_better) + ".",
        "- **Lower is better:** "
        + ", ".join(normalized_metrics)
        + " (after normalization: 0 = best, 1 = worst).",
        "",
        "| " + " | ".join(csv_headers) + " |",
        "|--------|" + "|".join(["-----------"] * len(ordered_metrics)) + "|",
    ]
    for method in ordered_methods:
        metrics = method_to_metrics[method]
        cells: list[str] = [method]
        for key in ordered_metrics:
            vals = metrics.get(key, [])
            if not vals:
                cells.append("—")
            else:
                avg_val = sum(vals) / len(vals)
                if "Monotonic %" in key:
                    cells.append(f"{avg_val:.1f}%")
                else:
                    cells.append(f"{avg_val:.4f}")
        lines.append("| " + " | ".join(cells) + " |")

    md_out = base_output_dir / "CROSS_DATASET_COMPARISON.md"
    md_out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {md_out}")


def plot_component_ablation(
    results: list[ComponentAblationResult],
    sigma_values: list[float],
    output_dir: Path,
) -> None:
    """Generate ablation plots: monotonic response grid, signature bars,
    reconstruction-controllability scatter, and ablation results table (MD)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_monotonic_response_grid(results, sigma_values, output_dir)
    _plot_signature_comparison(results, output_dir)
    _plot_recon_vs_controllability(results, output_dir)
    write_ablation_table_md(results, sigma_values, output_dir)
    _save_fingerprint_csv(results, output_dir)


def _plot_monotonic_response_grid(
    results: list[ComponentAblationResult],
    sigma_values: list[float],
    output_dir: Path,
) -> None:
    n_variants = len(results)
    fig, axes = plt.subplots(1, n_variants, figsize=(5 * n_variants, 5), sharey=True)
    if n_variants == 1:
        axes = [axes]

    sigma = np.array(sigma_values)
    for ax, r in zip(axes, results):
        for transf, tr in r.transformation_results.items():
            ax.plot(sigma, tr.mean_distances, "o-", linewidth=1.5,
                    label=f"{transf} (rho={tr.spearman_rho:.2f})")
            ax.fill_between(sigma,
                            tr.mean_distances - tr.std_distances,
                            tr.mean_distances + tr.std_distances,
                            alpha=0.08)
        ax.set_xlabel("sigma", fontsize=11)
        ax.set_title(r.name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Wasserstein Distance", fontsize=11)

    fig.suptitle("Monotonic Response by Ablation Variant",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "monotonic_response_grid.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def _plot_signature_comparison(
    results: list[ComponentAblationResult],
    output_dir: Path,
) -> None:
    metrics = ["autocorrelation", "linearity", "amplitude_dependence"]
    labels = ["Autocorrelation", "Linearity (R^2)", "Amplitude Dep."]

    for metric, label in zip(metrics, labels):
        fig, ax = plt.subplots(figsize=(10, 5))
        n_transf = len(ALL_TRANSFORMATIONS)
        x = np.arange(n_transf)
        bar_width = 0.8 / (len(results) + 1)

        for i, r in enumerate(results):
            vals = [
                getattr(r.transformation_results[t].fingerprint, metric)
                for t in ALL_TRANSFORMATIONS
            ]
            ax.bar(x + i * bar_width, vals, bar_width, label=r.name)

        direct_vals = [
            getattr(results[0].transformation_results[t].direct_fingerprint, metric)
            for t in ALL_TRANSFORMATIONS
        ]
        ax.bar(x + len(results) * bar_width, direct_vals, bar_width,
               label="Direct", color="gray", alpha=0.6)

        ax.set_xticks(x + bar_width * len(results) / 2)
        ax.set_xticklabels(ALL_TRANSFORMATIONS, rotation=20, ha="right")
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

        plt.tight_layout()
        out = output_dir / f"signature_{metric}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close()


def _plot_recon_vs_controllability(
    results: list[ComponentAblationResult],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for r in results:
        rho_vals = [tr.spearman_rho for tr in r.transformation_results.values()]
        mean_rho = float(np.mean(rho_vals))
        marker = "o" if r.latent_mode == LatentMode.TEMPORAL else "s"
        color = "#2196F3" if r.equiv_weight > 0 else "#F44336"
        ax.scatter(r.recon_mse, mean_rho, s=150, marker=marker,
                   color=color, zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(r.name, (r.recon_mse, mean_rho),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Reconstruction MSE", fontsize=12)
    ax.set_ylabel("Mean Spearman rho (controllability)", fontsize=12)
    ax.set_title("Reconstruction vs Controllability",
                 fontsize=14, fontweight="bold")
    ax.axhline(y=1.0, color="green", linestyle=":", alpha=0.4, label="Perfect rho=1")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "recon_vs_controllability.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def print_component_ablation_report(
    results: list[ComponentAblationResult],
) -> None:
    """Print a publication-ready ablation table."""
    print("\n" + "=" * 100)
    print("COMPONENT ABLATION STUDY REPORT")
    print("=" * 100)

    header = (
        f"{'Variant':<28s}  {'Latent':<9s}  {'Equiv':<6s}  {'Dyn':<4s}  "
        f"{'ReconMSE':>9s}  {'MeanRho':>8s}  {'Mono%':>6s}  "
        f"{'MeanFPDist':>10s}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        rhos = [tr.spearman_rho for tr in r.transformation_results.values()]
        monos = [tr.is_monotonic for tr in r.transformation_results.values()]
        fp_dists = [tr.fingerprint_distance for tr in r.transformation_results.values()]
        mono_pct = np.mean(monos) * 100.0
        dyn_str = "Y" if r.use_dynamic_features else "N"

        print(
            f"{r.name:<28s}  {r.latent_mode.value:<9s}  "
            f"{r.equiv_weight:<6.1f}  {dyn_str:<4s}  {r.recon_mse:9.4f}  "
            f"{np.mean(rhos):8.4f}  {mono_pct:5.1f}%  "
            f"{np.mean(fp_dists):10.4f}"
        )

    print("=" * 100)

    print("\nPer-transformation breakdown:")
    print("-" * 100)
    header2 = (
        f"{'Variant':<22s}  {'Transform':<16s}  "
        f"{'rho':>6s}  {'Mono':>5s}  "
        f"{'AutoCorr(L)':>11s}  {'AutoCorr(D)':>11s}  "
        f"{'Linear(L)':>10s}  {'Linear(D)':>10s}  "
        f"{'FPDist':>7s}"
    )
    print(header2)
    print("-" * 100)
    for r in results:
        for t, tr in r.transformation_results.items():
            mono_str = "Y" if tr.is_monotonic else "N"
            print(
                f"{r.name:<22s}  {t:<16s}  "
                f"{tr.spearman_rho:6.3f}  {mono_str:>5s}  "
                f"{tr.fingerprint.autocorrelation:11.4f}  "
                f"{tr.direct_fingerprint.autocorrelation:11.4f}  "
                f"{tr.fingerprint.linearity:10.4f}  "
                f"{tr.direct_fingerprint.linearity:10.4f}  "
                f"{tr.fingerprint_distance:7.3f}"
            )
    print("=" * 100)


STANDARD_CONFIGS: list[ComponentAblationConfig] = [
    ComponentAblationConfig(
        name="A: Global, No Equiv",
        latent_mode=LatentMode.GLOBAL,
        equiv_weight=0.0,
    ),
    ComponentAblationConfig(
        name="B: Temporal, No Equiv",
        latent_mode=LatentMode.TEMPORAL,
        equiv_weight=0.0,
    ),
    ComponentAblationConfig(
        name="C: Global + Equiv",
        latent_mode=LatentMode.GLOBAL,
        equiv_weight=1.0,
    ),
    ComponentAblationConfig(
        name="D: Full L-GTA",
        latent_mode=LatentMode.TEMPORAL,
        equiv_weight=1.0,
    ),
    ComponentAblationConfig(
        name="E: Simple Enc (temporal)",
        latent_mode=LatentMode.TEMPORAL,
        equiv_weight=0.0,
        encoder_type=EncoderType.SIMPLE,
    ),
    ComponentAblationConfig(
        name="F: Full + channel attn",
        latent_mode=LatentMode.TEMPORAL,
        equiv_weight=0.0,
        use_channel_attention=True,
    ),
    ComponentAblationConfig(
        name="G: Full L-GTA, no dynamic",
        latent_mode=LatentMode.TEMPORAL,
        equiv_weight=1.0,
        use_dynamic_features=False,
    ),
    ComponentAblationConfig(
        name="H: Channel attn, no dynamic",
        latent_mode=LatentMode.TEMPORAL,
        equiv_weight=0.0,
        use_channel_attention=True,
        use_dynamic_features=False,
    ),
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run component ablation study for L-GTA (optionally for one or all datasets)."
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
        help="Time series frequency (e.g. Q, D, M). Default: from DEFAULT_ABLATION_DATASETS for the chosen dataset.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run ablation for all supported datasets with their default frequencies.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/results/component_ablation"),
        help="Base directory for results; each (dataset, freq) gets a subfolder (e.g. output_dir/tourism/Q/, output_dir/m3/Y/).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable loading/saving of model weights and transformation results; train and evaluate from scratch.",
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Only run the comparison table step (TSFEL/pymdma metrics and comparison_results.csv/.md). Loads ablation from cache; use after ablation has already been run.",
    )
    parser.add_argument(
        "--summaries-only",
        action="store_true",
        help="Only write cross-dataset summaries (CROSS_DATASET_ABLATION and CROSS_DATASET_COMPARISON) from existing per-dataset CSV files. No training or per-dataset runs.",
    )
    args = parser.parse_args()

    base_output_dir = args.output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    if args.summaries_only:
        write_cross_dataset_ablation_summary(base_output_dir, DEFAULT_ABLATION_DATASETS)
        write_cross_dataset_comparison_summary(base_output_dir, DEFAULT_ABLATION_DATASETS)
        plot_cross_dataset_signature_summary(base_output_dir, DEFAULT_ABLATION_DATASETS)
        plot_cross_dataset_signature_summary_averaged(base_output_dir, DEFAULT_ABLATION_DATASETS)
        raise SystemExit(0)

    CACHE_ABLATION_ROOT.mkdir(parents=True, exist_ok=True)

    def run_one(dataset_name: str, freq: str) -> None:
        dataset_output_dir = base_output_dir / dataset_name / freq
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        weights_dir = CACHE_ABLATION_ROOT / dataset_name / freq / "model_weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}")
        print(f"Component ablation: dataset={dataset_name}, freq={freq}")
        print(f"  Output: {dataset_output_dir}")
        print(f"  Weights: {weights_dir}")
        print("="*70)
        run_result = run_component_ablation(
            STANDARD_CONFIGS,
            dataset_name=dataset_name,
            freq=freq,
            plots_output_dir=None if args.comparison_only else dataset_output_dir,
            weights_dir=weights_dir,
            plot_per_sigma=not args.comparison_only,
            use_cache=not args.no_cache,
        )
        if not args.comparison_only:
            plot_component_ablation(
                run_result.results,
                STANDARD_CONFIGS[0].sigma_values,
                dataset_output_dir,
            )
            print_component_ablation_report(run_result.results)
        write_comparison_table_md(
            run_result,
            STANDARD_CONFIGS,
            dataset_output_dir,
            dataset_name=dataset_name,
            freq=freq,
        )

    if args.all_datasets:
        for dataset_name, freq in DEFAULT_ABLATION_DATASETS:
            run_one(dataset_name, freq)
        write_cross_dataset_ablation_summary(base_output_dir, DEFAULT_ABLATION_DATASETS)
        write_cross_dataset_comparison_summary(base_output_dir, DEFAULT_ABLATION_DATASETS)
        plot_cross_dataset_signature_summary(base_output_dir, DEFAULT_ABLATION_DATASETS)
        plot_cross_dataset_signature_summary_averaged(base_output_dir, DEFAULT_ABLATION_DATASETS)
    else:
        dataset_name = args.dataset if args.dataset is not None else "tourism"
        freq = args.freq
        if freq is None:
            freq = next(
                (f for d, f in DEFAULT_ABLATION_DATASETS if d == dataset_name),
                "Q",
            )
        run_one(dataset_name, freq)
