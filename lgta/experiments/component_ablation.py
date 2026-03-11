"""
Component ablation study for L-GTA. Isolates the contribution of the
two key innovations (temporal latent space, equivariant decoder training)
using a 2x2 matrix plus an encoder-type axis. Measures controllability,
reconstruction quality, and transformation signature preservation.
"""

from __future__ import annotations

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
from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from lgta.model.generate_data import generate_synthetic_data
from lgta.model.models import EncoderType, LatentMode
from lgta.transformations.manipulate_data import ManipulateData

DEFAULT_ABLATION_DATASETS: list[tuple[str, str]] = [
    ("tourism", "Q"),
    ("wiki2", "D"),
    ("labour", "M"),
    ("m3", "Q"),
]

CACHE_ABLATION_ROOT = Path("assets/cache/component_ablation")


def _results_cache_dir(dataset_name: str) -> Path:
    return CACHE_ABLATION_ROOT / dataset_name / "results"


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
    latent_dim: int = 4
    kl_weight_max: float = 0.1
    kl_anneal_epochs: int = 30
    epochs: int = 1000
    sigma_values: list[float] = field(
        default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0]
    )
    n_repetitions: int = 5

    @property
    def model_key(self) -> str:
        ch = "_chattn" if self.use_channel_attention else ""
        return (
            f"{self.latent_mode.value}_enc{self.encoder_type.value}{ch}"
            f"_eq{self.equiv_weight}_lat{self.latent_dim}"
            f"_kl{self.kl_weight_max}"
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
    recon_mse: float
    transformation_results: dict[str, TransformationResult]


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
) -> list[ComponentAblationResult]:
    """Train each variant and evaluate across all transformations.

    Models are saved under weights_dir with a unique suffix per config (model_key)
    and loaded on subsequent runs when use_cache is True. Transformation
    evaluation results are cached under assets/cache/component_ablation/<dataset>/results.
    If plots_output_dir is set, for each variant generates synthetic data
    and saves original-vs-synthetic plot(s) under plots_output_dir (no subfolders).
    If plot_per_sigma is False: one plot per variant using plot_synthetic_sigma.
    If plot_per_sigma is True: one plot per (variant, sigma) for each sigma in
    that variant's config.sigma_values, using plot_synthetic_transformation.
    """
    vae_creator = CreateTransformedVersionsCVAE(
        dataset_name=dataset_name,
        freq=freq,
        scaler_type=scaler_type,
        weights_dir=weights_dir,
        device=torch.device("cpu"),
    )

    trained_models: dict[str, tuple] = {}
    results: list[ComponentAblationResult] = []
    X_orig: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    results_cache_dir = _results_cache_dir(dataset_name) if use_cache else None

    for config in configs:
        if config.model_key not in trained_models:
            print(f"\n{'='*60}")
            print(f"Training: {config.name}")
            print(f"  latent_mode={config.latent_mode.value}, "
                  f"equiv={config.equiv_weight}, "
                  f"encoder={config.encoder_type.value}, "
                  f"channel_attn={config.use_channel_attention}")
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
            trained_models[config.model_key] = (model, z_mean, recon_mse)

        model, z_mean, recon_mse = trained_models[config.model_key]

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
                model, z_mean, _ = trained_models[config.model_key]
                for sigma in sigma_values:
                    X_synthetic = generate_synthetic_data(
                        model,
                        z_mean,
                        vae_creator,
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

    return results


def plot_component_ablation(
    results: list[ComponentAblationResult],
    sigma_values: list[float],
    output_dir: Path,
) -> None:
    """Generate ablation plots: monotonic response grid, signature bars,
    reconstruction-controllability scatter, and summary heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_monotonic_response_grid(results, sigma_values, output_dir)
    _plot_signature_comparison(results, output_dir)
    _plot_recon_vs_controllability(results, output_dir)
    _plot_summary_heatmap(results, sigma_values, output_dir)


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


def _plot_summary_heatmap(
    results: list[ComponentAblationResult],
    sigma_values: list[float],
    output_dir: Path,
) -> None:
    variant_names = [r.name for r in results]
    metric_names = [
        "Mean rho", "Direct rho", "Monotonic %", "Direct Mono %",
        "Recon MSE", "Mean FP dist",
    ]
    n_variants = len(results)
    n_metrics = len(metric_names)

    direct_rho_mean = np.mean([
        tr.direct_spearman_rho for tr in results[0].transformation_results.values()
    ])
    direct_mono_mean = np.mean([
        tr.direct_is_monotonic for tr in results[0].transformation_results.values()
    ]) * 100.0
    direct_mse_mean = np.mean([
        tr.direct_mse for tr in results[0].transformation_results.values()
    ])

    n_rows = n_variants + 1
    data = np.zeros((n_rows, n_metrics))

    for i, r in enumerate(results):
        rhos = [tr.spearman_rho for tr in r.transformation_results.values()]
        monos = [tr.is_monotonic for tr in r.transformation_results.values()]
        fp_dists = [tr.fingerprint_distance for tr in r.transformation_results.values()]
        data[i, 0] = np.mean(rhos)
        data[i, 1] = direct_rho_mean
        data[i, 2] = np.mean(monos) * 100.0
        data[i, 3] = direct_mono_mean
        data[i, 4] = r.recon_mse
        data[i, 5] = np.mean(fp_dists)

    data[n_variants, 0] = direct_rho_mean
    data[n_variants, 1] = direct_rho_mean
    data[n_variants, 2] = direct_mono_mean
    data[n_variants, 3] = direct_mono_mean
    data[n_variants, 4] = direct_mse_mean
    data[n_variants, 5] = 0.0

    variant_names = list(variant_names) + ["Direct (baseline)"]
    plot_data = data

    fig, ax = plt.subplots(figsize=(10, max(3, n_rows * 0.8 + 1)))
    im = ax.imshow(plot_data, aspect="auto", cmap="RdYlGn_r")

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(variant_names, fontsize=10)

    for i in range(n_rows):
        for j in range(n_metrics):
            fmt = ".1f" if j in (2, 3) else ".3f"
            ax.text(j, i, f"{data[i, j]:{fmt}}", ha="center", va="center",
                    fontsize=9, color="black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    sigma_str = ", ".join(str(s) for s in sigma_values)
    ax.set_title(
        f"Component Ablation Summary (σ ∈ {{{sigma_str}}})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    out = output_dir / "summary_heatmap.png"
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
        f"{'Variant':<28s}  {'Latent':<9s}  {'Equiv':<6s}  "
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

        print(
            f"{r.name:<28s}  {r.latent_mode.value:<9s}  "
            f"{r.equiv_weight:<6.1f}  {r.recon_mse:9.4f}  "
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
        help="Base directory for results; each dataset gets a subfolder named by dataset (e.g. output_dir/tourism/).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable loading/saving of model weights and transformation results; train and evaluate from scratch.",
    )
    args = parser.parse_args()

    base_output_dir = args.output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)
    CACHE_ABLATION_ROOT.mkdir(parents=True, exist_ok=True)

    def run_one(dataset_name: str, freq: str) -> None:
        dataset_output_dir = base_output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        weights_dir = CACHE_ABLATION_ROOT / dataset_name / "model_weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}")
        print(f"Component ablation: dataset={dataset_name}, freq={freq}")
        print(f"  Output: {dataset_output_dir}")
        print(f"  Weights: {weights_dir}")
        print("="*70)
        ablation_results = run_component_ablation(
            STANDARD_CONFIGS,
            dataset_name=dataset_name,
            freq=freq,
            plots_output_dir=dataset_output_dir,
            weights_dir=weights_dir,
            plot_per_sigma=True,
            use_cache=not args.no_cache,
        )
        plot_component_ablation(
            ablation_results,
            STANDARD_CONFIGS[0].sigma_values,
            dataset_output_dir,
        )
        print_component_ablation_report(ablation_results)

    if args.all_datasets:
        for dataset_name, freq in DEFAULT_ABLATION_DATASETS:
            run_one(dataset_name, freq)
    else:
        dataset_name = args.dataset if args.dataset is not None else "tourism"
        freq = args.freq
        if freq is None:
            freq = next(
                (f for d, f in DEFAULT_ABLATION_DATASETS if d == dataset_name),
                "Q",
            )
        run_one(dataset_name, freq)
