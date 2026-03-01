"""
Monotonic response experiment: sweeps sigma to verify L-GTA produces
a smooth, monotonically increasing Wasserstein distance curve while
direct augmentation is noisier and may plateau or become erratic.
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wasserstein_distance

from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from lgta.model.generate_data import generate_synthetic_data
from lgta.transformations.manipulate_data import ManipulateData


@dataclass
class SweepConfig:
    """Configuration for the monotonic response sweep experiment."""

    dataset_name: str = "tourism_small"
    freq: str = "Q"
    transformation: str = "jitter"
    sigma_values: list[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0])
    n_repetitions: int = 5
    load_weights: bool = True
    series_to_plot: int = 0
    epochs: int = 1500
    latent_dim: int = 16
    kl_anneal_epochs: int = 100
    kl_weight_max: float = 0.1
    equiv_weight: float = 0.0
    scaler_type: str = "standard"
    output_dir: Path = field(
        default_factory=lambda: Path("assets/results/monotonic_response")
    )

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class SweepResults:
    """Results from the monotonic response sweep experiment."""

    sigma_values: list[float]
    lgta_distances: np.ndarray
    direct_distances: np.ndarray
    lgta_spearman_rho: float
    lgta_spearman_p: float
    direct_spearman_rho: float
    direct_spearman_p: float
    X_orig: np.ndarray
    lgta_samples: list[np.ndarray]
    direct_samples: list[np.ndarray]


@dataclass
class PretrainedComponents:
    """Holds the trained CVAE and associated data for reuse across sweeps."""

    vae_creator: CreateTransformedVersionsCVAE
    model: object
    z_mean: np.ndarray
    X_orig: np.ndarray


def _compute_wasserstein(X_ref: np.ndarray, X_aug: np.ndarray) -> float:
    """Compute mean per-series Wasserstein distance."""
    distances = [
        wasserstein_distance(X_ref[:, i], X_aug[:, i]) for i in range(X_ref.shape[1])
    ]
    return float(np.mean(distances))


def train_model(config: SweepConfig) -> PretrainedComponents:
    """Train the CVAE once and return reusable components."""
    vae_creator = CreateTransformedVersionsCVAE(
        dataset_name=config.dataset_name,
        freq=config.freq,
        scaler_type=config.scaler_type,
    )
    model, _, _ = vae_creator.fit(
        epochs=config.epochs,
        latent_dim=config.latent_dim,
        kl_anneal_epochs=config.kl_anneal_epochs,
        kl_weight_max=config.kl_weight_max,
        load_weights=config.load_weights,
        equiv_weight=config.equiv_weight,
    )
    X_recon, _, z_mean, _ = vae_creator.predict(
        model,
        detemporalize_method="mean",
    )
    X_orig = vae_creator.X_train_raw
    recon_mse = float(np.mean((X_orig - X_recon) ** 2))
    print(f"  Data shape: {X_orig.shape}")
    print(f"  Reconstruction MSE: {recon_mse:.6f}")
    return PretrainedComponents(vae_creator, model, z_mean, X_orig)


def run_sweep(
    config: SweepConfig,
    pretrained: PretrainedComponents,
) -> SweepResults:
    """Sweep sigma measuring Wasserstein distance for both L-GTA and direct."""
    print("=" * 60)
    print(f"SWEEP: {config.transformation}")
    print("=" * 60)
    print(f"Sigma values: {config.sigma_values}")

    vae_creator = pretrained.vae_creator
    model = pretrained.model
    z_mean = pretrained.z_mean
    X_orig = pretrained.X_orig

    n_sigma = len(config.sigma_values)
    lgta_distances = np.zeros((n_sigma, config.n_repetitions))
    direct_distances = np.zeros((n_sigma, config.n_repetitions))
    lgta_samples: list[np.ndarray] = []
    direct_samples: list[np.ndarray] = []

    for i, sigma in enumerate(config.sigma_values):
        print(f"\n  sigma={sigma:.3f} ({i + 1}/{n_sigma})")
        for rep in range(config.n_repetitions):
            X_lgta = generate_synthetic_data(
                model,
                z_mean,
                vae_creator,
                config.transformation,
                [sigma],
            )
            X_orig_scaled = vae_creator.scaler_target.transform(X_orig)
            X_direct_scaled = ManipulateData(
                x=X_orig_scaled,
                transformation=config.transformation,
                parameters=[sigma],
            ).apply_transf()
            X_direct = vae_creator.scaler_target.inverse_transform(
                X_direct_scaled,
            )

            lgta_distances[i, rep] = _compute_wasserstein(X_orig, X_lgta)
            direct_distances[i, rep] = _compute_wasserstein(X_orig, X_direct)
            print(
                f"    rep {rep + 1}/{config.n_repetitions}: "
                f"L-GTA={lgta_distances[i, rep]:.4f}, "
                f"Direct={direct_distances[i, rep]:.4f}"
            )

        lgta_samples.append(X_lgta)
        direct_samples.append(X_direct)

    lgta_means = lgta_distances.mean(axis=1)
    direct_means = direct_distances.mean(axis=1)
    lgta_rho, lgta_p = spearmanr(config.sigma_values, lgta_means)
    direct_rho, direct_p = spearmanr(config.sigma_values, direct_means)

    return SweepResults(
        sigma_values=config.sigma_values,
        lgta_distances=lgta_distances,
        direct_distances=direct_distances,
        lgta_spearman_rho=float(lgta_rho),
        lgta_spearman_p=float(lgta_p),
        direct_spearman_rho=float(direct_rho),
        direct_spearman_p=float(direct_p),
        X_orig=X_orig,
        lgta_samples=lgta_samples,
        direct_samples=direct_samples,
    )


def plot_monotonic_response(
    results: SweepResults,
    output_dir: Path,
    transformation: str,
) -> Path:
    """Plot the monotonic response curves with error bands."""
    sigma = np.array(results.sigma_values)
    lgta_mean = results.lgta_distances.mean(axis=1)
    lgta_std = results.lgta_distances.std(axis=1)
    direct_mean = results.direct_distances.mean(axis=1)
    direct_std = results.direct_distances.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        sigma,
        lgta_mean,
        "o-",
        color="#2196F3",
        linewidth=2.5,
        markersize=8,
        label=f"L-GTA (\u03c1={results.lgta_spearman_rho:.3f})",
    )
    ax.fill_between(
        sigma, lgta_mean - lgta_std, lgta_mean + lgta_std, color="#2196F3", alpha=0.15
    )
    ax.plot(
        sigma,
        direct_mean,
        "s--",
        color="#F44336",
        linewidth=2.5,
        markersize=8,
        label=f"Direct (\u03c1={results.direct_spearman_rho:.3f})",
    )
    ax.fill_between(
        sigma,
        direct_mean - direct_std,
        direct_mean + direct_std,
        color="#F44336",
        alpha=0.15,
    )

    ax.set_xlabel("\u03c3", fontsize=14)
    ax.set_ylabel("Wasserstein Distance", fontsize=14)
    title = f"Monotonic Response ({transformation}): L-GTA vs Direct"
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = output_dir / "monotonic_response_curve.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")
    plt.close()
    return output_file


def plot_series_comparison(
    results: SweepResults,
    output_dir: Path,
    transformation: str,
    series_idx: int = 0,
) -> Path:
    """Plot sigma levels x 2 columns (L-GTA | Direct)."""
    n_sigma = len(results.sigma_values)
    fig, axes = plt.subplots(
        n_sigma, 2, figsize=(14, 3 * n_sigma), sharex=True, sharey=True
    )
    if n_sigma == 1:
        axes = axes.reshape(1, -1)

    orig_series = results.X_orig[:, series_idx]

    for row, sigma in enumerate(results.sigma_values):
        lgta_series = results.lgta_samples[row][:, series_idx]
        direct_series = results.direct_samples[row][:, series_idx]

        ax_l = axes[row, 0]
        ax_l.plot(orig_series, color="black", linewidth=1.2, label="Original")
        ax_l.plot(
            lgta_series,
            color="#2196F3",
            linewidth=1.2,
            alpha=0.8,
            linestyle="--",
            label="L-GTA",
        )
        ax_l.set_ylabel(f"\u03c3={sigma}")
        if row == 0:
            ax_l.set_title("L-GTA (latent augmentation)")
            ax_l.legend(fontsize=8, loc="upper right")
        ax_l.grid(True, alpha=0.2)

        ax_r = axes[row, 1]
        ax_r.plot(orig_series, color="black", linewidth=1.2, label="Original")
        ax_r.plot(
            direct_series,
            color="#F44336",
            linewidth=1.2,
            alpha=0.8,
            linestyle="--",
            label="Direct",
        )
        if row == 0:
            ax_r.set_title("Direct augmentation")
            ax_r.legend(fontsize=8, loc="upper right")
        ax_r.grid(True, alpha=0.2)

    axes[-1, 0].set_xlabel("Time step")
    axes[-1, 1].set_xlabel("Time step")
    fig.suptitle(
        f"Original vs Synthetic ({transformation}) \u2014 series {series_idx}",
        fontsize=16,
        fontweight="bold",
        y=1.002,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    output_file = output_dir / "series_comparison_grid.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")
    plt.close()
    return output_file


def print_controllability_report(results: SweepResults) -> None:
    """Print controllability scores and summary table."""
    print("\n" + "=" * 60)
    print("CONTROLLABILITY ANALYSIS REPORT")
    print("=" * 60)
    print(f"\n  Spearman rho  (sigma vs Wasserstein):")
    print(
        f"    L-GTA:  {results.lgta_spearman_rho:.4f}  "
        f"(p={results.lgta_spearman_p:.4f})"
    )
    print(
        f"    Direct: {results.direct_spearman_rho:.4f}  "
        f"(p={results.direct_spearman_p:.4f})"
    )

    print(f"\n  {'sigma':>8s}  {'L-GTA':>14s}  {'Direct':>14s}")
    print(f"  {'---':>8s}  {'---':>14s}  {'---':>14s}")
    for i, sigma in enumerate(results.sigma_values):
        lm = results.lgta_distances[i].mean()
        ls = results.lgta_distances[i].std()
        dm = results.direct_distances[i].mean()
        ds = results.direct_distances[i].std()
        print(f"  {sigma:8.3f}  {lm:7.4f}+/-{ls:.4f}  {dm:7.4f}+/-{ds:.4f}")

    lgta_m = results.lgta_distances.mean(axis=1)
    direct_m = results.direct_distances.mean(axis=1)
    lgta_mono = all(lgta_m[j] <= lgta_m[j + 1] for j in range(len(lgta_m) - 1))
    direct_mono = all(direct_m[j] <= direct_m[j + 1] for j in range(len(direct_m) - 1))
    print(
        f"\n  Monotonicity:  L-GTA={'YES' if lgta_mono else 'NO'}, "
        f"Direct={'YES' if direct_mono else 'NO'}"
    )
    print("=" * 60)


ALL_TRANSFORMATIONS: list[str] = [
    "jitter",
    "scaling",
    "magnitude_warp",
    "drift",
    "trend",
]


def _run_sweep_for_transformation(
    config: SweepConfig,
    pretrained: PretrainedComponents,
    transformation: str,
    base_output_dir: Path,
) -> SweepResults:
    config.transformation = transformation
    config.output_dir = base_output_dir / transformation
    config.output_dir.mkdir(parents=True, exist_ok=True)

    results = run_sweep(config, pretrained)
    plot_monotonic_response(results, config.output_dir, transformation)
    plot_series_comparison(
        results,
        config.output_dir,
        transformation,
        series_idx=config.series_to_plot,
    )
    print_controllability_report(results)
    return results


def _run_signature_analysis(
    all_results: dict[str, SweepResults],
    base_output_dir: Path,
    series_idx: int,
) -> None:
    """Compute fingerprints for each transformation and produce comparison
    plots and reports that prove LGTA preserves the transformation character."""
    from lgta.experiments.transformation_signatures import (
        compute_fingerprint,
        plot_fingerprint_comparison,
        plot_residual_comparison,
        print_signature_report,
    )

    transformations = list(all_results.keys())
    X_orig = all_results[transformations[0]].X_orig

    lgta_fps = {}
    direct_fps = {}
    lgta_samples = {}
    direct_samples = {}

    for t in transformations:
        r = all_results[t]
        lgta_fps[t] = compute_fingerprint(X_orig, r.lgta_samples[-1])
        direct_fps[t] = compute_fingerprint(X_orig, r.direct_samples[-1])
        lgta_samples[t] = r.lgta_samples[-1]
        direct_samples[t] = r.direct_samples[-1]

    print_signature_report(transformations, lgta_fps, direct_fps)
    plot_fingerprint_comparison(transformations, lgta_fps, direct_fps, base_output_dir)
    plot_residual_comparison(
        X_orig, lgta_samples, direct_samples, transformations,
        series_idx, base_output_dir,
    )


if __name__ == "__main__":
    base_dir = Path("assets/results/monotonic_response_tourism")
    config = SweepConfig(
        dataset_name="tourism_small",
        freq="Q",
        latent_dim=16,
        epochs=1500,
        kl_anneal_epochs=100,
        kl_weight_max=0.1,
        equiv_weight=1.0,
        scaler_type="standard",
        load_weights=False,
    )
    pretrained = train_model(config)

    all_results: dict[str, SweepResults] = {}
    for transf in ALL_TRANSFORMATIONS:
        all_results[transf] = _run_sweep_for_transformation(
            config, pretrained, transf, base_dir,
        )

    _run_signature_analysis(all_results, base_dir, config.series_to_plot)
