"""
Monotonic response experiment: sweeps sigma to verify L-GTA produces
a smooth, monotonically increasing Wasserstein distance curve while
direct augmentation is noisier and may plateau or become erratic.

Both methods are compared against the original data X_orig. Wasserstein
distances are computed in MinMax-scaled [0,1] space so that L-GTA (which
decodes through the CVAE) and direct augmentation are on the same scale.
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
    sigma_values: list[float] = field(
        default_factory=lambda: [0.05, 0.25, 0.5, 1.0, 2.0]
    )
    n_repetitions: int = 5
    load_weights: bool = True
    series_to_plot: int = 0
    epochs: int = 1000
    latent_dim: int = 16
    kl_anneal_epochs: int = 100
    use_dynamic_features: bool = True
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


def _compute_wasserstein(X_ref: np.ndarray, X_aug: np.ndarray) -> float:
    """Compute mean per-series Wasserstein distance.

    Both inputs have shape (n_timesteps, n_series). The 1-D Wasserstein
    distance is computed for each series and averaged.
    """
    distances = [
        wasserstein_distance(X_ref[:, i], X_aug[:, i])
        for i in range(X_ref.shape[1])
    ]
    return float(np.mean(distances))


def run_sweep(config: SweepConfig) -> SweepResults:
    """Train the CVAE once, then sweep sigma measuring Wasserstein distance
    for both L-GTA (latent augmentation) and direct (raw-data) augmentation.
    """
    print("=" * 60)
    print("MONOTONIC RESPONSE SWEEP EXPERIMENT")
    print("=" * 60)
    print(f"\nDataset: {config.dataset_name}, Freq: {config.freq}")
    print(f"Transformation: {config.transformation}")
    print(f"Sigma values: {config.sigma_values}")
    print(f"Repetitions per sigma: {config.n_repetitions}")

    print(f"\n[1/3] Training CVAE model (latent_dim={config.latent_dim}, "
          f"kl_anneal_epochs={config.kl_anneal_epochs})...")
    vae_creator = CreateTransformedVersionsCVAE(
        dataset_name=config.dataset_name,
        freq=config.freq,
    )
    model, _, _ = vae_creator.fit(
        epochs=config.epochs,
        latent_dim=config.latent_dim,
        kl_anneal_epochs=config.kl_anneal_epochs,
        load_weights=config.load_weights,
        use_dynamic_features=config.use_dynamic_features,
    )
    _, z, z_mean, z_log_var = vae_creator.predict(model)
    X_orig = vae_creator.X_train_raw
    print(f"  Data shape: {X_orig.shape}")
    print(f"  X_orig range: [{X_orig.min():.1f}, {X_orig.max():.1f}]")

    print("\n[2/3] Sweeping sigma values...")
    n_sigma = len(config.sigma_values)
    lgta_distances = np.zeros((n_sigma, config.n_repetitions))
    direct_distances = np.zeros((n_sigma, config.n_repetitions))
    lgta_samples: list[np.ndarray] = []
    direct_samples: list[np.ndarray] = []

    for i, sigma in enumerate(config.sigma_values):
        print(f"\n  sigma={sigma:.3f} ({i + 1}/{n_sigma})")
        for rep in range(config.n_repetitions):
            X_lgta = generate_synthetic_data(
                model, z_mean, vae_creator, config.transformation, [sigma]
            )
            X_direct = ManipulateData(
                x=X_orig,
                transformation=config.transformation,
                parameters=[sigma],
            ).apply_transf()

            lgta_distances[i, rep] = _compute_wasserstein(X_orig, X_lgta)
            direct_distances[i, rep] = _compute_wasserstein(X_orig, X_direct)
            print(
                f"    rep {rep + 1}/{config.n_repetitions}: "
                f"L-GTA={lgta_distances[i, rep]:.4f}, "
                f"Direct={direct_distances[i, rep]:.4f}"
            )

        lgta_samples.append(X_lgta)
        direct_samples.append(X_direct)
        print(
            f"  X_lgta  range: [{X_lgta.min():.1f}, {X_lgta.max():.1f}]"
        )
        print(
            f"  X_direct range: [{X_direct.min():.1f}, {X_direct.max():.1f}]"
        )

    print("\n[3/3] Computing controllability scores...")
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


def plot_monotonic_response(results: SweepResults, output_dir: Path) -> Path:
    """Plot the monotonic response curves with error bands."""
    sigma = np.array(results.sigma_values)
    lgta_mean = results.lgta_distances.mean(axis=1)
    lgta_std = results.lgta_distances.std(axis=1)
    direct_mean = results.direct_distances.mean(axis=1)
    direct_std = results.direct_distances.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sigma, lgta_mean, "o-", color="#2196F3", linewidth=2.5,
            markersize=8,
            label=f"L-GTA (\u03c1={results.lgta_spearman_rho:.3f})")
    ax.fill_between(sigma, lgta_mean - lgta_std, lgta_mean + lgta_std,
                     color="#2196F3", alpha=0.15)
    ax.plot(sigma, direct_mean, "s--", color="#F44336", linewidth=2.5,
            markersize=8,
            label=f"Direct (\u03c1={results.direct_spearman_rho:.3f})")
    ax.fill_between(sigma, direct_mean - direct_std, direct_mean + direct_std,
                     color="#F44336", alpha=0.15)

    ax.set_xlabel("Transformation Parameter (\u03c3)", fontsize=14)
    ax.set_ylabel("Wasserstein Distance (scaled)", fontsize=14)
    ax.set_title("Monotonic Response: L-GTA vs Direct Augmentation",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    output_file = output_dir / "monotonic_response_curve.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")
    plt.close()
    return output_file


def plot_series_comparison(
    results: SweepResults, output_dir: Path, series_idx: int = 0,
) -> Path:
    """Plot 5 rows (sigma levels) x 2 columns (L-GTA | Direct).

    Each subplot overlays the original series (black) and the synthetic
    series (coloured, dashed) for one sigma level, giving a visual
    diagnostic of how each method distorts the data.
    """
    n_sigma = len(results.sigma_values)
    fig, axes = plt.subplots(n_sigma, 2, figsize=(14, 3 * n_sigma),
                             sharex=True, sharey=True)
    if n_sigma == 1:
        axes = axes.reshape(1, -1)

    orig_series = results.X_orig[:, series_idx]

    for row, sigma in enumerate(results.sigma_values):
        lgta_series = results.lgta_samples[row][:, series_idx]
        direct_series = results.direct_samples[row][:, series_idx]

        ax_l = axes[row, 0]
        ax_l.plot(orig_series, color="black", linewidth=1.2, label="Original")
        ax_l.plot(lgta_series, color="#2196F3", linewidth=1.2,
                  alpha=0.8, linestyle="--", label="L-GTA")
        ax_l.set_ylabel(f"\u03c3={sigma}")
        if row == 0:
            ax_l.set_title("L-GTA (latent augmentation)")
            ax_l.legend(fontsize=8, loc="upper right")
        ax_l.grid(True, alpha=0.2)

        ax_r = axes[row, 1]
        ax_r.plot(orig_series, color="black", linewidth=1.2, label="Original")
        ax_r.plot(direct_series, color="#F44336", linewidth=1.2,
                  alpha=0.8, linestyle="--", label="Direct")
        if row == 0:
            ax_r.set_title("Direct augmentation")
            ax_r.legend(fontsize=8, loc="upper right")
        ax_r.grid(True, alpha=0.2)

    axes[-1, 0].set_xlabel("Time step")
    axes[-1, 1].set_xlabel("Time step")
    fig.suptitle(f"Original vs Synthetic — series {series_idx}",
                 fontsize=16, fontweight="bold", y=1.002)
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
    print(f"    L-GTA:  {results.lgta_spearman_rho:.4f}  "
          f"(p={results.lgta_spearman_p:.4f})")
    print(f"    Direct: {results.direct_spearman_rho:.4f}  "
          f"(p={results.direct_spearman_p:.4f})")

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
    lgta_mono = all(lgta_m[j] <= lgta_m[j+1] for j in range(len(lgta_m)-1))
    direct_mono = all(direct_m[j] <= direct_m[j+1] for j in range(len(direct_m)-1))
    print(f"\n  Monotonicity:  L-GTA={'YES' if lgta_mono else 'NO'}, "
          f"Direct={'YES' if direct_mono else 'NO'}")
    print("=" * 60)


def _run_config(config: SweepConfig) -> None:
    results = run_sweep(config)
    plot_monotonic_response(results, config.output_dir)
    plot_series_comparison(results, config.output_dir,
                           series_idx=config.series_to_plot)
    print_controllability_report(results)


if __name__ == "__main__":
    synthetic_config = SweepConfig(
        dataset_name="synthetic",
        freq="D",
        latent_dim=16,
        kl_anneal_epochs=100,
        load_weights=False,
        output_dir=Path("assets/results/monotonic_response_synthetic"),
    )
    _run_config(synthetic_config)

    tourism_config = SweepConfig(
        dataset_name="tourism_small",
        freq="Q",
        latent_dim=16,
        kl_anneal_epochs=100,
        load_weights=False,
        output_dir=Path("assets/results/monotonic_response_tourism"),
    )
    _run_config(tourism_config)
