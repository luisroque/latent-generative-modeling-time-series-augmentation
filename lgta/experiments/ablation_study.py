"""
Ablation study for L-GTA components. Sweeps KL weight, latent
dimensionality, transformation type, encoder architecture, and
detemporalization method to measure their impact on controllability
(monotonic Wasserstein response to increasing sigma).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wasserstein_distance

from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from lgta.model.generate_data import generate_synthetic_data
from lgta.model.models import EncoderType
from lgta.transformations.manipulate_data import ManipulateData


@dataclass
class AblationConfig:
    """Single ablation configuration to evaluate."""

    name: str
    kl_weight_max: float = 0.0001
    latent_dim: int = 48
    transformation: str = "jitter"
    epochs: int = 1500
    kl_anneal_epochs: int = 100
    sigma_values: list[float] = field(
        default_factory=lambda: [0.05, 0.25, 0.5, 1.0, 2.0]
    )
    n_repetitions: int = 5
    encoder_type: EncoderType = EncoderType.FULL
    detemporalize_method: Literal["mean", "center"] = "mean"
    free_bits: float = 0.0
    spectral_norm: bool = False
    cyclical_kl: bool = False

    @property
    def model_key(self) -> str:
        return (
            f"kl{self.kl_weight_max}_lat{self.latent_dim}"
            f"_enc{self.encoder_type.value}"
            f"_fb{self.free_bits}_sn{self.spectral_norm}"
            f"_cyc{self.cyclical_kl}"
        )


@dataclass
class AblationResult:
    """Metrics from one ablation run."""

    name: str
    recon_mse: float
    z_mean_abs: float
    z_std_per_dim: float
    spearman_rho: float
    spearman_p: float
    is_monotonic: bool
    mean_distances: np.ndarray
    std_distances: np.ndarray
    direct_spearman_rho: float
    direct_is_monotonic: bool


def _wasserstein(X_ref: np.ndarray, X_aug: np.ndarray) -> float:
    distances = [
        wasserstein_distance(X_ref[:, i], X_aug[:, i])
        for i in range(X_ref.shape[1])
    ]
    return float(np.mean(distances))


def _evaluate_config(
    config: AblationConfig,
    model,
    z_mean: np.ndarray,
    X_orig: np.ndarray,
    X_recon: np.ndarray,
    vae_creator: CreateTransformedVersionsCVAE,
) -> AblationResult:
    """Evaluate a single ablation configuration (sigma sweep)."""
    n_sigma = len(config.sigma_values)
    lgta_dists = np.zeros((n_sigma, config.n_repetitions))
    direct_dists = np.zeros((n_sigma, config.n_repetitions))

    for i, sigma in enumerate(config.sigma_values):
        for rep in range(config.n_repetitions):
            X_lgta = generate_synthetic_data(
                model, z_mean, vae_creator,
                config.transformation, [sigma],
                detemporalize_method=config.detemporalize_method,
            )
            X_direct = ManipulateData(
                x=X_orig, transformation=config.transformation,
                parameters=[sigma],
            ).apply_transf()

            lgta_dists[i, rep] = _wasserstein(X_orig, X_lgta)
            direct_dists[i, rep] = _wasserstein(X_orig, X_direct)

    lgta_means = lgta_dists.mean(axis=1)
    direct_means = direct_dists.mean(axis=1)
    lgta_rho, lgta_p = spearmanr(config.sigma_values, lgta_means)
    direct_rho, _ = spearmanr(config.sigma_values, direct_means)

    recon_mse = float(np.mean((X_orig - X_recon) ** 2))
    z_mean_abs = float(np.mean(np.abs(z_mean)))
    z_std_per_dim = float(np.mean(np.std(z_mean, axis=0)))
    lgta_mono = all(
        lgta_means[j] <= lgta_means[j + 1]
        for j in range(len(lgta_means) - 1)
    )
    direct_mono = all(
        direct_means[j] <= direct_means[j + 1]
        for j in range(len(direct_means) - 1)
    )

    return AblationResult(
        name=config.name,
        recon_mse=recon_mse,
        z_mean_abs=z_mean_abs,
        z_std_per_dim=z_std_per_dim,
        spearman_rho=float(lgta_rho),
        spearman_p=float(lgta_p),
        is_monotonic=lgta_mono,
        mean_distances=lgta_means,
        std_distances=lgta_dists.std(axis=1),
        direct_spearman_rho=float(direct_rho),
        direct_is_monotonic=direct_mono,
    )


def _train_model(
    config: AblationConfig,
    vae_creator: CreateTransformedVersionsCVAE,
):
    """Train a CVAE with the given ablation config and return model + latents."""
    print(f"\n--- Training: {config.name} (kl={config.kl_weight_max}, "
          f"d={config.latent_dim}, enc={config.encoder_type.value}, "
          f"fb={config.free_bits}, sn={config.spectral_norm}, "
          f"cyc={config.cyclical_kl}) ---")

    model, _, _ = vae_creator.fit(
        epochs=config.epochs,
        latent_dim=config.latent_dim,
        kl_anneal_epochs=config.kl_anneal_epochs,
        kl_weight_max=config.kl_weight_max,
        load_weights=False,
        encoder_type=config.encoder_type,
        free_bits=config.free_bits,
        spectral_norm=config.spectral_norm,
        cyclical_kl=config.cyclical_kl,
    )
    X_recon, _, z_mean, _ = vae_creator.predict(
        model, detemporalize_method="mean",
    )
    return model, X_recon, z_mean


def run_ablation(
    configs: list[AblationConfig],
    dataset_name: str = "tourism_small",
    freq: str = "Q",
) -> list[AblationResult]:
    """Run the full ablation suite. Reuses model training when configs share
    the same model_key."""
    vae_creator = CreateTransformedVersionsCVAE(
        dataset_name=dataset_name, freq=freq,
    )
    X_orig = vae_creator.df.astype(np.float32).to_numpy()

    trained_models: dict[str, tuple] = {}
    results: list[AblationResult] = []

    for config in configs:
        if config.model_key not in trained_models:
            model, X_recon, z_mean = _train_model(config, vae_creator)
            trained_models[config.model_key] = (model, X_recon, z_mean)

        model, X_recon, z_mean = trained_models[config.model_key]
        print(f"\n--- Evaluating: {config.name} ---")
        result = _evaluate_config(
            config, model, z_mean, X_orig, X_recon, vae_creator,
        )
        results.append(result)
        print(f"  Recon MSE={result.recon_mse:.6f}  "
              f"rho={result.spearman_rho:.3f}  "
              f"Mono={'YES' if result.is_monotonic else 'NO'}")

    return results


def plot_ablation(
    results: list[AblationResult],
    sigma_values: list[float],
    output_dir: Path,
) -> None:
    """Plot sigma vs Wasserstein curves and reconstruction-controllability trade-off."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for r in results:
        ax.plot(sigma_values, r.mean_distances, "o-", linewidth=2,
                label=f"{r.name} (rho={r.spearman_rho:.2f})")
        ax.fill_between(
            sigma_values,
            r.mean_distances - r.std_distances,
            r.mean_distances + r.std_distances,
            alpha=0.1,
        )
    ax.set_xlabel("sigma", fontsize=13)
    ax.set_ylabel("Wasserstein Distance", fontsize=13)
    ax.set_title("Monotonic Response by Configuration",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for r in results:
        marker = "o" if r.is_monotonic else "x"
        ax.scatter(r.recon_mse, r.spearman_rho, s=120,
                   marker=marker, zorder=5)
        ax.annotate(r.name, (r.recon_mse, r.spearman_rho),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_xlabel("Reconstruction MSE", fontsize=13)
    ax.set_ylabel("Spearman rho (controllability)", fontsize=13)
    ax.set_title("Reconstruction vs Controllability",
                 fontsize=14, fontweight="bold")
    ax.axhline(y=1.0, color="green", linestyle=":", alpha=0.4,
               label="Perfect rho=1")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "ablation_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Ablation plot saved to: {output_file}")
    plt.close()


def print_ablation_report(results: list[AblationResult]) -> None:
    header = (
        f"{'Config':<28s} {'ReconMSE':>10s} {'|z|':>8s} "
        f"{'z_std':>8s} {'rho':>8s} {'Mono':>6s} {'Dir_rho':>10s}"
    )
    print("\n" + "=" * 90)
    print("ABLATION STUDY REPORT")
    print("=" * 90)
    print(header)
    print("-" * 90)
    for r in results:
        mono_str = "YES" if r.is_monotonic else "NO"
        print(
            f"{r.name:<28s} {r.recon_mse:10.6f} {r.z_mean_abs:8.3f} "
            f"{r.z_std_per_dim:8.3f} {r.spearman_rho:8.3f} {mono_str:>6s} "
            f"{r.direct_spearman_rho:10.3f}"
        )
    print("=" * 90)

    best = max(results, key=lambda r: r.spearman_rho)
    print(f"\nBest controllability: {best.name} (rho={best.spearman_rho:.3f}, "
          f"Mono={'YES' if best.is_monotonic else 'NO'}, "
          f"ReconMSE={best.recon_mse:.6f})")


STANDARD_CONFIGS: list[AblationConfig] = [
    # --- Axis 1: KL weight sweep (d=48, full encoder) ---
    AblationConfig(
        name="ae_baseline(kl=0)",
        kl_weight_max=0.0, latent_dim=48,
    ),
    AblationConfig(
        name="kl=1e-4,d=48",
        kl_weight_max=0.0001, latent_dim=48,
    ),
    AblationConfig(
        name="kl=1e-3,d=48",
        kl_weight_max=0.001, latent_dim=48,
    ),
    AblationConfig(
        name="kl=1e-2,d=48",
        kl_weight_max=0.01, latent_dim=48,
    ),
    AblationConfig(
        name="kl=1e-1,d=48",
        kl_weight_max=0.1, latent_dim=48,
    ),
    AblationConfig(
        name="kl=1.0,d=48",
        kl_weight_max=1.0, latent_dim=48,
    ),
    # --- Axis 2: Latent dimensionality sweep (kl=1e-2) ---
    AblationConfig(
        name="kl=1e-2,d=8",
        kl_weight_max=0.01, latent_dim=8,
    ),
    AblationConfig(
        name="kl=1e-2,d=16",
        kl_weight_max=0.01, latent_dim=16,
    ),
    AblationConfig(
        name="kl=1e-2,d=32",
        kl_weight_max=0.01, latent_dim=32,
    ),
    # --- Axis 3: Transformation type (kl=1e-2, d=48) ---
    AblationConfig(
        name="kl=1e-2+scaling",
        kl_weight_max=0.01, latent_dim=48,
        transformation="scaling",
    ),
    AblationConfig(
        name="kl=1e-2+magnitude_warp",
        kl_weight_max=0.01, latent_dim=48,
        transformation="magnitude_warp",
    ),
    # --- Axis 4: Encoder architecture ---
    AblationConfig(
        name="kl=1e-2+simple_enc",
        kl_weight_max=0.01, latent_dim=48,
        encoder_type=EncoderType.SIMPLE,
    ),
    # --- Axis 5: Detemporalize method ---
    AblationConfig(
        name="kl=1e-2+center_detemp",
        kl_weight_max=0.01, latent_dim=48,
        detemporalize_method="center",
    ),
    # --- Improvements ---
    AblationConfig(
        name="kl=1e-2+free_bits",
        kl_weight_max=0.01, latent_dim=48,
        free_bits=0.25,
    ),
    AblationConfig(
        name="kl=1e-2+spectral_norm",
        kl_weight_max=0.01, latent_dim=48,
        spectral_norm=True,
    ),
    AblationConfig(
        name="kl=1e-2+cyclical_kl",
        kl_weight_max=0.01, latent_dim=48,
        cyclical_kl=True,
    ),
]


if __name__ == "__main__":
    output_dir = Path("assets/results/ablation_study")
    output_dir.mkdir(parents=True, exist_ok=True)

    sigma_values = STANDARD_CONFIGS[0].sigma_values
    results = run_ablation(STANDARD_CONFIGS)
    plot_ablation(results, sigma_values, output_dir)
    print_ablation_report(results)
