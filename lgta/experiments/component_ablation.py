"""
Component ablation study for L-GTA. Isolates the contribution of the
two key innovations (temporal latent space, equivariant decoder training)
using a 2x2 matrix plus an encoder-type axis. Measures controllability,
reconstruction quality, and transformation signature preservation.
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wasserstein_distance

from lgta.experiments.transformation_signatures import (
    TransformationFingerprint,
    compute_fingerprint,
)
from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from lgta.model.generate_data import generate_synthetic_data
from lgta.model.models import EncoderType, LatentMode
from lgta.transformations.manipulate_data import ManipulateData


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
    latent_dim: int = 16
    kl_weight_max: float = 0.1
    kl_anneal_epochs: int = 100
    epochs: int = 1500
    sigma_values: list[float] = field(
        default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0]
    )
    n_repetitions: int = 5

    @property
    def model_key(self) -> str:
        return (
            f"{self.latent_mode.value}_enc{self.encoder_type.value}"
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


@dataclass
class ComponentAblationResult:
    """Full result for one ablation variant across all transformations."""

    name: str
    latent_mode: LatentMode
    equiv_weight: float
    encoder_type: EncoderType
    recon_mse: float
    transformation_results: dict[str, TransformationResult]


def _fingerprint_distance(
    lgta: TransformationFingerprint,
    direct: TransformationFingerprint,
) -> float:
    """Euclidean distance between LGTA and direct fingerprint vectors."""
    v_lgta = np.array([
        lgta.autocorrelation, lgta.linearity, lgta.amplitude_dependence,
    ])
    v_direct = np.array([
        direct.autocorrelation, direct.linearity, direct.amplitude_dependence,
    ])
    return float(np.linalg.norm(v_lgta - v_direct))


def _compute_wasserstein(X_ref: np.ndarray, X_aug: np.ndarray) -> float:
    distances = [
        wasserstein_distance(X_ref[:, i], X_aug[:, i])
        for i in range(X_ref.shape[1])
    ]
    return float(np.mean(distances))


def _evaluate_transformation(
    config: ComponentAblationConfig,
    model,
    z_mean: np.ndarray,
    X_orig: np.ndarray,
    vae_creator: CreateTransformedVersionsCVAE,
    transformation: str,
) -> TransformationResult:
    """Sweep sigma for one transformation, compute controllability + fingerprint."""
    n_sigma = len(config.sigma_values)
    lgta_dists = np.zeros((n_sigma, config.n_repetitions))

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

            lgta_dists[i, rep] = _compute_wasserstein(X_orig, X_lgta)

            if i == n_sigma - 1 and rep == config.n_repetitions - 1:
                last_X_lgta = X_lgta
                last_X_direct = X_direct

    lgta_means = lgta_dists.mean(axis=1)
    lgta_rho, _ = spearmanr(config.sigma_values, lgta_means)
    lgta_mono = all(
        lgta_means[j] <= lgta_means[j + 1] for j in range(len(lgta_means) - 1)
    )

    assert last_X_lgta is not None and last_X_direct is not None
    lgta_fp = compute_fingerprint(X_orig, last_X_lgta)
    direct_fp = compute_fingerprint(X_orig, last_X_direct)
    fp_dist = _fingerprint_distance(lgta_fp, direct_fp)

    return TransformationResult(
        transformation=transformation,
        spearman_rho=float(lgta_rho),
        is_monotonic=lgta_mono,
        mean_distances=lgta_means,
        std_distances=lgta_dists.std(axis=1),
        fingerprint=lgta_fp,
        direct_fingerprint=direct_fp,
        fingerprint_distance=fp_dist,
    )


def run_component_ablation(
    configs: list[ComponentAblationConfig],
    dataset_name: str = "tourism_small",
    freq: str = "Q",
    scaler_type: str = "standard",
) -> list[ComponentAblationResult]:
    """Train each variant and evaluate across all transformations."""
    vae_creator = CreateTransformedVersionsCVAE(
        dataset_name=dataset_name, freq=freq, scaler_type=scaler_type,
    )

    trained_models: dict[str, tuple] = {}
    results: list[ComponentAblationResult] = []
    X_orig: np.ndarray | None = None

    for config in configs:
        if config.model_key not in trained_models:
            print(f"\n{'='*60}")
            print(f"Training: {config.name}")
            print(f"  latent_mode={config.latent_mode.value}, "
                  f"equiv={config.equiv_weight}, "
                  f"encoder={config.encoder_type.value}")
            print(f"{'='*60}")

            model, _, _ = vae_creator.fit(
                epochs=config.epochs,
                latent_dim=config.latent_dim,
                kl_anneal_epochs=config.kl_anneal_epochs,
                kl_weight_max=config.kl_weight_max,
                load_weights=False,
                encoder_type=config.encoder_type,
                equiv_weight=config.equiv_weight,
                latent_mode=config.latent_mode,
            )
            if X_orig is None:
                X_orig = vae_creator.X_train_raw
            X_recon, _, z_mean, _ = vae_creator.predict(
                model, detemporalize_method="mean",
            )
            recon_mse = float(np.mean((X_orig - X_recon) ** 2))
            trained_models[config.model_key] = (model, z_mean, recon_mse)

        model, z_mean, recon_mse = trained_models[config.model_key]

        transf_results: dict[str, TransformationResult] = {}
        for transf in ALL_TRANSFORMATIONS:
            print(f"\n  Evaluating {config.name} / {transf}...")
            transf_results[transf] = _evaluate_transformation(
                config, model, z_mean, X_orig, vae_creator, transf,
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
            recon_mse=recon_mse,
            transformation_results=transf_results,
        ))

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
    _plot_summary_heatmap(results, output_dir)


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
    output_dir: Path,
) -> None:
    variant_names = [r.name for r in results]
    metric_names = [
        "Mean rho", "Monotonic %", "Recon MSE", "Mean FP dist",
    ]

    data = np.zeros((len(results), len(metric_names)))
    for i, r in enumerate(results):
        rhos = [tr.spearman_rho for tr in r.transformation_results.values()]
        monos = [tr.is_monotonic for tr in r.transformation_results.values()]
        fp_dists = [tr.fingerprint_distance for tr in r.transformation_results.values()]
        data[i, 0] = np.mean(rhos)
        data[i, 1] = np.mean(monos) * 100.0
        data[i, 2] = r.recon_mse
        data[i, 3] = np.mean(fp_dists)

    fig, ax = plt.subplots(figsize=(8, max(3, len(results) * 0.8 + 1)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_yticks(range(len(variant_names)))
    ax.set_yticklabels(variant_names, fontsize=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            fmt = ".1f" if j == 1 else ".3f"
            ax.text(j, i, f"{data[i, j]:{fmt}}", ha="center", va="center",
                    fontsize=9, color="black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Component Ablation Summary", fontsize=14, fontweight="bold")
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
]


if __name__ == "__main__":
    output_dir = Path("assets/results/component_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_results = run_component_ablation(STANDARD_CONFIGS)
    plot_component_ablation(
        ablation_results,
        STANDARD_CONFIGS[0].sigma_values,
        output_dir,
    )
    print_component_ablation_report(ablation_results)
