"""
Transformation signature analysis: verifies that the transformation applied
in latent space is faithfully reflected in the decoded synthetic data. For
each transformation, three fingerprint metrics are computed on the residuals
(synthetic - original) to characterize the transformation type. LGTA output
should match direct augmentation fingerprints.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TransformationFingerprint:
    """Three-metric fingerprint that uniquely identifies a transformation type.

    - autocorrelation: lag-1 autocorrelation of residuals. High for smooth
      transforms (magnitude_warp, drift, trend); low for i.i.d. (jitter, scaling).
    - linearity: R-squared of a linear fit to the residuals. High only for
      trend; low for all others.
    - amplitude_dependence: absolute Pearson correlation between |residuals|
      and |original|. High for multiplicative transforms (scaling, magnitude_warp);
      low for additive (jitter, drift, trend).
    """

    autocorrelation: float
    linearity: float
    amplitude_dependence: float


def compute_fingerprint(
    X_orig: np.ndarray,
    X_synth: np.ndarray,
) -> TransformationFingerprint:
    """Compute the transformation fingerprint from residuals averaged across
    all series in the dataset."""
    residual = X_synth - X_orig
    T, S = residual.shape

    autocorrs: list[float] = []
    linearities: list[float] = []
    amp_deps: list[float] = []

    t_axis = np.arange(T, dtype=float)
    for s in range(S):
        r = residual[:, s]
        if np.std(r) < 1e-12:
            autocorrs.append(0.0)
            linearities.append(0.0)
            amp_deps.append(0.0)
            continue

        ac = np.corrcoef(r[:-1], r[1:])[0, 1]
        autocorrs.append(float(ac))

        coeffs = np.polyfit(t_axis, r, 1)
        fitted = np.polyval(coeffs, t_axis)
        ss_res = np.sum((r - fitted) ** 2)
        ss_tot = np.sum((r - r.mean()) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        linearities.append(float(r_squared))

        orig_s = X_orig[:, s]
        if np.std(orig_s) > 1e-12:
            ad = float(np.abs(np.corrcoef(np.abs(r), np.abs(orig_s))[0, 1]))
        else:
            ad = 0.0
        amp_deps.append(ad)

    return TransformationFingerprint(
        autocorrelation=float(np.mean(autocorrs)),
        linearity=float(np.mean(linearities)),
        amplitude_dependence=float(np.mean(amp_deps)),
    )


def plot_fingerprint_comparison(
    transformations: list[str],
    lgta_fps: dict[str, TransformationFingerprint],
    direct_fps: dict[str, TransformationFingerprint],
    output_dir: Path,
) -> Path:
    """Grouped bar chart comparing LGTA vs Direct fingerprints."""
    metrics = ["autocorrelation", "linearity", "amplitude_dependence"]
    metric_labels = ["Autocorrelation", "Linearity (R²)", "Amplitude Dep."]
    n_transf = len(transformations)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_transf)
    width = 0.35

    for ax, metric, label in zip(axes, metrics, metric_labels):
        lgta_vals = [getattr(lgta_fps[t], metric) for t in transformations]
        direct_vals = [getattr(direct_fps[t], metric) for t in transformations]

        ax.bar(x - width / 2, lgta_vals, width, label="L-GTA", color="#2196F3")
        ax.bar(x + width / 2, direct_vals, width, label="Direct", color="#F44336")
        ax.set_xticks(x)
        ax.set_xticklabels(transformations, rotation=30, ha="right", fontsize=9)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        "Transformation Fingerprint: L-GTA vs Direct",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    output_file = output_dir / "fingerprint_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Fingerprint plot saved to: {output_file}")
    plt.close()
    return output_file


def plot_residual_comparison(
    X_orig: np.ndarray,
    lgta_samples: dict[str, np.ndarray],
    direct_samples: dict[str, np.ndarray],
    transformations: list[str],
    series_idx: int,
    output_dir: Path,
) -> Path:
    """Plot residuals (synthetic - original) for each transformation,
    LGTA vs Direct side by side, to visually confirm the transformation
    character is preserved."""
    n_transf = len(transformations)
    fig, axes = plt.subplots(n_transf, 2, figsize=(14, 3 * n_transf), sharex=True)
    if n_transf == 1:
        axes = axes.reshape(1, -1)

    orig_series = X_orig[:, series_idx]

    for row, transf in enumerate(transformations):
        lgta_residual = lgta_samples[transf][:, series_idx] - orig_series
        direct_residual = direct_samples[transf][:, series_idx] - orig_series

        ax_l = axes[row, 0]
        ax_l.plot(lgta_residual, color="#2196F3", linewidth=1.2)
        ax_l.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax_l.set_ylabel(transf, fontsize=10)
        if row == 0:
            ax_l.set_title("L-GTA Residuals", fontsize=12, fontweight="bold")
        ax_l.grid(True, alpha=0.2)

        ax_r = axes[row, 1]
        ax_r.plot(direct_residual, color="#F44336", linewidth=1.2)
        ax_r.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        if row == 0:
            ax_r.set_title("Direct Residuals", fontsize=12, fontweight="bold")
        ax_r.grid(True, alpha=0.2)

    axes[-1, 0].set_xlabel("Time step")
    axes[-1, 1].set_xlabel("Time step")
    fig.suptitle(
        f"Residual Character Comparison \u2014 series {series_idx}",
        fontsize=15,
        fontweight="bold",
        y=1.002,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    output_file = output_dir / "residual_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Residual comparison plot saved to: {output_file}")
    plt.close()
    return output_file


def print_signature_report(
    transformations: list[str],
    lgta_fps: dict[str, TransformationFingerprint],
    direct_fps: dict[str, TransformationFingerprint],
) -> None:
    """Print a comparison table of fingerprint metrics."""
    print("\n" + "=" * 80)
    print("TRANSFORMATION SIGNATURE VERIFICATION")
    print("=" * 80)
    header = (
        f"{'Transformation':<16s}  "
        f"{'AutoCorr(L)':>11s} {'AutoCorr(D)':>11s}  "
        f"{'Linear(L)':>10s} {'Linear(D)':>10s}  "
        f"{'AmpDep(L)':>10s} {'AmpDep(D)':>10s}"
    )
    print(header)
    print("-" * 80)
    for t in transformations:
        lf = lgta_fps[t]
        df = direct_fps[t]
        print(
            f"{t:<16s}  "
            f"{lf.autocorrelation:11.4f} {df.autocorrelation:11.4f}  "
            f"{lf.linearity:10.4f} {df.linearity:10.4f}  "
            f"{lf.amplitude_dependence:10.4f} {df.amplitude_dependence:10.4f}"
        )

    print("\nKey: (L)=L-GTA, (D)=Direct")
    print("Matching values indicate the transformation character is preserved.")
    print("=" * 80)
