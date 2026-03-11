"""
Transformation signature analysis: verifies that the transformation applied
in latent space is faithfully reflected in the decoded synthetic data. For
each transformation, three fingerprint metrics are computed on the residuals
(synthetic - original) to characterize the transformation type. LGTA output
should match direct augmentation fingerprints.
"""

from dataclasses import dataclass

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
    valid_mask: np.ndarray | None = None,
) -> TransformationFingerprint:
    """Compute the transformation fingerprint from residuals averaged across
    all series. If valid_mask is provided (True = observed), only observed
    positions are used per series."""
    residual = X_synth - X_orig
    T, S = residual.shape

    autocorrs: list[float] = []
    linearities: list[float] = []
    amp_deps: list[float] = []

    for s in range(S):
        r = residual[:, s].copy()
        orig_s = X_orig[:, s]
        if valid_mask is not None:
            valid = np.asarray(valid_mask[:, s], dtype=bool)
            if np.sum(valid) < 2:
                autocorrs.append(0.0)
                linearities.append(0.0)
                amp_deps.append(0.0)
                continue
            r = r[valid]
            orig_s = orig_s[valid]

        if np.std(r) < 1e-12:
            autocorrs.append(0.0)
            linearities.append(0.0)
            amp_deps.append(0.0)
            continue

        n_valid = len(r)
        ac = np.corrcoef(r[:-1], r[1:])[0, 1]
        autocorrs.append(float(ac) if not np.isnan(ac) else 0.0)

        t_axis = np.arange(n_valid, dtype=float)
        coeffs = np.polyfit(t_axis, r, 1)
        fitted = np.polyval(coeffs, t_axis)
        ss_res = np.sum((r - fitted) ** 2)
        ss_tot = np.sum((r - r.mean()) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        linearities.append(float(r_squared))

        if np.std(orig_s) > 1e-12:
            ad = float(np.abs(np.corrcoef(np.abs(r), np.abs(orig_s))[0, 1]))
            if np.isnan(ad):
                ad = 0.0
        else:
            ad = 0.0
        amp_deps.append(ad)

    n_series = len(autocorrs)
    return TransformationFingerprint(
        autocorrelation=float(np.mean(autocorrs)) if n_series else 0.0,
        linearity=float(np.mean(linearities)) if n_series else 0.0,
        amplitude_dependence=float(np.mean(amp_deps)) if n_series else 0.0,
    )
