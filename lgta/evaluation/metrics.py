"""Evaluation metrics for time series augmentation using pymdma framework.

Features are extracted from each series using TSFEL (statistical, temporal,
spectral domains), then compared via pymdma feature-based metrics for
fidelity, diversity, and privacy assessment.

pymdma: https://pymdma.readthedocs.io/en/latest/time_series/synth_val/
tsfel:  https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html
"""

import hashlib
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tsfel

from pymdma.time_series.measures.synthesis_val import (
    Authenticity,
    CosineSimilarity,
    Coverage,
    Density,
    FrechetDistance,
    ImprovedPrecision,
    ImprovedRecall,
    MMD,
    WassersteinDistance,
)

warnings.filterwarnings("ignore")

_TSFEL_CFG = tsfel.get_features_by_domain()


# Explicit TSFEL feature sets per domain. Using fixed lists avoids duplicate-label
# and other TSFEL/pandas issues and makes the used feature set visible in code.
# Bump _FEATURE_CONFIG_VERSION below if any of these lists change.

TSFEL_STATISTICAL_FEATURES: tuple[str, ...] = (
    "Absolute energy",
    "Average power",
    "ECDF",
    "ECDF Percentile",
    "ECDF Percentile Count",
    "Entropy",
    "Histogram mode",
    "Interquartile range",
    "Kurtosis",
    "Max",
    "Mean",
    "Mean absolute deviation",
    "Median",
    "Median absolute deviation",
    "Min",
    "Peak to peak distance",
    "Root mean square",
    "Skewness",
    "Standard deviation",
    "Variance",
)

TSFEL_TEMPORAL_FEATURES: tuple[str, ...] = (
    "Area under the curve",
    "Autocorrelation",
    "Centroid",
    "Lempel-Ziv complexity",
    "Mean absolute diff",
    "Mean diff",
    "Median absolute diff",
    "Median diff",
    "Negative turning points",
    "Neighbourhood peaks",
    "Positive turning points",
    "Signal distance",
    "Slope",
    "Sum absolute diff",
    "Zero crossing rate",
)

TSFEL_SPECTRAL_FEATURES: tuple[str, ...] = (
    "Fundamental frequency",
    "Human range energy",
    "LPCC",
    "MFCC",
    "Max power spectrum",
    "Maximum frequency",
    "Median frequency",
    "Power bandwidth",
    "Spectral centroid",
    "Spectral decrease",
    "Spectral distance",
    "Spectral entropy",
    "Spectral kurtosis",
    "Spectral positive turning points",
    "Spectral roll-off",
    "Spectral roll-on",
    "Spectral skewness",
    "Spectral slope",
    "Spectral spread",
    "Spectral variation",
    "Spectrogram mean coefficient",
    "Wavelet entropy",
)


def _get_feature_names_from_cfg(cfg: dict) -> list[str]:
    """Return sorted list of feature names from a TSFEL domain config (e.g. {'statistical': {name: params}})."""
    names: list[str] = []
    for domain_key, inner in cfg.items():
        if isinstance(inner, dict):
            names.extend(inner.keys())
    return sorted(set(names))


def _build_domain_configs() -> dict[str, dict]:
    """Build TSFEL configs per domain using explicit feature lists (no probing, no duplicates)."""
    cfgs: dict[str, dict] = {}

    stat_full = tsfel.get_features_by_domain("statistical")
    stat_all = stat_full.get("statistical", {})
    cfgs["statistical"] = {
        "statistical": {
            name: stat_all[name]
            for name in TSFEL_STATISTICAL_FEATURES
            if name in stat_all
        }
    }

    temp_full = tsfel.get_features_by_domain("temporal")
    temp_all = temp_full.get("temporal", {})
    cfgs["temporal"] = {
        "temporal": {
            name: temp_all[name]
            for name in TSFEL_TEMPORAL_FEATURES
            if name in temp_all
        }
    }

    spec_full = tsfel.get_features_by_domain("spectral")
    spec_all = spec_full.get("spectral", {})
    cfgs["spectral"] = {
        "spectral": {
            name: spec_all[name]
            for name in TSFEL_SPECTRAL_FEATURES
            if name in spec_all
        }
    }

    print("[TSFEL] Feature sets by domain (no duplicates):", flush=True)
    for domain in ("statistical", "temporal", "spectral"):
        names = _get_feature_names_from_cfg(cfgs[domain])
        print(f"  {domain}: {len(names)} features", flush=True)
        for n in names:
            print(f"    - {n}", flush=True)
    return cfgs


_TSFEL_CFG_BY_DOMAIN: dict[str, dict] = _build_domain_configs()


_FEATURE_CONFIG_VERSION = "v2"  # bump when TSFEL feature set changes


def _feature_cache_key(X: np.ndarray, sampling_freq: int) -> str:
    """Deterministic hash from array content, shape, sampling frequency, and feature config version."""
    h = hashlib.sha256()
    h.update(X.tobytes())
    h.update(f"{X.shape}|{sampling_freq}".encode())
    h.update(_FEATURE_CONFIG_VERSION.encode())
    return h.hexdigest()[:16]


def extract_tsfel_features(
    X: np.ndarray,
    sampling_freq: int = 1,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Extract TSFEL features (statistical, temporal, spectral) from each series.

    Each column in X is treated as an independent univariate time series.
    Features that are all-NaN across series are dropped; remaining NaN/inf
    values are replaced with 0.

    When ``cache_dir`` is provided, features are saved as a ``.npy`` file keyed
    by a hash of the input data and sampling frequency.  On subsequent calls
    with identical inputs the cached file is loaded directly, skipping the
    (potentially expensive) TSFEL extraction.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(n_timesteps, n_series)``.
    sampling_freq : int
        Sampling frequency passed to TSFEL for spectral features.
    cache_dir : Path or None
        Directory for caching extracted features.  ``None`` disables caching.

    Returns
    -------
    np.ndarray
        Shape ``(n_series, n_features)``.
    """
    if cache_dir is not None:
        cache_path = cache_dir / f"tsfel_features_{_feature_cache_key(X, sampling_freq)}.npy"
        if cache_path.exists():
            return np.load(cache_path)

    feature_rows: list[np.ndarray] = []
    domains: list[str] = ["statistical", "temporal", "spectral"]

    for i in range(X.shape[1]):
        series = pd.DataFrame(X[:, i], columns=["value"])

        per_domain_feats: list[np.ndarray] = []
        for domain in domains:
            cfg = _TSFEL_CFG_BY_DOMAIN.get(domain)
            if not cfg:
                continue
            try:
                feats_df = tsfel.time_series_features_extractor(
                    cfg, series, fs=sampling_freq, verbose=0
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                print(
                    f"[TSFEL] Skipping domain '{domain}' for series {i} due to error: {exc}"
                )
                continue

            # Drop duplicate columns within this domain to avoid pandas
            # reindexing errors and log when this happens so we can monitor it.
            if feats_df.columns.duplicated().any():  # pragma: no cover - rare
                n_dup = int(feats_df.columns.duplicated().sum())
                print(
                    f"[TSFEL] Dropping {n_dup} duplicate feature columns in "
                    f"domain '{domain}' for series {i}."
                )
                feats_df = feats_df.loc[:, ~feats_df.columns.duplicated()]

            per_domain_feats.append(feats_df.values[0])

        if not per_domain_feats:
            raise RuntimeError(
                f"TSFEL feature extraction failed for all domains for series {i}."
            )

        feature_rows.append(np.concatenate(per_domain_feats, axis=-1))

    features = np.array(feature_rows, dtype=np.float64)

    valid_cols = ~np.all(np.isnan(features), axis=0)
    features = features[:, valid_cols]
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, features)

    return features


class FidelityMetrics:
    """Manifold and distributional metrics measuring how realistic augmented series are."""

    def __init__(self, k: int = 5):
        self._precision = ImprovedPrecision(k=k)
        self._density = Density(k=k)
        self._frechet = FrechetDistance(compute_ratios=False)
        self._wasserstein = WassersteinDistance(compute_ratios=False)
        self._mmd = MMD(kernel="linear", compute_ratios=False)
        self._cosine = CosineSimilarity()

    def compute(
        self, real_feats: np.ndarray, fake_feats: np.ndarray
    ) -> Dict[str, float]:
        return {
            "improved_precision": self._precision.compute(
                real_feats, fake_feats
            ).dataset_level.value,
            "density": self._density.compute(
                real_feats, fake_feats
            ).dataset_level.value,
            "frechet_distance": self._frechet.compute(
                real_feats, fake_feats
            ).dataset_level.value,
            "wasserstein_distance": self._wasserstein.compute(
                real_feats, fake_feats
            ).dataset_level.value,
            "mmd": self._mmd.compute(real_feats, fake_feats).dataset_level.value,
            "cosine_similarity": self._cosine.compute(
                real_feats, fake_feats
            ).dataset_level.value,
        }


class DiversityMetrics:
    """Manifold coverage metrics measuring diversity of augmented series."""

    def __init__(self, k: int = 5):
        self._recall = ImprovedRecall(k=k)
        self._coverage = Coverage(k=k)

    def compute(
        self, real_feats: np.ndarray, fake_feats: np.ndarray
    ) -> Dict[str, float]:
        return {
            "improved_recall": self._recall.compute(
                real_feats, fake_feats
            ).dataset_level.value,
            "coverage": self._coverage.compute(
                real_feats, fake_feats
            ).dataset_level.value,
        }


class PrivacyMetrics:
    """Authenticity metric checking augmented series are novel, not memorised."""

    def __init__(self) -> None:
        self._authenticity = Authenticity()

    def compute(
        self, real_feats: np.ndarray, fake_feats: np.ndarray
    ) -> Dict[str, float]:
        return {
            "authenticity": self._authenticity.compute(
                real_feats, fake_feats
            ).dataset_level.value,
        }


class MetricsAggregator:
    """Extracts TSFEL features once, caches them, and runs all pymdma metrics.

    Categories: fidelity, diversity, privacy.
    ``sampling_freq`` is forwarded to TSFEL for spectral feature extraction
    (e.g. 4 for quarterly, 12 for monthly, 52 for weekly, 365 for daily).
    ``cache_dir`` enables on-disk caching of the extracted feature arrays so
    repeated runs with the same data skip the TSFEL extraction step.
    """

    def __init__(
        self,
        k: int = 5,
        sampling_freq: int = 1,
        cache_dir: Path | None = None,
    ) -> None:
        self.sampling_freq = sampling_freq
        self.cache_dir = cache_dir
        self.fidelity = FidelityMetrics(k=k)
        self.diversity = DiversityMetrics(k=k)
        self.privacy = PrivacyMetrics()

    def _extract(self, X: np.ndarray) -> np.ndarray:
        return extract_tsfel_features(X, self.sampling_freq, self.cache_dir)

    def compute_all_metrics(
        self,
        X_orig: np.ndarray,
        X_lgta: np.ndarray,
        X_benchmark: np.ndarray,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        real_feats = self._extract(X_orig)

        results: Dict[str, Dict[str, Dict[str, float]]] = {}
        for method, X_aug in [("lgta", X_lgta), ("benchmark", X_benchmark)]:
            fake_feats = self._extract(X_aug)
            results[method] = {
                "fidelity": self.fidelity.compute(real_feats, fake_feats),
                "diversity": self.diversity.compute(real_feats, fake_feats),
                "privacy": self.privacy.compute(real_feats, fake_feats),
            }

        return results

    def compute_metrics_single(
        self,
        X_orig: np.ndarray,
        X_synthetic: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute fidelity, diversity, and privacy for one pair.

        Returns the same nested structure as one method entry from
        ``compute_all_metrics``.
        """
        real_feats = self._extract(X_orig)
        fake_feats = self._extract(X_synthetic)

        return {
            "fidelity": self.fidelity.compute(real_feats, fake_feats),
            "diversity": self.diversity.compute(real_feats, fake_feats),
            "privacy": self.privacy.compute(real_feats, fake_feats),
        }
