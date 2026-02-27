"""
Evaluation metrics for time series augmentation using pymdma framework.

All metrics come from pymdma's synthesis validation module, organized by:
- Fidelity: How realistic are the augmented series (feature-based quality)
- Diversity: Are augmented series diverse yet meaningful (feature-based quality)
- Privacy: Are augmented series genuinely novel, not memorized (feature-based privacy)
- Data Quality: Are key signal properties preserved (data-based quality)

pymdma reference: https://pymdma.readthedocs.io/en/latest/time_series/synth_val/
"""

import numpy as np
from typing import Dict, List
import warnings

from pymdma.time_series.measures.synthesis_val import (
    ImprovedPrecision,
    ImprovedRecall,
    Density,
    Coverage,
    FrechetDistance,
    WassersteinDistance,
    MMD,
    CosineSimilarity,
    Authenticity,
    DTW,
    CrossCorrelation,
    SpectralCoherence,
    SpectralWassersteinDistance,
)

warnings.filterwarnings("ignore")


def _to_features(X: np.ndarray) -> np.ndarray:
    """Convert (n_timesteps, n_series) to (n_series, n_timesteps) for feature-based metrics."""
    return X.T


def _to_signals(X: np.ndarray) -> List[np.ndarray]:
    """Convert (n_timesteps, n_series) to list of (n_timesteps, 1) arrays for data-based metrics."""
    return [X[:, i : i + 1] for i in range(X.shape[1])]


class FidelityMetrics:
    """Measures how realistic augmented series are.

    ImprovedPrecision and Density measure fidelity via manifold estimation.
    FrechetDistance, WassersteinDistance, MMD, CosineSimilarity measure
    distributional distance between original and augmented feature sets.
    """

    def __init__(self, k: int = 5):
        self._precision = ImprovedPrecision(k=k)
        self._density = Density(k=k)
        self._frechet = FrechetDistance(compute_ratios=False)
        self._wasserstein = WassersteinDistance(compute_ratios=False)
        self._mmd = MMD(kernel="linear", compute_ratios=False)
        self._cosine = CosineSimilarity()

    def compute(self, X_orig: np.ndarray, X_augmented: np.ndarray) -> Dict[str, float]:
        real_feats = _to_features(X_orig)
        fake_feats = _to_features(X_augmented)

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
    """Measures diversity of augmented series.

    ImprovedRecall measures how well the generated data covers the real distribution.
    Coverage is a robust alternative that is less sensitive to outliers.
    """

    def __init__(self, k: int = 5):
        self._recall = ImprovedRecall(k=k)
        self._coverage = Coverage(k=k)

    def compute(self, X_orig: np.ndarray, X_augmented: np.ndarray) -> Dict[str, float]:
        real_feats = _to_features(X_orig)
        fake_feats = _to_features(X_augmented)

        return {
            "improved_recall": self._recall.compute(
                real_feats, fake_feats
            ).dataset_level.value,
            "coverage": self._coverage.compute(
                real_feats, fake_feats
            ).dataset_level.value,
        }


class PrivacyMetrics:
    """Measures whether augmented series are genuinely novel.

    Authenticity checks that synthetic samples are sufficiently distinct from
    any real sample, detecting memorization/copying.
    """

    def __init__(self):
        self._authenticity = Authenticity()

    def compute(self, X_orig: np.ndarray, X_augmented: np.ndarray) -> Dict[str, float]:
        real_feats = _to_features(X_orig)
        fake_feats = _to_features(X_augmented)

        return {
            "authenticity": self._authenticity.compute(
                real_feats, fake_feats
            ).dataset_level.value,
        }


class DataQualityMetrics:
    """Measures preservation of signal properties using pymdma data-based metrics.

    DTW and CrossCorrelation compare temporal structure.
    SpectralCoherence and SpectralWassersteinDistance compare frequency-domain
    properties and require a sampling frequency (fs) parameter per dataset.
    DTW is O(N*M) so we sample a subset of series for efficiency.
    """

    def __init__(self, sampling_freq: int = 1, max_series: int = 30):
        self._dtw = DTW()
        self._cross_corr = CrossCorrelation(mode="full", reduction="max")
        self._spectral_coherence = SpectralCoherence(fs=sampling_freq)
        self._spectral_wasserstein = SpectralWassersteinDistance(fs=sampling_freq)
        self._max_series = max_series

    def compute(self, X_orig: np.ndarray, X_augmented: np.ndarray) -> Dict[str, float]:
        n_series = X_orig.shape[1]
        n_sample = min(n_series, self._max_series)

        if n_series > n_sample:
            indices = np.random.choice(n_series, n_sample, replace=False)
        else:
            indices = np.arange(n_series)

        ref_sigs = _to_signals(X_orig[:, indices])
        tgt_sigs = _to_signals(X_augmented[:, indices])

        dtw_result = self._dtw.compute(ref_sigs, tgt_sigs)
        cc_result = self._cross_corr.compute(ref_sigs, tgt_sigs)

        real_spectral = _to_features(X_orig)
        fake_spectral = _to_features(X_augmented)
        spectral_coherence_val: float = float("nan")
        spectral_wasserstein_val: float = float("nan")
        try:
            sc_result = self._spectral_coherence.compute(real_spectral, fake_spectral)
            swd_result = self._spectral_wasserstein.compute(
                real_spectral, fake_spectral
            )
            spectral_coherence_val = float(sc_result.dataset_level.value)
            spectral_wasserstein_val = float(swd_result.dataset_level.value)
        except ValueError:
            pass

        return {
            "dtw": float(dtw_result.dataset_level.value),
            "cross_correlation": float(cc_result.dataset_level.value),
            "spectral_coherence": spectral_coherence_val,
            "spectral_wasserstein": spectral_wasserstein_val,
        }


class MetricsAggregator:
    """Aggregates all pymdma metrics for comprehensive evaluation.

    Categories: fidelity, diversity, privacy, data_quality.
    The sampling_freq parameter should be set per dataset (e.g. 12 for monthly,
    52 for weekly, 365 for daily) and is passed to spectral metrics.
    """

    def __init__(
        self,
        k: int = 5,
        sampling_freq: int = 1,
        max_series_for_dtw: int = 30,
    ):
        self.fidelity = FidelityMetrics(k=k)
        self.diversity = DiversityMetrics(k=k)
        self.privacy = PrivacyMetrics()
        self.data_quality = DataQualityMetrics(
            sampling_freq=sampling_freq, max_series=max_series_for_dtw
        )

    def compute_all_metrics(
        self,
        X_orig: np.ndarray,
        X_lgta: np.ndarray,
        X_benchmark: np.ndarray,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        results: Dict[str, Dict[str, Dict[str, float]]] = {
            "lgta": {},
            "benchmark": {},
        }

        for method, X_aug in [("lgta", X_lgta), ("benchmark", X_benchmark)]:
            results[method]["fidelity"] = self.fidelity.compute(X_orig, X_aug)
            results[method]["diversity"] = self.diversity.compute(X_orig, X_aug)
            results[method]["privacy"] = self.privacy.compute(X_orig, X_aug)
            results[method]["data_quality"] = self.data_quality.compute(X_orig, X_aug)

        return results
