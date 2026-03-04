"""
Abstract interface for time series generative benchmarks. Every benchmark
generator receives a data matrix of shape (n_timesteps, n_series), learns to
produce synthetic series, and returns a matrix of the same shape.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TimeSeriesGenerator(ABC):
    """Base class for all time series augmentation / generation benchmarks."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._scaler = StandardScaler()
        self._n_timesteps: int = 0
        self._n_series: int = 0
        self.device = _get_device()

    def fit(self, data: np.ndarray) -> "TimeSeriesGenerator":
        """Fit on *original* data of shape ``(n_timesteps, n_series)``.

        Applies per-series standardisation, stores shape metadata, then
        delegates to the subclass ``_fit`` hook.
        """
        self._n_timesteps, self._n_series = data.shape
        self._scaler.fit(data)
        data_scaled = self._scaler.transform(data)
        self._fit(data_scaled)
        return self

    def generate(self) -> np.ndarray:
        """Return synthetic data of shape ``(n_timesteps, n_series)``."""
        data_scaled = self._generate()
        return self._scaler.inverse_transform(data_scaled)

    @abstractmethod
    def _fit(self, data: np.ndarray) -> None:
        """Subclass training hook. *data* is already standardised."""

    @abstractmethod
    def _generate(self) -> np.ndarray:
        """Subclass generation hook. Must return standardised data."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
