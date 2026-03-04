"""
Direct transformation benchmark: applies classical signal-level augmentations
(jitter, scaling, magnitude warp, drift, trend) to the original data matrix
without any learned generative model.
"""

import numpy as np

from lgta.benchmarks.base import TimeSeriesGenerator
from lgta.transformations.manipulate_data import ManipulateData


class DirectTransformGenerator(TimeSeriesGenerator):
    """Applies a single named augmentation to the original data at generation
    time. The ``fit`` call simply stores the (scaled) data."""

    def __init__(
        self,
        transformation: str = "jitter",
        sigma: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.transformation = transformation
        self.sigma = sigma
        self._data_scaled: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"Direct({self.transformation})"

    def _fit(self, data: np.ndarray) -> None:
        self._data_scaled = data.copy()

    def _generate(self) -> np.ndarray:
        assert self._data_scaled is not None
        return ManipulateData(
            x=self._data_scaled,
            transformation=self.transformation,
            parameters=[self.sigma],
        ).apply_transf()
