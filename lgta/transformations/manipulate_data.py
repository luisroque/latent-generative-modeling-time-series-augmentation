"""
Time series augmentation transformations applied to raw data or latent space
representations. Every transformation is parameterized by a single sigma value
that directly controls the perturbation magnitude. Only value-space
transformations are included; time-axis warps are excluded because they
break the fixed-length assumption of the CVAE latent code.
"""

from scipy.interpolate import CubicSpline
import numpy as np


TransformationName = str

TRANSFORMATION_REGISTRY: dict[TransformationName, str] = {
    "jitter": "_jitter",
    "scaling": "_scaling",
    "magnitude_warp": "_magnitude_warp",
    "drift": "_drift",
    "trend": "_trend",
}


class ManipulateData:
    def __init__(
        self,
        x: np.ndarray,
        transformation: TransformationName,
        parameters: list[float],
    ):
        self.x = np.array(x)
        self.transformation = transformation
        self.orig_steps = np.arange(self.x.shape[0])
        self.sigma = parameters[0] if parameters else 0.0

    def _jitter(self) -> np.ndarray:
        return self.x + np.random.normal(
            loc=0.0, scale=self.sigma, size=self.x.shape
        )

    def _scaling(self) -> np.ndarray:
        factor = np.random.normal(
            loc=1.0, scale=self.sigma, size=(self.x.shape[0], self.x.shape[1])
        )
        return self.x * factor

    def _magnitude_warp(self, knot: int = 4) -> np.ndarray:
        random_warps = np.random.normal(
            loc=1.0, scale=self.sigma, size=(knot + 2, self.x.shape[1])
        )
        warp_steps = np.linspace(0, self.x.shape[0] - 1.0, num=knot + 2)
        warper = np.zeros((self.x.shape[0], self.x.shape[1]))

        for i in range(self.x.shape[1]):
            warper[:, i] = np.array(
                [CubicSpline(warp_steps, random_warps[:, i])(self.orig_steps)]
            )
        return self.x * warper

    def _drift(self) -> np.ndarray:
        """Cumulative random walk perturbation. Unlike jitter (i.i.d.), the
        perturbation at each step accumulates, producing autocorrelated
        trending deviations whose magnitude scales with sigma."""
        T = self.x.shape[0]
        steps = np.random.normal(
            loc=0.0, scale=self.sigma / np.sqrt(T), size=self.x.shape
        )
        return self.x + np.cumsum(steps, axis=0)

    def _trend(self) -> np.ndarray:
        """Linear trend injection. Adds a random linear ramp per series
        with slope proportional to sigma."""
        T = self.x.shape[0]
        ramp = np.linspace(-0.5, 0.5, T).reshape(-1, 1)
        slopes = np.random.normal(loc=0.0, scale=1.0, size=(1, self.x.shape[1]))
        return self.x + self.sigma * slopes * ramp

    def apply_transf(self) -> np.ndarray:
        method_name = TRANSFORMATION_REGISTRY.get(self.transformation)
        if method_name is None:
            raise ValueError(
                f"Unknown transformation '{self.transformation}'. "
                f"Valid options: {list(TRANSFORMATION_REGISTRY.keys())}"
            )
        return getattr(self, method_name)()
