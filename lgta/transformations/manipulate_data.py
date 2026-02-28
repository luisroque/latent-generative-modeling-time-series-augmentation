"""
Time series augmentation transformations applied to raw data or latent space
representations. Every transformation is parameterized by a single sigma value
that directly controls the perturbation magnitude, keeping all transformations
on the same conceptual scale.
"""

from scipy.interpolate import CubicSpline
import numpy as np


TransformationName = str

TRANSFORMATION_REGISTRY: dict[TransformationName, str] = {
    "jitter": "_jitter",
    "scaling": "_scaling",
    "magnitude_warp": "_magnitude_warp",
    "time_warp": "_time_warp",
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

    def _time_warp(self, knot: int = 4) -> np.ndarray:
        random_warps = np.random.normal(
            loc=1.0, scale=self.sigma, size=(knot + 2, self.x.shape[1])
        )
        warp_steps = np.linspace(0, self.x.shape[0] - 1.0, num=knot + 2)
        time_warp = np.zeros((self.x.shape[0], self.x.shape[1]))
        ret = np.zeros((self.x.shape[0], self.x.shape[1]))

        for i in range(self.x.shape[1]):
            time_warp[:, i] = CubicSpline(warp_steps, warp_steps * random_warps[:, i])(
                self.orig_steps
            )
            ret[:, i] = np.interp(self.orig_steps, time_warp[:, i], self.x[:, i])
        return ret

    def apply_transf(self) -> np.ndarray:
        method_name = TRANSFORMATION_REGISTRY.get(self.transformation)
        if method_name is None:
            raise ValueError(
                f"Unknown transformation '{self.transformation}'. "
                f"Valid options: {list(TRANSFORMATION_REGISTRY.keys())}"
            )
        return getattr(self, method_name)()
