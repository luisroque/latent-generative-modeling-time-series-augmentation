"""Benchmark time series generators for downstream forecasting comparison."""

from lgta.benchmarks.base import TimeSeriesGenerator
from lgta.benchmarks.baseline_vae import BaselineVAEGenerator
from lgta.benchmarks.diffusion_ts import DiffusionTSGenerator
from lgta.benchmarks.direct import DirectTransformGenerator
from lgta.benchmarks.timegan import TimeGANGenerator
from lgta.benchmarks.timevae import TimeVAEGenerator
from lgta.benchmarks.timevae_windowed import TimeVAEWindowedGenerator


def get_default_benchmark_generators(
    seed: int = 42, window_size: int = 10
) -> list[TimeSeriesGenerator]:
    """Return the default list of benchmark generators.

    window_size is forwarded to windowed generators so they stay aligned
    with the experiment's (and LGTA CVAE's) window size.
    """
    return [
        TimeGANGenerator(seed=seed),
        TimeVAEGenerator(seed=seed),
        TimeVAEWindowedGenerator(seed=seed, window_size=window_size),
        BaselineVAEGenerator(seed=seed),
        DiffusionTSGenerator(seed=seed),
        DirectTransformGenerator(transformation="jitter", sigma=0.5, seed=seed),
    ]


__all__ = [
    "TimeSeriesGenerator",
    "BaselineVAEGenerator",
    "DirectTransformGenerator",
    "DiffusionTSGenerator",
    "TimeGANGenerator",
    "TimeVAEGenerator",
    "TimeVAEWindowedGenerator",
    "get_default_benchmark_generators",
]
