"""Benchmark time series generators for downstream forecasting comparison."""

from lgta.benchmarks.base import TimeSeriesGenerator
from lgta.benchmarks.direct import DirectTransformGenerator
from lgta.benchmarks.timegan import TimeGANGenerator
from lgta.benchmarks.timevae import TimeVAEGenerator


def get_default_benchmark_generators(seed: int = 42) -> list[TimeSeriesGenerator]:
    """Return the default list of benchmark generators."""
    return [
        TimeGANGenerator(seed=seed),
        TimeVAEGenerator(seed=seed),
        DirectTransformGenerator(transformation="jitter", sigma=0.5, seed=seed),
    ]


__all__ = [
    "TimeSeriesGenerator",
    "DirectTransformGenerator",
    "TimeGANGenerator",
    "TimeVAEGenerator",
    "get_default_benchmark_generators",
]
