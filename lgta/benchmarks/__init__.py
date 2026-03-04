"""Benchmark time series generators for downstream forecasting comparison."""

from lgta.benchmarks.base import TimeSeriesGenerator
from lgta.benchmarks.direct import DirectTransformGenerator
from lgta.benchmarks.timegan import TimeGANGenerator
from lgta.benchmarks.timevae import TimeVAEGenerator

try:
    from lgta.benchmarks.tsdiff import TSDiffGenerator
    _TSDiff_AVAILABLE = True
except ImportError:
    TSDiffGenerator = None  # type: ignore[misc, assignment]
    _TSDiff_AVAILABLE = False


def get_default_benchmark_generators(seed: int = 42) -> list[TimeSeriesGenerator]:
    """Return the default list of benchmark generators. TSDiff is included only
    when uncond-ts-diff is installed to avoid dependency clashes in the main env."""
    generators: list[TimeSeriesGenerator] = [
        TimeGANGenerator(seed=seed),
        TimeVAEGenerator(seed=seed),
        DirectTransformGenerator(transformation="jitter", sigma=0.5, seed=seed),
    ]
    if _TSDiff_AVAILABLE and TSDiffGenerator is not None:
        generators.insert(2, TSDiffGenerator(seed=seed))
    return generators


__all__ = [
    "TimeSeriesGenerator",
    "DirectTransformGenerator",
    "TimeGANGenerator",
    "TimeVAEGenerator",
    "TSDiffGenerator",
    "get_default_benchmark_generators",
]
