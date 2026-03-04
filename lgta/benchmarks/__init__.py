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


def get_benchmark_generators_lgta(seed: int = 42) -> list[TimeSeriesGenerator]:
    """Benchmark generators for the lgta env (no TSDiff). Use with downstream_forecasting --mode lgta."""
    return [
        TimeGANGenerator(seed=seed),
        TimeVAEGenerator(seed=seed),
        DirectTransformGenerator(transformation="jitter", sigma=0.5, seed=seed),
    ]


def get_benchmark_generators_tsdiff(seed: int = 42) -> list[TimeSeriesGenerator]:
    """Benchmark generators for the lgta-tsdiff env (TSDiff only). Use with downstream_forecasting --mode tsdiff."""
    if _TSDiff_AVAILABLE and TSDiffGenerator is not None:
        return [TSDiffGenerator(seed=seed)]
    return []


__all__ = [
    "TimeSeriesGenerator",
    "DirectTransformGenerator",
    "TimeGANGenerator",
    "TimeVAEGenerator",
    "TSDiffGenerator",
    "get_default_benchmark_generators",
    "get_benchmark_generators_lgta",
    "get_benchmark_generators_tsdiff",
]
