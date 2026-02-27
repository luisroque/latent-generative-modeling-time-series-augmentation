"""
Statistical hypothesis testing for comparing LGTA vs Benchmark approaches.

This module provides statistical tests to determine if observed differences
between methods are statistically significant.
"""

import numpy as np
from typing import Dict, Tuple, List
from scipy import stats
from dataclasses import dataclass


@dataclass
class TestResult:
    """Result of a statistical test."""

    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: float
    interpretation: str
    winner: str  # 'lgta', 'benchmark', or 'tie'


class StatisticalTester:
    """Performs statistical tests to compare LGTA vs Benchmark."""

    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical tester.
        
        Args:
            alpha: Significance level for hypothesis tests
        """
        self.alpha = alpha

    def paired_t_test(
        self, lgta_values: np.ndarray, benchmark_values: np.ndarray, metric_name: str,
        lower_is_better: bool = True
    ) -> TestResult:
        """
        Perform paired t-test comparing LGTA vs Benchmark.
        
        Args:
            lgta_values: Metric values for LGTA
            benchmark_values: Metric values for benchmark
            metric_name: Name of the metric being tested
            lower_is_better: Whether lower values are better for this metric
            
        Returns:
            TestResult object with test outcomes
        """
        # Ensure we have numeric arrays
        try:
            lgta_values = np.asarray(lgta_values, dtype=float)
            benchmark_values = np.asarray(benchmark_values, dtype=float)
        except (ValueError, TypeError):
            return TestResult(
                test_name=f"Paired t-test: {metric_name}",
                statistic=np.nan,
                p_value=np.nan,
                is_significant=False,
                effect_size=np.nan,
                interpretation="Cannot convert to numeric values",
                winner="tie",
            )
        
        # Remove NaN values
        mask = ~(np.isnan(lgta_values) | np.isnan(benchmark_values))
        lgta_clean = lgta_values[mask]
        benchmark_clean = benchmark_values[mask]

        if len(lgta_clean) < 3:
            return TestResult(
                test_name=f"Paired t-test: {metric_name}",
                statistic=np.nan,
                p_value=np.nan,
                is_significant=False,
                effect_size=np.nan,
                interpretation="Insufficient data for testing",
                winner="tie",
            )

        # Perform paired t-test
        statistic, p_value = stats.ttest_rel(lgta_clean, benchmark_clean)

        # Calculate effect size (Cohen's d)
        differences = lgta_clean - benchmark_clean
        effect_size = np.mean(differences) / (np.std(differences, ddof=1) + 1e-8)

        # Determine winner
        is_significant = p_value < self.alpha
        if not is_significant:
            winner = "tie"
            interpretation = f"No significant difference (p={p_value:.4f})"
        else:
            lgta_mean = np.mean(lgta_clean)
            benchmark_mean = np.mean(benchmark_clean)
            
            if lower_is_better:
                winner = "lgta" if lgta_mean < benchmark_mean else "benchmark"
                interpretation = (
                    f"{'LGTA' if winner == 'lgta' else 'Benchmark'} is significantly better "
                    f"(p={p_value:.4f}, effect_size={effect_size:.3f})"
                )
            else:
                winner = "lgta" if lgta_mean > benchmark_mean else "benchmark"
                interpretation = (
                    f"{'LGTA' if winner == 'lgta' else 'Benchmark'} is significantly better "
                    f"(p={p_value:.4f}, effect_size={effect_size:.3f})"
                )

        return TestResult(
            test_name=f"Paired t-test: {metric_name}",
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            effect_size=float(effect_size),
            interpretation=interpretation,
            winner=winner,
        )

    def wilcoxon_test(
        self, lgta_values: np.ndarray, benchmark_values: np.ndarray, metric_name: str,
        lower_is_better: bool = True
    ) -> TestResult:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to t-test).
        
        Args:
            lgta_values: Metric values for LGTA
            benchmark_values: Metric values for benchmark
            metric_name: Name of the metric being tested
            lower_is_better: Whether lower values are better for this metric
            
        Returns:
            TestResult object with test outcomes
        """
        # Ensure we have numeric arrays
        try:
            lgta_values = np.asarray(lgta_values, dtype=float)
            benchmark_values = np.asarray(benchmark_values, dtype=float)
        except (ValueError, TypeError):
            return TestResult(
                test_name=f"Wilcoxon test: {metric_name}",
                statistic=np.nan,
                p_value=np.nan,
                is_significant=False,
                effect_size=np.nan,
                interpretation="Cannot convert to numeric values",
                winner="tie",
            )
        
        # Remove NaN values
        mask = ~(np.isnan(lgta_values) | np.isnan(benchmark_values))
        lgta_clean = lgta_values[mask]
        benchmark_clean = benchmark_values[mask]

        if len(lgta_clean) < 3:
            return TestResult(
                test_name=f"Wilcoxon test: {metric_name}",
                statistic=np.nan,
                p_value=np.nan,
                is_significant=False,
                effect_size=np.nan,
                interpretation="Insufficient data for testing",
                winner="tie",
            )

        # Perform Wilcoxon test
        try:
            statistic, p_value = stats.wilcoxon(lgta_clean, benchmark_clean)
        except ValueError:
            return TestResult(
                test_name=f"Wilcoxon test: {metric_name}",
                statistic=np.nan,
                p_value=np.nan,
                is_significant=False,
                effect_size=np.nan,
                interpretation="Test failed (zero differences)",
                winner="tie",
            )

        # Calculate effect size (rank-biserial correlation)
        differences = lgta_clean - benchmark_clean
        n = len(differences)
        r = 1 - (2 * statistic) / (n * (n + 1))

        # Determine winner
        is_significant = p_value < self.alpha
        if not is_significant:
            winner = "tie"
            interpretation = f"No significant difference (p={p_value:.4f})"
        else:
            lgta_median = np.median(lgta_clean)
            benchmark_median = np.median(benchmark_clean)
            
            if lower_is_better:
                winner = "lgta" if lgta_median < benchmark_median else "benchmark"
                interpretation = (
                    f"{'LGTA' if winner == 'lgta' else 'Benchmark'} is significantly better "
                    f"(p={p_value:.4f}, effect_size={r:.3f})"
                )
            else:
                winner = "lgta" if lgta_median > benchmark_median else "benchmark"
                interpretation = (
                    f"{'LGTA' if winner == 'lgta' else 'Benchmark'} is significantly better "
                    f"(p={p_value:.4f}, effect_size={r:.3f})"
                )

        return TestResult(
            test_name=f"Wilcoxon test: {metric_name}",
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            effect_size=float(r),
            interpretation=interpretation,
            winner=winner,
        )

    def bootstrap_test(
        self,
        lgta_values: np.ndarray,
        benchmark_values: np.ndarray,
        metric_name: str,
        lower_is_better: bool = True,
        n_bootstrap: int = 1000,
    ) -> TestResult:
        """
        Perform bootstrap hypothesis test.
        
        Args:
            lgta_values: Metric values for LGTA
            benchmark_values: Metric values for benchmark
            metric_name: Name of the metric being tested
            lower_is_better: Whether lower values are better
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            TestResult object with test outcomes
        """
        # Ensure we have numeric arrays
        try:
            lgta_values = np.asarray(lgta_values, dtype=float)
            benchmark_values = np.asarray(benchmark_values, dtype=float)
        except (ValueError, TypeError):
            return TestResult(
                test_name=f"Bootstrap test: {metric_name}",
                statistic=np.nan,
                p_value=np.nan,
                is_significant=False,
                effect_size=np.nan,
                interpretation="Cannot convert to numeric values",
                winner="tie",
            )
        
        # Remove NaN values
        mask = ~(np.isnan(lgta_values) | np.isnan(benchmark_values))
        lgta_clean = lgta_values[mask]
        benchmark_clean = benchmark_values[mask]

        if len(lgta_clean) < 3:
            return TestResult(
                test_name=f"Bootstrap test: {metric_name}",
                statistic=np.nan,
                p_value=np.nan,
                is_significant=False,
                effect_size=np.nan,
                interpretation="Insufficient data for testing",
                winner="tie",
            )

        # Observed difference
        obs_diff = np.mean(lgta_clean) - np.mean(benchmark_clean)

        # Bootstrap resampling
        differences = []
        for _ in range(n_bootstrap):
            lgta_sample = np.random.choice(lgta_clean, size=len(lgta_clean), replace=True)
            bench_sample = np.random.choice(
                benchmark_clean, size=len(benchmark_clean), replace=True
            )
            diff = np.mean(lgta_sample) - np.mean(bench_sample)
            differences.append(diff)

        differences = np.array(differences)

        # Calculate p-value (two-tailed)
        if lower_is_better:
            p_value = np.mean(differences >= 0)  # Probability of LGTA being worse
        else:
            p_value = np.mean(differences <= 0)  # Probability of LGTA being worse

        # Effect size as standardized difference
        effect_size = obs_diff / (np.std(differences) + 1e-8)

        # Determine winner
        is_significant = p_value < self.alpha
        if not is_significant:
            winner = "tie"
            interpretation = f"No significant difference (p={p_value:.4f})"
        else:
            if lower_is_better:
                winner = "lgta" if obs_diff < 0 else "benchmark"
            else:
                winner = "lgta" if obs_diff > 0 else "benchmark"
            interpretation = (
                f"{'LGTA' if winner == 'lgta' else 'Benchmark'} is significantly better "
                f"(p={p_value:.4f}, effect_size={effect_size:.3f})"
            )

        return TestResult(
            test_name=f"Bootstrap test: {metric_name}",
            statistic=float(obs_diff),
            p_value=float(p_value),
            is_significant=is_significant,
            effect_size=float(effect_size),
            interpretation=interpretation,
            winner=winner,
        )

    def comprehensive_comparison(
        self,
        lgta_metrics: Dict[str, np.ndarray],
        benchmark_metrics: Dict[str, np.ndarray],
        metrics_config: Dict[str, bool],
    ) -> Dict[str, List[TestResult]]:
        """
        Perform comprehensive statistical comparison across all metrics.
        
        Args:
            lgta_metrics: Dictionary of metric_name -> values for LGTA
            benchmark_metrics: Dictionary of metric_name -> values for benchmark
            metrics_config: Dictionary of metric_name -> lower_is_better flag
            
        Returns:
            Dictionary of test results organized by metric
        """
        results = {}

        for metric_name in lgta_metrics.keys():
            if metric_name not in benchmark_metrics:
                continue

            lgta_vals = lgta_metrics[metric_name]
            bench_vals = benchmark_metrics[metric_name]
            lower_is_better = metrics_config.get(metric_name, True)

            # Ensure arrays
            if not isinstance(lgta_vals, np.ndarray):
                lgta_vals = np.array([lgta_vals])
            if not isinstance(bench_vals, np.ndarray):
                bench_vals = np.array([bench_vals])

            # Run multiple tests
            test_results = []

            # Parametric test
            t_result = self.paired_t_test(
                lgta_vals, bench_vals, metric_name, lower_is_better
            )
            test_results.append(t_result)

            # Non-parametric test
            w_result = self.wilcoxon_test(
                lgta_vals, bench_vals, metric_name, lower_is_better
            )
            test_results.append(w_result)

            # Bootstrap test
            b_result = self.bootstrap_test(
                lgta_vals, bench_vals, metric_name, lower_is_better
            )
            test_results.append(b_result)

            results[metric_name] = test_results

        return results


def summarize_test_results(
    test_results: Dict[str, List[TestResult]]
) -> Dict[str, any]:
    """
    Summarize test results to determine overall winner.
    
    Args:
        test_results: Dictionary of metric_name -> list of TestResults
        
    Returns:
        Summary dictionary with overall assessment
    """
    lgta_wins = 0
    benchmark_wins = 0
    ties = 0
    total_tests = 0

    detailed_results = []

    for metric_name, results in test_results.items():
        # Use majority voting across tests for this metric
        metric_winners = [r.winner for r in results]
        lgta_count = metric_winners.count("lgta")
        bench_count = metric_winners.count("benchmark")

        if lgta_count > bench_count:
            lgta_wins += 1
            metric_winner = "lgta"
        elif bench_count > lgta_count:
            benchmark_wins += 1
            metric_winner = "benchmark"
        else:
            ties += 1
            metric_winner = "tie"

        total_tests += 1

        # Get the most confident result
        sig_results = [r for r in results if r.is_significant]
        if sig_results:
            best_result = min(sig_results, key=lambda x: x.p_value)
            interpretation = best_result.interpretation
        else:
            interpretation = "No significant difference across tests"

        detailed_results.append(
            {
                "metric": metric_name,
                "winner": metric_winner,
                "interpretation": interpretation,
                "all_test_results": results,
            }
        )

    # Determine overall winner
    if lgta_wins > benchmark_wins:
        overall_winner = "LGTA"
        confidence = lgta_wins / total_tests
    elif benchmark_wins > lgta_wins:
        overall_winner = "Benchmark"
        confidence = benchmark_wins / total_tests
    else:
        overall_winner = "Tie"
        confidence = ties / total_tests

    return {
        "overall_winner": overall_winner,
        "confidence": confidence,
        "lgta_wins": lgta_wins,
        "benchmark_wins": benchmark_wins,
        "ties": ties,
        "total_metrics": total_tests,
        "detailed_results": detailed_results,
    }

