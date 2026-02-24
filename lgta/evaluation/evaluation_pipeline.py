"""
Comprehensive evaluation pipeline for comparing LGTA vs Benchmark approaches.

This module orchestrates the entire evaluation process, running multiple
experiments, collecting metrics, performing statistical tests, and generating
comprehensive reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from lgta.evaluation.metrics import MetricsAggregator
from lgta.evaluation.statistical_tests import StatisticalTester, summarize_test_results
from lgta.evaluation.evaluation_comparison import preprocess_train_evaluate
from lgta.model.generate_data import generate_datasets

FREQ_TO_SAMPLING_FREQ: Dict[str, int] = {
    "Y": 1,
    "A": 1,
    "Q": 4,
    "M": 12,
    "W": 52,
    "D": 365,
    "H": 8760,
}


@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipeline.

    sampling_freq is used by spectral metrics (e.g. 12 for monthly, 52 for weekly).
    When left at default 1, it is derived from freq.
    """

    dataset_name: str
    freq: str
    transformation_type: str
    transformation_params: List[float]
    benchmark_params: Dict[str, float]
    benchmark_version: int
    n_repetitions: int = 10
    sampling_freq: int = 1
    output_dir: Optional[Path] = None
    alpha: float = 0.05

    def __post_init__(self):
        if self.sampling_freq == 1:
            self.sampling_freq = FREQ_TO_SAMPLING_FREQ.get(self.freq.upper().strip(), 1)
        if self.output_dir is None:
            self.output_dir = Path(
                f"assets/results/{self.dataset_name}_{self.transformation_type}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class EvaluationResults:
    """Results from a single evaluation run."""

    config: EvaluationConfig
    metrics_lgta: Dict[str, Any] = field(default_factory=dict)
    metrics_benchmark: Dict[str, Any] = field(default_factory=dict)
    downstream_performance: Optional[pd.DataFrame] = None
    test_results: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


class EvaluationPipeline:
    """Orchestrates comprehensive evaluation of LGTA vs Benchmark."""

    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluation pipeline.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.metrics_aggregator = MetricsAggregator(sampling_freq=config.sampling_freq)
        self.statistical_tester = StatisticalTester(alpha=config.alpha)
        self.results: List[EvaluationResults] = []
        self._sample_series_data: Optional[
            Tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = None

    def run_single_evaluation(
        self,
        model,
        z: np.ndarray,
        create_dataset_vae,
        X_orig: np.ndarray,
        run_id: int = 0,
    ) -> EvaluationResults:
        """
        Run a single evaluation experiment.

        Args:
            model: Trained VAE model
            z: Latent representations
            create_dataset_vae: Dataset creation object
            X_orig: Original data
            run_id: Identifier for this run

        Returns:
            EvaluationResults object
        """
        print(f"\n{'='*80}")
        print(f"Running Evaluation #{run_id + 1}/{self.config.n_repetitions}")
        print(f"Transformation: {self.config.transformation_type}")
        print(f"Parameters: {self.config.transformation_params}")
        print(f"{'='*80}\n")

        # Generate augmented datasets
        X_orig_generated, X_lgta, X_benchmark = generate_datasets(
            self.config.dataset_name,
            self.config.freq,
            model,
            z,
            create_dataset_vae,
            X_orig,
            self.config.transformation_type,
            self.config.transformation_params,
            self.config.benchmark_params,
            self.config.benchmark_version,
        )

        if run_id == 0:
            self._sample_series_data = (X_orig_generated, X_lgta, X_benchmark)

        # Compute all metrics
        print("Computing comprehensive metrics...")
        all_metrics = self.metrics_aggregator.compute_all_metrics(
            X_orig_generated, X_lgta, X_benchmark
        )

        # Compute downstream performance (TSTR)
        print("Evaluating downstream task performance...")
        try:
            downstream_results = preprocess_train_evaluate(
                X_orig_generated,
                X_lgta,
                X_benchmark,
                n_features=X_orig_generated.shape[1],
            )
        except Exception as e:
            print(f"Warning: Downstream evaluation failed: {e}")
            downstream_results = None

        # Store results
        result = EvaluationResults(
            config=self.config,
            metrics_lgta=all_metrics["lgta"],
            metrics_benchmark=all_metrics["benchmark"],
            downstream_performance=downstream_results,
        )

        return result

    def run_multiple_evaluations(
        self,
        model,
        z: np.ndarray,
        create_dataset_vae,
        X_orig: np.ndarray,
    ) -> List[EvaluationResults]:
        """
        Run multiple evaluation experiments for statistical robustness.

        Args:
            model: Trained VAE model
            z: Latent representations
            create_dataset_vae: Dataset creation object
            X_orig: Original data

        Returns:
            List of EvaluationResults
        """
        results = []

        for i in range(self.config.n_repetitions):
            result = self.run_single_evaluation(
                model, z, create_dataset_vae, X_orig, run_id=i
            )
            results.append(result)

        self.results = results
        return results

    def aggregate_metrics(
        self, results: List[EvaluationResults]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Aggregate metrics across multiple runs.

        Args:
            results: List of evaluation results

        Returns:
            Tuple of (lgta_metrics, benchmark_metrics) where each is a dict
            mapping metric_name to array of values across runs
        """
        lgta_aggregated = {}
        benchmark_aggregated = {}

        # Flatten nested metric dictionaries (exclude non-numeric metrics)
        for result in results:
            for category, metrics in result.metrics_lgta.items():
                for metric_name, value in metrics.items():
                    # Skip non-numeric metrics (like diagnosis dictionaries)
                    if isinstance(value, (dict, list, str)):
                        continue
                    key = f"{category}.{metric_name}"
                    if key not in lgta_aggregated:
                        lgta_aggregated[key] = []
                    lgta_aggregated[key].append(value)

            for category, metrics in result.metrics_benchmark.items():
                for metric_name, value in metrics.items():
                    # Skip non-numeric metrics (like diagnosis dictionaries)
                    if isinstance(value, (dict, list, str)):
                        continue
                    key = f"{category}.{metric_name}"
                    if key not in benchmark_aggregated:
                        benchmark_aggregated[key] = []
                    benchmark_aggregated[key].append(value)

        # Convert to arrays
        lgta_aggregated = {k: np.array(v) for k, v in lgta_aggregated.items()}
        benchmark_aggregated = {k: np.array(v) for k, v in benchmark_aggregated.items()}

        return lgta_aggregated, benchmark_aggregated

    def perform_statistical_tests(
        self,
        lgta_metrics: Dict[str, np.ndarray],
        benchmark_metrics: Dict[str, np.ndarray],
    ) -> Dict[str, any]:
        """
        Perform comprehensive statistical testing.

        Args:
            lgta_metrics: Aggregated LGTA metrics
            benchmark_metrics: Aggregated benchmark metrics

        Returns:
            Dictionary with test results and summary
        """
        metrics_config = {
            "fidelity.improved_precision": False,
            "fidelity.density": False,
            "fidelity.frechet_distance": True,
            "fidelity.wasserstein_distance": True,
            "fidelity.mmd": True,
            "fidelity.cosine_similarity": True,
            "diversity.improved_recall": False,
            "diversity.coverage": False,
            "privacy.authenticity": False,
            "data_quality.dtw": True,
            "data_quality.cross_correlation": False,
            "data_quality.spectral_coherence": False,
            "data_quality.spectral_wasserstein": True,
        }

        print("\nPerforming statistical tests...")
        test_results = self.statistical_tester.comprehensive_comparison(
            lgta_metrics, benchmark_metrics, metrics_config
        )

        summary = summarize_test_results(test_results)

        return {"test_results": test_results, "summary": summary}

    def generate_report(self, save: bool = True) -> Dict[str, any]:
        """
        Generate comprehensive evaluation report.

        Args:
            save: Whether to save report to disk

        Returns:
            Dictionary with complete report
        """
        if not self.results:
            raise ValueError("No results available. Run evaluations first.")

        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("=" * 80 + "\n")

        # Aggregate metrics
        lgta_metrics, benchmark_metrics = self.aggregate_metrics(self.results)

        # Perform statistical tests
        statistical_results = self.perform_statistical_tests(
            lgta_metrics, benchmark_metrics
        )

        # Aggregate downstream performance
        downstream_summary = self._aggregate_downstream_performance()

        # Compute category-level winners
        category_winners = self._compute_category_winners(
            statistical_results["test_results"]
        )

        # Create report dictionary
        report = {
            "config": {
                "dataset": self.config.dataset_name,
                "transformation": self.config.transformation_type,
                "params": self.config.transformation_params,
                "n_repetitions": self.config.n_repetitions,
            },
            "metrics_summary": {
                "lgta": self._compute_metrics_summary(lgta_metrics),
                "benchmark": self._compute_metrics_summary(benchmark_metrics),
            },
            "statistical_tests": statistical_results["summary"],
            "category_winners": category_winners,
            "downstream_performance": downstream_summary,
            "detailed_test_results": statistical_results["test_results"],
        }

        # Print summary
        self._print_summary(report)

        # Save report
        if save:
            self._save_report(report)

        # Generate visualizations
        self._generate_visualizations(lgta_metrics, benchmark_metrics, category_winners)

        return report

    def _compute_metrics_summary(
        self, metrics: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for metrics."""
        summary = {}
        for metric_name, values in metrics.items():
            summary[metric_name] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        return summary

    def _compute_category_winners(
        self, test_results: Dict[str, List]
    ) -> Dict[str, Dict[str, any]]:
        """
        Compute winners for each category (realism, consistency, etc.).

        Args:
            test_results: Test results from statistical tests

        Returns:
            Dictionary with winner per category
        """
        categories = {
            "fidelity": [],
            "diversity": [],
            "privacy": [],
            "data_quality": [],
        }

        # Group metrics by category
        for metric_name, results in test_results.items():
            # Extract category from metric name (e.g., "realism.wasserstein_mean" -> "realism")
            category = metric_name.split(".")[0]
            if category in categories:
                # Use majority voting across tests
                winners = [r.winner for r in results]
                lgta_count = winners.count("lgta")
                bench_count = winners.count("benchmark")

                if lgta_count > bench_count:
                    metric_winner = "lgta"
                elif bench_count > lgta_count:
                    metric_winner = "benchmark"
                else:
                    metric_winner = "tie"

                categories[category].append(metric_winner)

        # Determine winner for each category
        category_winners = {}
        for category, winners in categories.items():
            if not winners:
                continue

            lgta_wins = winners.count("lgta")
            bench_wins = winners.count("benchmark")
            ties = winners.count("tie")

            if lgta_wins > bench_wins:
                overall_winner = "lgta"
            elif bench_wins > lgta_wins:
                overall_winner = "benchmark"
            else:
                overall_winner = "tie"

            category_winners[category] = {
                "winner": overall_winner,
                "lgta_wins": lgta_wins,
                "benchmark_wins": bench_wins,
                "ties": ties,
                "total_metrics": len(winners),
            }

        return category_winners

    def _aggregate_downstream_performance(self) -> Optional[Dict[str, any]]:
        """Aggregate downstream performance across runs."""
        valid_results = [
            r.downstream_performance
            for r in self.results
            if r.downstream_performance is not None
        ]

        if not valid_results:
            return None

        # Extract MSE values
        mse_lgta = []
        mse_benchmark = []
        mse_real = []

        for df in valid_results:
            if "L-GTA" in df.columns:
                mse_lgta.append(df.loc["MSE", "L-GTA"])
            if "Benchmark" in df.columns:
                mse_benchmark.append(df.loc["MSE", "Benchmark"])
            if "Real" in df.columns:
                mse_real.append(df.loc["MSE", "Real"])

        summary = {}
        if mse_lgta:
            summary["lgta_mse"] = {
                "mean": float(np.mean(mse_lgta)),
                "std": float(np.std(mse_lgta)),
            }
        if mse_benchmark:
            summary["benchmark_mse"] = {
                "mean": float(np.mean(mse_benchmark)),
                "std": float(np.std(mse_benchmark)),
            }
        if mse_real:
            summary["real_mse"] = {
                "mean": float(np.mean(mse_real)),
                "std": float(np.std(mse_real)),
            }

        return summary

    def _print_summary(self, report: Dict[str, any]):
        """Print human-readable summary."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)

        metrics_lgta = report["metrics_summary"]["lgta"]
        metrics_bench = report["metrics_summary"]["benchmark"]

        # Overall winner
        summary = report["statistical_tests"]
        print(f"\n🏆 OVERALL WINNER: {summary['overall_winner']}")
        print(f"   Confidence: {summary['confidence']:.1%}")
        print(
            f"   Wins: LGTA={summary['lgta_wins']}, "
            f"Benchmark={summary['benchmark_wins']}, Ties={summary['ties']}"
        )

        # Category-level winners
        if "category_winners" in report:
            print("\n" + "-" * 80)
            print("WINNERS BY CATEGORY")
            print("-" * 80)
            category_winners = report["category_winners"]
            for category, winner_info in category_winners.items():
                winner = winner_info["winner"]
                lgta_wins = winner_info["lgta_wins"]
                bench_wins = winner_info["benchmark_wins"]
                ties = winner_info["ties"]

                emoji = (
                    "🟢"
                    if winner == "lgta"
                    else ("🔴" if winner == "benchmark" else "🟡")
                )
                print(f"\n{category.upper()}: {emoji} {winner.upper()}")
                print(
                    f"  Metrics won: LGTA={lgta_wins}, Benchmark={bench_wins}, Ties={ties}"
                )

        # Key metrics comparison by category
        print("\n" + "-" * 80)
        print("KEY METRICS COMPARISON BY CATEGORY")
        print("-" * 80)

        metrics_lgta = report["metrics_summary"]["lgta"]
        metrics_bench = report["metrics_summary"]["benchmark"]

        metrics_by_category = {
            "FIDELITY": [
                "fidelity.improved_precision",
                "fidelity.density",
                "fidelity.frechet_distance",
                "fidelity.wasserstein_distance",
                "fidelity.mmd",
            ],
            "DIVERSITY": [
                "diversity.improved_recall",
                "diversity.coverage",
            ],
            "PRIVACY": [
                "privacy.authenticity",
            ],
            "DATA QUALITY": [
                "data_quality.dtw",
                "data_quality.cross_correlation",
                "data_quality.spectral_coherence",
                "data_quality.spectral_wasserstein",
            ],
        }

        lower_is_better = {
            "fidelity.frechet_distance",
            "fidelity.wasserstein_distance",
            "fidelity.mmd",
            "data_quality.dtw",
            "data_quality.spectral_wasserstein",
        }

        for category_name, metric_list in metrics_by_category.items():
            print(f"\n{category_name}:")
            for metric in metric_list:
                if metric in metrics_lgta and metric in metrics_bench:
                    lgta_val = metrics_lgta[metric]["mean"]
                    bench_val = metrics_bench[metric]["mean"]

                    # Determine winner
                    if metric in lower_is_better:
                        winner_mark = "✅" if lgta_val < bench_val else "❌"
                    else:
                        winner_mark = "✅" if lgta_val > bench_val else "❌"

                    metric_short = metric.split(".")[-1].replace("_", " ").title()
                    print(f"  {metric_short}:")
                    print(
                        f"    LGTA:      {lgta_val:.4f} {winner_mark if lgta_val != bench_val else '='}"
                    )
                    print(
                        f"    Benchmark: {bench_val:.4f} {winner_mark if bench_val != lgta_val else '='}"
                    )

        # Downstream performance
        if report["downstream_performance"]:
            print("\n" + "-" * 80)
            print("DOWNSTREAM TASK PERFORMANCE (MSE)")
            print("-" * 80)
            ds = report["downstream_performance"]
            if "lgta_mse" in ds:
                print(
                    f"LGTA:      {ds['lgta_mse']['mean']:.4f} ± {ds['lgta_mse']['std']:.4f}"
                )
            if "benchmark_mse" in ds:
                print(
                    f"Benchmark: {ds['benchmark_mse']['mean']:.4f} ± {ds['benchmark_mse']['std']:.4f}"
                )
            if "real_mse" in ds:
                print(
                    f"Real:      {ds['real_mse']['mean']:.4f} ± {ds['real_mse']['std']:.4f}"
                )

        print("\n" + "=" * 80 + "\n")

    def _save_report(self, report: Dict[str, any]):
        """Save report to JSON file."""
        # Remove non-serializable objects
        serializable_report = self._make_serializable(report)

        output_file = self.config.output_dir / "evaluation_report.json"
        with open(output_file, "w") as f:
            json.dump(serializable_report, f, indent=2)

        print(f"Report saved to: {output_file}")

    def _plot_series_comparison(self) -> None:
        """
        Plot original and synthetic series overlaid: Original vs LGTA and Original vs
        Benchmark for each sample series. Transformation and params are shown once
        (same for both methods). Uses the first run's data from _sample_series_data.
        """
        X_orig, X_lgta, X_benchmark = self._sample_series_data
        n_series = X_orig.shape[0]
        n_show = min(4, n_series)
        indices = np.linspace(0, n_series - 1, n_show, dtype=int)

        transformation = self.config.transformation_type
        params = self.config.transformation_params
        params_str = ", ".join(f"{p:.2g}" for p in params)

        fig, axes = plt.subplots(
            n_show, 2, figsize=(12, 3 * n_show), sharex="col", sharey="row"
        )
        if n_show == 1:
            axes = axes.reshape(1, -1)

        # Original: black, solid, drawn first. Synthetic: softer colors, alpha, dashed, on top.
        color_orig = "black"
        color_lgta = "#F4A460"
        color_bench = "#50A060"
        alpha_synthetic = 0.75
        linestyle_synthetic = "--"

        for row, idx in enumerate(indices):
            orig_series = np.atleast_1d(np.squeeze(X_orig[idx]))
            lgta_series = np.atleast_1d(np.squeeze(X_lgta[idx]))
            bench_series = np.atleast_1d(np.squeeze(X_benchmark[idx]))

            ax_lgta = axes[row, 0]
            ax_lgta.plot(orig_series, color=color_orig, linewidth=1.2, label="Original")
            ax_lgta.plot(
                lgta_series,
                color=color_lgta,
                linewidth=1.2,
                alpha=alpha_synthetic,
                linestyle=linestyle_synthetic,
                label="LGTA",
            )
            ax_lgta.set_title("Original vs LGTA" if row == 0 else "")
            ax_lgta.set_ylabel(f"Series {idx}\nValue")
            ax_lgta.legend(loc="upper right", fontsize=8)
            ax_lgta.grid(True, alpha=0.3)

            ax_bench = axes[row, 1]
            ax_bench.plot(
                orig_series, color=color_orig, linewidth=1.2, label="Original"
            )
            ax_bench.plot(
                bench_series,
                color=color_bench,
                linewidth=1.2,
                alpha=alpha_synthetic,
                linestyle=linestyle_synthetic,
                label="Benchmark",
            )
            ax_bench.set_title("Original vs Benchmark" if row == 0 else "")
            ax_bench.set_ylabel("")
            ax_bench.legend(loc="upper right", fontsize=8)
            ax_bench.grid(True, alpha=0.3)

        for col in range(2):
            axes[-1, col].set_xlabel("Time")

        fig.suptitle(
            f"Series comparison — transformation: {transformation}, params: [{params_str}]",
            fontsize=14,
            fontweight="bold",
            y=1.002,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        output_file = self.config.output_dir / "series_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"  ✓ Series comparison saved to: {output_file}")
        plt.close()

    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _generate_visualizations(
        self,
        lgta_metrics: Dict[str, np.ndarray],
        benchmark_metrics: Dict[str, np.ndarray],
        category_winners: Dict[str, Dict[str, Any]],
    ):
        """Generate comparison visualizations per category."""
        print("\nGenerating visualizations...")

        if self._sample_series_data is not None:
            self._plot_series_comparison()

        metrics_by_category = {
            "fidelity": [
                ("fidelity.improved_precision", "Improved Precision", "higher"),
                ("fidelity.density", "Density", "higher"),
                ("fidelity.frechet_distance", "Frechet Distance", "lower"),
                ("fidelity.wasserstein_distance", "Wasserstein Distance", "lower"),
                ("fidelity.mmd", "MMD", "lower"),
                ("fidelity.cosine_similarity", "Cosine Similarity", "lower"),
            ],
            "diversity": [
                ("diversity.improved_recall", "Improved Recall", "higher"),
                ("diversity.coverage", "Coverage", "higher"),
            ],
            "privacy": [
                ("privacy.authenticity", "Authenticity", "higher"),
            ],
            "data_quality": [
                ("data_quality.dtw", "DTW Distance", "lower"),
                ("data_quality.cross_correlation", "Cross-Correlation", "higher"),
                ("data_quality.spectral_coherence", "Spectral Coherence", "higher"),
                ("data_quality.spectral_wasserstein", "Spectral Wasserstein", "lower"),
            ],
        }

        # Generate one plot per category
        for category, metrics in metrics_by_category.items():
            # Filter metrics that exist in the data
            available_metrics = [
                (key, label, direction)
                for key, label, direction in metrics
                if key in lgta_metrics and key in benchmark_metrics
            ]

            if not available_metrics:
                continue

            n_metrics = len(available_metrics)
            if n_metrics == 0:
                continue

            # Determine subplot layout
            if n_metrics <= 2:
                nrows, ncols = 1, n_metrics
                figsize = (6 * n_metrics, 5)
            elif n_metrics <= 4:
                nrows, ncols = 2, 2
                figsize = (12, 10)
            else:
                nrows = (n_metrics + 2) // 3
                ncols = 3
                figsize = (15, 5 * nrows)

            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            if n_metrics == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if nrows * ncols > 1 else [axes]

            # Get category winner info
            winner_info = category_winners.get(category, {})
            category_winner = winner_info.get("winner", "unknown")
            winner_emoji = (
                "🟢 LGTA"
                if category_winner == "lgta"
                else ("🔴 Benchmark" if category_winner == "benchmark" else "🟡 Tie")
            )

            for idx, (metric_key, metric_label, direction) in enumerate(
                available_metrics
            ):
                ax = axes[idx]
                data = [lgta_metrics[metric_key], benchmark_metrics[metric_key]]
                labels = ["LGTA", "Benchmark"]

                # Create box plot
                bp = ax.boxplot(data, labels=labels, patch_artist=True)

                # Color boxes based on which is better
                lgta_mean = np.mean(lgta_metrics[metric_key])
                bench_mean = np.mean(benchmark_metrics[metric_key])

                if direction == "lower":
                    lgta_better = lgta_mean < bench_mean
                else:
                    lgta_better = lgta_mean > bench_mean

                if lgta_better:
                    bp["boxes"][0].set_facecolor("#90EE90")  # Light green
                    bp["boxes"][1].set_facecolor("#FFB6C1")  # Light red
                else:
                    bp["boxes"][0].set_facecolor("#FFB6C1")  # Light red
                    bp["boxes"][1].set_facecolor("#90EE90")  # Light green

                # Add title with direction indicator
                direction_text = (
                    "↓ lower is better"
                    if direction == "lower"
                    else "↑ higher is better"
                )
                ax.set_title(
                    f"{metric_label}\n({direction_text})",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.set_ylabel("Value", fontsize=10)
                ax.grid(axis="y", alpha=0.3)

                # Add mean values as text
                ax.text(
                    0.02,
                    0.98,
                    f"LGTA: {lgta_mean:.4f}\nBench: {bench_mean:.4f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            # Hide unused subplots
            for idx in range(len(available_metrics), len(axes)):
                axes[idx].axis("off")

            # Overall title
            fig.suptitle(
                f"{category.upper()} Metrics Comparison - Winner: {winner_emoji}",
                fontsize=16,
                fontweight="bold",
                y=0.995,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.985])

            # Save plot
            output_file = self.config.output_dir / f"comparison_{category}.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"  ✓ {category.capitalize()} comparison saved to: {output_file}")
            plt.close()

        # Generate overall summary plot
        self._generate_summary_plot(category_winners)

    def _generate_summary_plot(self, category_winners: Dict[str, Dict[str, any]]):
        """Generate a summary plot showing winners per category."""
        if not category_winners:
            return

        categories = list(category_winners.keys())
        lgta_wins_list = [category_winners[cat]["lgta_wins"] for cat in categories]
        bench_wins_list = [
            category_winners[cat]["benchmark_wins"] for cat in categories
        ]
        ties_list = [category_winners[cat]["ties"] for cat in categories]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(categories))
        width = 0.25

        bars1 = ax.bar(
            x - width, lgta_wins_list, width, label="LGTA Wins", color="#4CAF50"
        )
        bars2 = ax.bar(
            x, bench_wins_list, width, label="Benchmark Wins", color="#F44336"
        )
        bars3 = ax.bar(x + width, ties_list, width, label="Ties", color="#FFC107")

        ax.set_xlabel("Category", fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of Metrics Won", fontsize=12, fontweight="bold")
        ax.set_title(
            "Winners per Category: LGTA vs Benchmark", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels([cat.capitalize() for cat in categories])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        plt.tight_layout()
        output_file = self.config.output_dir / "category_summary.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"  ✓ Category summary saved to: {output_file}")
        plt.close()


def run_evaluation_pipeline(
    config: EvaluationConfig,
    model,
    z: np.ndarray,
    create_dataset_vae,
    X_orig: np.ndarray,
) -> Dict[str, any]:
    """
    Convenience function to run complete evaluation pipeline.

    Args:
        config: Evaluation configuration
        model: Trained VAE model
        z: Latent representations
        create_dataset_vae: Dataset creation object
        X_orig: Original data

    Returns:
        Complete evaluation report
    """
    pipeline = EvaluationPipeline(config)

    # Run multiple evaluations
    pipeline.run_multiple_evaluations(model, z, create_dataset_vae, X_orig)

    # Generate and return report
    report = pipeline.generate_report(save=True)

    return report
