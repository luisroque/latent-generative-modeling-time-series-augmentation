"""
Improved pipeline for objective evaluation of LGTA vs Benchmark transformations.

This script provides a comprehensive, objective comparison using:
1. Multiple evaluation metrics across 5 dimensions
2. Statistical hypothesis testing
3. Multiple experimental runs for robustness
4. Automated reporting and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from lgta.evaluation.evaluation_pipeline import (
    run_evaluation_pipeline,
    EvaluationConfig,
)


def run_comprehensive_evaluation(
    dataset: str = "tourism_small",
    freq: str = "Q",
    n_repetitions: int = 10,
):
    """
    Run comprehensive evaluation for a dataset.

    Args:
        dataset: Dataset name ('tourism_small' or 'synthetic')
        freq: Frequency ('Q', 'D', etc.)
        n_repetitions: Number of evaluation runs for statistical robustness
    """
    print("=" * 80)
    print(f"COMPREHENSIVE EVALUATION: {dataset.upper()} DATASET")
    print("=" * 80)

    print("\n[1/4] Training VAE model...")
    create_dataset_vae = CreateTransformedVersionsCVAE(
        dataset_name=dataset, freq=freq
    )
    model, _, _ = create_dataset_vae.fit()
    X_hat, z, _, _ = create_dataset_vae.predict(model)
    X_orig = create_dataset_vae.X_train_raw

    print(f"   ✓ Model trained. Data shape: {X_orig.shape}")

    # Step 2: Define transformations to evaluate
    print("\n[2/4] Defining transformations to evaluate...")

    # Transformation configurations based on the dataset
    if dataset == "tourism_small":
        transformations = [
            {
                "name": "jitter",
                "params": [0.5],
                "benchmark_params": {
                    "jitter": 0.5,
                    "scaling": 0.1,
                    "magnitude_warp": 0.1,
                    "time_warp": 0.05,
                },
                "version": 5,
            },
            {
                "name": "scaling",
                "params": [0.25],
                "benchmark_params": {
                    "jitter": 0.375,
                    "scaling": 0.1,
                    "magnitude_warp": 0.1,
                    "time_warp": 0.05,
                },
                "version": 4,
            },
            {
                "name": "magnitude_warp",
                "params": [0.1],
                "benchmark_params": {
                    "jitter": 0.375,
                    "scaling": 0.1,
                    "magnitude_warp": 0.1,
                    "time_warp": 0.05,
                },
                "version": 4,
            },
        ]
    else:
        # Default configuration for other datasets
        transformations = [
            {
                "name": "jitter",
                "params": [0.5],
                "benchmark_params": {
                    "jitter": 0.5,
                    "scaling": 0.1,
                    "magnitude_warp": 0.1,
                    "time_warp": 0.05,
                },
                "version": 5,
            },
        ]

    print(f"   ✓ Evaluating {len(transformations)} transformation(s)")

    # Step 3: Run evaluations for each transformation
    print("\n[3/4] Running comprehensive evaluations...")
    all_reports = {}

    for trans_config in transformations:
        print(f"\n{'─' * 80}")
        print(f"Evaluating: {trans_config['name'].upper()}")
        print(f"{'─' * 80}")

        # Create evaluation configuration
        eval_config = EvaluationConfig(
            dataset_name=dataset,
            freq=freq,
            transformation_type=trans_config["name"],
            transformation_params=trans_config["params"],
            benchmark_params=trans_config["benchmark_params"],
            benchmark_version=trans_config["version"],
            n_repetitions=n_repetitions,
            output_dir=Path(f"assets/results/{dataset}_{trans_config['name']}"),
        )

        # Run evaluation pipeline
        report = run_evaluation_pipeline(
            config=eval_config,
            model=model,
            z=z,
            create_dataset_vae=create_dataset_vae,
            X_orig=X_orig,
        )

        all_reports[trans_config["name"]] = report

    # Step 4: Generate summary across all transformations
    print("\n[4/4] Generating overall summary...")
    generate_overall_summary(all_reports, dataset)

    return all_reports


def generate_overall_summary(all_reports: dict, dataset: str):
    """
    Generate summary comparing results across all transformations.

    Args:
        all_reports: Dictionary of transformation_name -> report
        dataset: Dataset name
    """
    print("\n" + "=" * 80)
    print(f"OVERALL SUMMARY FOR {dataset.upper()} DATASET")
    print("=" * 80)

    # Count wins per method
    lgta_wins = 0
    benchmark_wins = 0
    ties = 0

    print("\nResults by Transformation:")
    print("-" * 80)

    for trans_name, report in all_reports.items():
        winner = report["statistical_tests"]["overall_winner"]
        confidence = report["statistical_tests"]["confidence"]

        print(f"\n{trans_name.upper()}:")
        print(f"  Winner: {winner} (confidence: {confidence:.1%})")

        if winner == "LGTA":
            lgta_wins += 1
            emoji = "🟢"
        elif winner == "Benchmark":
            benchmark_wins += 1
            emoji = "🔴"
        else:
            ties += 1
            emoji = "🟡"

        print(f"  {emoji} Status")

    # Overall assessment
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print(f"\nAcross {len(all_reports)} transformation(s):")
    print(f"  LGTA wins:      {lgta_wins}")
    print(f"  Benchmark wins: {benchmark_wins}")
    print(f"  Ties:           {ties}")

    if lgta_wins > benchmark_wins:
        print("\n✅ CONCLUSION: LGTA approach is objectively better")
    elif benchmark_wins > lgta_wins:
        print("\n❌ CONCLUSION: Benchmark approach is objectively better")
    else:
        print("\n⚖️  CONCLUSION: No clear winner (methods are comparable)")

    print("\n" + "=" * 80 + "\n")


def compare_transformation_magnitudes(
    dataset: str = "tourism_small",
    freq: str = "Q",
    transformation: str = "jitter",
    magnitudes: list = None,
    n_repetitions: int = 5,
):
    """
    Compare LGTA vs Benchmark across different transformation magnitudes.

    Args:
        dataset: Dataset name
        freq: Frequency
        top: Top N series
        transformation: Transformation type
        magnitudes: List of magnitude values to test
        n_repetitions: Number of evaluation runs per magnitude
    """
    if magnitudes is None:
        magnitudes = [0.3, 0.5, 0.7, 0.9]

    print("=" * 80)
    print(f"MAGNITUDE COMPARISON: {transformation.upper()}")
    print("=" * 80)

    # Train model once
    print("\nTraining VAE model...")
    create_dataset_vae = CreateTransformedVersionsCVAE(
        dataset_name=dataset, freq=freq
    )
    model, _, _ = create_dataset_vae.fit()
    X_hat, z, _, _ = create_dataset_vae.predict(model)
    X_orig = create_dataset_vae.X_train_raw

    results_by_magnitude = {}

    for magnitude in magnitudes:
        print(f"\n{'─' * 80}")
        print(f"Evaluating magnitude: {magnitude}")
        print(f"{'─' * 80}")

        # Adjust benchmark params to match magnitude
        benchmark_params = {
            "jitter": magnitude,
            "scaling": 0.1,
            "magnitude_warp": 0.1,
            "time_warp": 0.05,
        }

        eval_config = EvaluationConfig(
            dataset_name=dataset,
            freq=freq,
            transformation_type=transformation,
            transformation_params=[magnitude],
            benchmark_params=benchmark_params,
            benchmark_version=5,
            n_repetitions=n_repetitions,
            output_dir=Path(
                f"assets/results/{dataset}_{transformation}_mag_{magnitude}"
            ),
        )

        report = run_evaluation_pipeline(
            config=eval_config,
            model=model,
            z=z,
            create_dataset_vae=create_dataset_vae,
            X_orig=X_orig,
        )

        results_by_magnitude[magnitude] = report

    # Visualize results across magnitudes
    visualize_magnitude_comparison(results_by_magnitude, transformation, dataset)

    return results_by_magnitude


def visualize_magnitude_comparison(results: dict, transformation: str, dataset: str):
    """
    Visualize how performance varies with transformation magnitude.

    Args:
        results: Dictionary of magnitude -> report
        transformation: Transformation name
        dataset: Dataset name
    """
    magnitudes = sorted(results.keys())

    # Extract key metrics
    lgta_wasserstein = []
    bench_wasserstein = []
    lgta_acf_error = []
    bench_acf_error = []
    lgta_wins = []

    for mag in magnitudes:
        report = results[mag]
        lgta_metrics = report["metrics_summary"]["lgta"]
        bench_metrics = report["metrics_summary"]["benchmark"]

        lgta_wasserstein.append(lgta_metrics["realism.wasserstein_mean"]["mean"])
        bench_wasserstein.append(bench_metrics["realism.wasserstein_mean"]["mean"])

        lgta_acf_error.append(lgta_metrics["consistency.autocorrelation_error"]["mean"])
        bench_acf_error.append(
            bench_metrics["consistency.autocorrelation_error"]["mean"]
        )

        lgta_wins.append(
            1 if report["statistical_tests"]["overall_winner"] == "LGTA" else 0
        )

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Wasserstein Distance
    axes[0].plot(magnitudes, lgta_wasserstein, marker="o", label="LGTA", linewidth=2)
    axes[0].plot(
        magnitudes, bench_wasserstein, marker="s", label="Benchmark", linewidth=2
    )
    axes[0].set_xlabel("Transformation Magnitude", fontsize=12)
    axes[0].set_ylabel("Wasserstein Distance", fontsize=12)
    axes[0].set_title("Realism: Wasserstein Distance\n(lower is better)", fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Autocorrelation Error
    axes[1].plot(magnitudes, lgta_acf_error, marker="o", label="LGTA", linewidth=2)
    axes[1].plot(
        magnitudes, bench_acf_error, marker="s", label="Benchmark", linewidth=2
    )
    axes[1].set_xlabel("Transformation Magnitude", fontsize=12)
    axes[1].set_ylabel("Autocorrelation Error", fontsize=12)
    axes[1].set_title(
        "Consistency: Autocorrelation Error\n(lower is better)", fontsize=14
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Plot 3: Win Rate
    axes[2].bar(
        range(len(magnitudes)), lgta_wins, tick_label=[str(m) for m in magnitudes]
    )
    axes[2].set_xlabel("Transformation Magnitude", fontsize=12)
    axes[2].set_ylabel("LGTA Wins (1=win, 0=loss/tie)", fontsize=12)
    axes[2].set_title("Statistical Test Results", fontsize=14)
    axes[2].set_ylim([0, 1.1])
    axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"Performance Across Magnitudes: {transformation.upper()} on {dataset.upper()}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    output_file = Path(
        f"assets/results/{dataset}_{transformation}_magnitude_comparison.png"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n📊 Magnitude comparison plot saved to: {output_file}")
    plt.close()


# Example usage
if __name__ == "__main__":
    # Configuration
    DATASET = "tourism_small"
    FREQ = "Q"
    N_REPETITIONS = 10

    print("\nStarting Comprehensive Evaluation Pipeline\n")

    reports = run_comprehensive_evaluation(
        dataset=DATASET, freq=FREQ, n_repetitions=N_REPETITIONS
    )

    # Option 2: Compare across magnitudes (uncomment to run)
    # magnitude_results = compare_transformation_magnitudes(
    #     dataset=DATASET,
    #     freq=FREQ,
    #     top=TOP,
    #     transformation="jitter",
    #     magnitudes=[0.3, 0.5, 0.7, 0.9],
    #     n_repetitions=5,
    # )

    print(
        "\n✅ Evaluation complete! Check the 'assets/results/' directory for detailed reports."
    )
