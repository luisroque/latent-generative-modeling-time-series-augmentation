from scipy.stats import wasserstein_distance
import numpy as np
from scipy.stats import iqr, skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from lgta.model.generate_data import generate_datasets
from lgta.transformations.apply_transformations_benchmark import (
    apply_transformations_and_standardize,
)
from lgta.visualization.comparison_analysis import (
    plot_transformations_comparison,
    plot_series_comparisons,
)
from lgta.transformations.compute_distances import compute_distances
from lgta.evaluation.evaluation_comparison import (
    preprocess_train_evaluate,
    plot_dimensionality_reduction,
    perform_dimensionality_reduction,
    calculate_reconstruction_error,
    kl_divergence,
)


def e2e_transformation(
    dataset,
    freq,
    model,
    z,
    create_dataset_vae,
    transformation,
    params,
    parameters_benchmark,
    version,
    X_orig,
    X_hat,
):
    """
    End-to-end processing and evaluation of data transformations.
    """
    print("-------------------------------------------------------------")
    print(f"Processing Transformation: {transformation.capitalize()}")
    print(f"Parameters: {params}")
    print("Benchmark Parameters:", parameters_benchmark)
    print("Benchmark Version:", version)
    print("-------------------------------------------------------------\n")

    transformed_datasets_benchmark = apply_transformations_and_standardize(
        dataset, freq, parameters_benchmark, standardize=False
    )
    X_orig, X_hat_transf, X_benchmark = generate_datasets(
        dataset,
        freq,
        model,
        z,
        create_dataset_vae,
        X_orig,
        transformation,
        params,
        parameters_benchmark,
        version,
    )

    # Visualize comparison between transformations
    print("\n--- Transformation Comparisons ---")
    plot_transformations_comparison(transformed_datasets_benchmark, X_orig, series=0)
    plot_series_comparisons(X_orig, X_hat_transf, X_benchmark, transformation)

    # Compute and analyze distances
    print("\n--- Distance Analysis ---")
    distances_xhat_wasserstein, distances_benchmark_wasserstein = compute_distances(
        X_hat_transf, X_orig, X_benchmark, wasserstein_distance
    )
    # plot_long_tail_comparisons(X_orig, X_hat_transf, X_benchmark, 'L-GTA Transformed', 'Benchmark Transformed', distances_xhat_wasserstein)
    summarize_and_plot_distributions(
        distances_xhat_wasserstein, distances_benchmark_wasserstein, ""
    )

    # Print statistical summaries of distances
    print("\nWasserstein Distance Analysis:")
    print(f"L-GTA (First 4 Series): {distances_xhat_wasserstein[:4]}")
    print(f"Benchmark (First 4 Series): {distances_benchmark_wasserstein[:4]}")

    # Perform and visualize dimensionality reduction
    print("\n--- Dimensionality Reduction & Visualization ---")
    perform_and_visualize_dimensionality_reduction(
        X_orig, X_hat_transf, X_benchmark, transformation
    )

    # Compute and print reconstruction errors and KL divergences
    print("\n--- Reconstruction Errors & KL Divergence ---")
    (
        reconstruction_error_orig,
        (
            reconstruction_error_lgta,
            kl_div_lgta_median,
            kl_div_lgta_iqr,
            kl_div_lgta_skew,
            kl_div_lgta_kurtosis,
        ),
        (
            reconstruction_error_benchmark,
            kl_div_benchmark_median,
            kl_div_benchmark_iqr,
            kl_div_benchmark_skew,
            kl_div_benchmark_kurtosis,
        ),
    ) = compute_and_print_errors(X_orig, X_hat_transf, X_benchmark)

    # Preprocess, train, and evaluate
    print("\n--- Model Performance Comparison ---")
    results_df = preprocess_train_evaluate(
        X_orig, X_hat, X_benchmark, n_features=X_orig.shape[1]
    )

    # Display results
    print("Model Performance Comparison:")
    print(results_df)
    return (
        (distances_xhat_wasserstein, distances_benchmark_wasserstein),
        (
            reconstruction_error_orig,
            (
                reconstruction_error_lgta,
                kl_div_lgta_median,
                kl_div_lgta_iqr,
                kl_div_lgta_skew,
                kl_div_lgta_kurtosis,
            ),
            (
                reconstruction_error_benchmark,
                kl_div_benchmark_median,
                kl_div_benchmark_iqr,
                kl_div_benchmark_skew,
                kl_div_benchmark_kurtosis,
            ),
        ),
        results_df,
    )


def perform_and_visualize_dimensionality_reduction(
    X_orig, X_hat_transf, X_benchmark, transformation
):
    pca_real, pca_synth, tsne_real, tsne_synth = perform_dimensionality_reduction(
        X_orig, X_hat_transf
    )
    pca_real_bench, pca_synth_bench, tsne_real_bench, tsne_synth_bench = (
        perform_dimensionality_reduction(X_orig, X_benchmark)
    )

    plot_dimensionality_reduction(
        pca_real,
        pca_synth,
        tsne_real,
        tsne_synth,
        f"PCA and t-SNE: Original vs {transformation.capitalize()} Transformed",
    )
    plot_dimensionality_reduction(
        pca_real_bench,
        pca_synth_bench,
        tsne_real_bench,
        tsne_synth_bench,
        f"PCA and t-SNE: Original vs {transformation.capitalize()} Benchmark",
    )


def compute_and_print_errors(X_orig, X_hat_transf, X_benchmark):
    reconstruction_error_orig = calculate_reconstruction_error(X_orig, 2)
    reconstruction_error_lgta = calculate_reconstruction_error(X_hat_transf, 2)
    reconstruction_error_benchmark = calculate_reconstruction_error(X_benchmark, 2)

    kl_div_lgta_median, kl_div_lgta_iqr, kl_div_lgta_skew, kl_div_lgta_kurtosis = (
        kl_divergence(X_orig, X_hat_transf)
    )
    (
        kl_div_benchmark_median,
        kl_div_benchmark_iqr,
        kl_div_benchmark_skew,
        kl_div_benchmark_kurtosis,
    ) = kl_divergence(X_orig, X_benchmark)

    print(f"Reconstruction Error (Original): {reconstruction_error_orig:.4f}")
    print(f"Reconstruction Error (L-GTA): {reconstruction_error_lgta:.4f}")
    print(f"Reconstruction Error (Benchmark): {reconstruction_error_benchmark:.4f}\n")

    print(
        f"KL Divergence (L-GTA): Median = {kl_div_lgta_median[0]:.4f}, IQR = {kl_div_lgta_iqr[0]:.4f}"
    )
    print(
        f"KL Divergence (Benchmark): Median = {kl_div_benchmark_median[0]:.4f}, IQR = {kl_div_benchmark_iqr[0]:.4f}"
    )
    return (
        reconstruction_error_orig,
        (
            reconstruction_error_lgta,
            kl_div_lgta_median[0],
            kl_div_lgta_iqr[0],
            kl_div_lgta_skew[0],
            kl_div_lgta_kurtosis,
        ),
        (
            reconstruction_error_benchmark,
            kl_div_benchmark_median[0],
            kl_div_benchmark_iqr[0],
            kl_div_benchmark_skew[0],
            kl_div_benchmark_kurtosis,
        ),
    )


def summarize_and_plot_distributions(distances_lgta, distances_benchmark, title):
    # Calculate metrics
    metrics_lgta = {
        "Mean": np.mean(distances_lgta),
        "Median": np.median(distances_lgta),
        "IQR": iqr(distances_lgta),
        "Skewness": skew(distances_lgta),
        "Kurtosis": kurtosis(distances_lgta),
    }

    metrics_benchmark = {
        "Mean": np.mean(distances_benchmark),
        "Median": np.median(distances_benchmark),
        "IQR": iqr(distances_benchmark),
        "Skewness": skew(distances_benchmark),
        "Kurtosis": kurtosis(distances_benchmark),
    }

    # Print metrics
    print(f"\nMetrics for L-GTA:\n{metrics_lgta}")
    print(f"\nMetrics for Benchmark:\n{metrics_benchmark}")

    # Plot distributions
    plt.figure(figsize=(14, 6))

    # Histograms
    plt.subplot(1, 2, 1)
    sns.histplot(
        distances_benchmark,
        color="blue",
        label="Benchmark",
        kde=True,
        stat="density",
        linewidth=0,
        alpha=0.5,
    )
    sns.histplot(
        distances_lgta,
        color="orange",
        label="L-GTA",
        kde=True,
        stat="density",
        linewidth=0,
    )
    plt.title(f"Distributions of Wasserstein Distances\n{title}")
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.legend()

    # Box plots
    plt.subplot(1, 2, 2)
    data_to_plot = [distances_lgta, distances_benchmark]
    plt.boxplot(data_to_plot, patch_artist=True, labels=["L-GTA", "Benchmark"])
    plt.title("Box Plot of Wasserstein Distances")
    plt.ylabel("Distance")

    plt.tight_layout()
    plt.show()
    plt.show()


def plot_magnitude_comparisons(X_orig, X_hat_transf, X_benchmark, params, n_series=4):
    """
    Creates separate plots for each time series with subplots arranged by transformation parameters
    (rows) and data type (columns: X_hat_transf and X_benchmark).
    The y-axis range is defined independently for each row.
    """
    colors = {
        "original": "black",
        "l_gta": "darkorange",
        "benchmark": "dodgerblue",
    }

    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    for series_idx in range(n_series):
        n_magnitudes = len(params)
        fig, axes = plt.subplots(
            n_magnitudes, 2, figsize=(12, 2.5 * n_magnitudes), sharex=True
        )

        fig.suptitle(
            f"Comparing L-GTA vs Benchmark for Different Magnitudes for a Single Series",
            fontsize=16,
            fontweight="bold",
        )

        for i, param in enumerate(params):
            # Calculate row-specific y-limits
            y_min = min(
                X_orig[:, series_idx].min(),
                X_hat_transf[param][:, series_idx].min(),
                X_benchmark[param][:, series_idx].min(),
            )
            y_max = max(
                X_orig[:, series_idx].max(),
                X_hat_transf[param][:, series_idx].max(),
                X_benchmark[param][:, series_idx].max(),
            )

            # Plot X_hat_transf vs Original
            axes[i, 0].plot(
                X_orig[:, series_idx],
                color=colors["original"],
                label="Original",
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
            )
            axes[i, 0].plot(
                X_hat_transf[param][:, series_idx],
                color=colors["l_gta"],
                label="L-GTA",
                linewidth=1.5,
                alpha=0.75,
            )
            axes[i, 0].set_title(f"L-GTA @ {param}")
            axes[i, 0].set_ylim(y_min, y_max)
            if i == 0:
                axes[i, 0].legend(loc="upper left", fontsize=8)
            if i == n_magnitudes - 1:
                axes[i, 0].set_xlabel("Time")
            axes[i, 0].set_ylabel("Value")

            # Plot X_benchmark vs Original
            axes[i, 1].plot(
                X_orig[:, series_idx],
                color=colors["original"],
                label="Original",
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
            )
            axes[i, 1].plot(
                X_benchmark[param][:, series_idx],
                color=colors["benchmark"],
                label="Benchmark",
                linewidth=1.5,
                alpha=0.55,
            )
            axes[i, 1].set_title(f"Benchmark @ {param}")
            axes[i, 1].set_ylim(y_min, y_max)
            if i == 0:
                axes[i, 1].legend(loc="upper left", fontsize=8)
            if i == n_magnitudes - 1:
                axes[i, 1].set_xlabel("Time")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def compare_diff_magnitudes(
    dataset,
    freq,
    model,
    z,
    create_dataset_vae,
    X_orig,
    n_series=8,
):
    transformations = [
        {
            "transformation": "magnitude_warp",
            "params": [0.1],
            "parameters_benchmark": {
                "jitter": 0.5,
                "scaling": 0.1,
                "magnitude_warp": 0.1,
                "time_warp": 0.05,
            },
            "version": 4,
        },
        {
            "transformation": "magnitude_warp",
            "params": [0.15],
            "parameters_benchmark": {
                "jitter": 0.7,
                "scaling": 0.1,
                "magnitude_warp": 0.15,
                "time_warp": 0.05,
            },
            "version": 4,
        },
        {
            "transformation": "magnitude_warp",
            "params": [0.2],
            "parameters_benchmark": {
                "jitter": 0.9,
                "scaling": 0.1,
                "magnitude_warp": 0.2,
                "time_warp": 0.05,
            },
            "version": 4,
        },
        {
            "transformation": "magnitude_warp",
            "params": [0.25],
            "parameters_benchmark": {
                "jitter": 1.2,
                "scaling": 0.1,
                "magnitude_warp": 0.25,
                "time_warp": 0.05,
            },
            "version": 4,
        },
        {
            "transformation": "magnitude_warp",
            "params": [0.3],
            "parameters_benchmark": {
                "jitter": 1.5,
                "scaling": 0.1,
                "magnitude_warp": 0.3,
                "time_warp": 0.05,
            },
            "version": 4,
        },
    ]
    X_hat_transf_all = {}
    X_benchmark_all = {}

    for transformation in transformations:
        X_orig, X_hat_transf, X_benchmark = generate_datasets(
            dataset,
            freq,
            model,
            z,
            create_dataset_vae,
            X_orig,
            transformation["transformation"],
            transformation["params"],
            transformation["parameters_benchmark"],
            transformation["version"],
        )

        X_hat_transf_all[transformation["params"][0]] = X_hat_transf
        X_benchmark_all[transformation["params"][0]] = X_benchmark

    plot_magnitude_comparisons(
        X_orig,
        X_hat_transf_all,
        X_benchmark_all,
        [t["params"][0] for t in transformations],
        n_series,
    )
