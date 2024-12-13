import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_series_comparisons(X_orig, X_lgta, X_benchmark, transformation, n_examples=4):
    """
    Plots comparisons between original, L-GTA generated, and benchmark generated series,
    styled similarly to plot_long_tail_comparisons.
    """
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", 3)

    fig, axes = plt.subplots(n_examples, 2, figsize=(20, 3 * n_examples), sharex=True)
    fig.suptitle(
        f"Generating synthetic data using the {transformation} transformation",
        fontsize=16,
        fontweight="bold",
    )

    for i in range(n_examples):
        # L-GTA generated vs original
        axes[i, 0].plot(
            X_orig[:, i], color=colors[0], label="Original", linewidth=2, alpha=0.75
        )
        axes[i, 0].plot(
            X_lgta[:, i],
            color=colors[1],
            label="L-GTA Generated",
            linewidth=2,
            alpha=0.75,
        )
        axes[i, 0].legend()
        axes[i, 0].set_title(f"Series {i + 1}: L-GTA Generated vs Original")
        axes[i, 0].set_ylabel("Magnitude")

        # benchmark generated vs original
        axes[i, 1].plot(
            X_orig[:, i], color=colors[0], label="Original", linewidth=2, alpha=0.75
        )
        axes[i, 1].plot(
            X_benchmark[:, i],
            color=colors[2],
            label="Benchmark Generated",
            linewidth=2,
            alpha=0.75,
        )
        axes[i, 1].legend()
        axes[i, 1].set_title(f"Series {i + 1}: Benchmark Generated vs Original")

    for ax in axes.flat:
        ax.label_outer()  # ensure that only the bottom and left subplots have tick labels
    plt.tight_layout()
    plt.show()


def plot_transformations_comparison(transformed_datasets, X_orig, series):
    """
    Plots comparisons between original data and transformed datasets across different transformations and their intensity levels.

    Parameters:
    - transformed_datasets: numpy.ndarray, transformed datasets with dimensions (4, 6, 10, n_samples, n_features).
    - X_orig: numpy.ndarray, original dataset with dimensions (n_samples, n_features).
    - sample_index: int, index of the sample to display from the transformed datasets.
    - n_features: int, index of the feature to plot from the datasets.
    """
    transformations = ["Jitter", "Scaling", "Magnitude Warp", "Time Warp"]
    levels = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]

    fig, axes = plt.subplots(4, 6, figsize=(24, 16), sharex=True, sharey=True)

    for i, transformation in enumerate(transformations):
        for j, level in enumerate(levels):
            ax = axes[i, j]
            # Select a specific transformation level and sample
            transformed_sample = transformed_datasets[i, j, 0, :, series]
            ax.plot(transformed_sample, label=f"Transformed - {level}")
            ax.plot(
                X_orig[:, series], alpha=0.55, label="Original Data", color="orange"
            )
            ax.legend(loc="upper right")

            ax.set_title(f"{transformation} - {level}")
            ax.set_xlabel("Time")
            if j == 0:
                ax.set_ylabel("Magnitude")

    plt.suptitle(
        f"Comparison of Original and Transformed Data for Series {series}", fontsize=16
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_long_tail_comparisons(
    X_orig, X_transf_1, X_transf_2, label_transf_1, label_transf_2, distances, top_n=3
):
    """
    Plots the top N series in the long tail of the distribution comparing original with two transformed datasets.

    Parameters:
    - X_orig: numpy.ndarray, original dataset.
    - X_transf_1: numpy.ndarray, first transformed dataset.
    - X_transf_2: numpy.ndarray, second transformed dataset.
    - label_transf_1: str, label for the first transformed dataset.
    - label_transf_2: str, label for the second transformed dataset.
    - distances: numpy.ndarray, distances used for sorting and selecting the long tail.
    - top_n: int, number of top series to plot from the long tail.
    """
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    long_tail_indices = np.argsort(distances)[-top_n:]
    colors = sns.color_palette("husl", 3)

    num_plots = len(long_tail_indices)
    fig, axes = plt.subplots(nrows=num_plots, ncols=3, figsize=(20, 12), sharex=True)

    for i, idx in enumerate(long_tail_indices):
        y_min = min(
            np.min(X_orig[idx]), np.min(X_transf_1[idx]), np.min(X_transf_2[idx])
        )
        y_max = max(
            np.max(X_orig[idx]), np.max(X_transf_1[idx]), np.max(X_transf_2[idx])
        )

        # Original series
        axes[i, 0].plot(X_orig[idx], color=colors[0], label="Original", linewidth=2)
        axes[i, 0].set_title(f"Original Series Index: {idx}")
        axes[i, 0].set_ylim(y_min, y_max)

        # First transformed series
        axes[i, 1].plot(
            X_transf_1[idx], color=colors[1], label=label_transf_1, linewidth=2
        )
        axes[i, 1].set_title(f"{label_transf_1} Series Index: {idx}")
        axes[i, 1].set_ylim(y_min, y_max)
        axes[i, 1].tick_params(axis="y", labelleft=False)

        # Second transformed series
        axes[i, 2].plot(
            X_transf_2[idx], color=colors[2], label=label_transf_2, linewidth=2
        )
        axes[i, 2].set_title(f"{label_transf_2} Series Index: {idx}")
        axes[i, 2].set_ylim(y_min, y_max)
        axes[i, 2].tick_params(axis="y", labelleft=False)

        # Set common labels
        for j in range(3):
            axes[i, j].set_xlabel("Time")
            if j == 0:
                axes[i, j].set_ylabel("Magnitude")

    fig.subplots_adjust(right=0.85, top=0.9)
    fig.suptitle(
        "Long Tail Comparison: Original vs Transformed Series",
        fontsize=16,
        fontweight="bold",
    )
    plt.legend()
    plt.show()


def plot_transformations_with_generate_datasets(
    dataset,
    freq,
    generate_datasets,
    X_orig,
    model,
    z,
    create_dataset_vae,
    transformations,
    num_series,
):
    """
    Plots transformations for multiple series, ensuring L-GTA and benchmark columns
    for each transformation share y-limits.
    """
    nrows = num_series
    ncols = 2 * len(transformations)
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(20, 3 * nrows), sharex=True
    )
    fig.suptitle(
        "Time Series Transformations Comparison", fontsize=20, fontweight="bold"
    )

    for idx_t, transformation_info in enumerate(transformations):
        X_orig_plot, X_hat_transf, X_benchmark = generate_datasets(
            dataset,
            freq,
            model,
            z,
            create_dataset_vae,
            X_orig,
            transformation_info["transformation"],
            transformation_info["params"],
            transformation_info["parameters_benchmark"],
            transformation_info["version"],
        )

        for idx_s in range(num_series):
            col_lgta = idx_t * 2
            col_benchmark = idx_t * 2 + 1

            # Find global min and max across both transformations for consistent y-limits
            global_min = min(
                X_orig_plot[:, idx_s].min(),
                X_hat_transf[:, idx_s].min(),
                X_benchmark[:, idx_s].min(),
            )
            global_max = max(
                X_orig_plot[:, idx_s].max(),
                X_hat_transf[:, idx_s].max(),
                X_benchmark[:, idx_s].max(),
            )

            # Plotting L-GTA Transformation
            axs[idx_s, col_lgta].plot(
                X_orig_plot[:, idx_s], label="Original", color="gray", linestyle="--"
            )
            axs[idx_s, col_lgta].plot(
                X_hat_transf[:, idx_s], label="L-GTA", color="darkorange", alpha=0.75
            )
            axs[idx_s, col_lgta].set_ylim(global_min, global_max)
            axs[idx_s, col_lgta].legend()
            axs[idx_s, col_lgta].grid(True, linestyle="--", linewidth=0.5)

            # Plotting Benchmark Transformation
            axs[idx_s, col_benchmark].plot(
                X_orig_plot[:, idx_s], label="Original", color="gray", linestyle="--"
            )
            axs[idx_s, col_benchmark].plot(
                X_benchmark[:, idx_s], label="Benchmark", color="dodgerblue", alpha=0.55
            )
            axs[idx_s, col_benchmark].set_ylim(global_min, global_max)
            axs[idx_s, col_benchmark].legend()
            axs[idx_s, col_benchmark].grid(True, linestyle="--", linewidth=0.5)

            # Setting titles
            if idx_s == 0:
                axs[idx_s, col_lgta].set_title(
                    f'{transformation_info["transformation"].capitalize()} - L-GTA',
                    fontsize=16,
                    fontweight="bold",
                    pad=20,
                )
                axs[idx_s, col_benchmark].set_title(
                    f'{transformation_info["transformation"].capitalize()} - Benchmark',
                    fontsize=16,
                    fontweight="bold",
                    pad=20,
                )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.text(0.5, 0.01, "Time", ha="center", va="bottom", fontsize=14)
    fig.text(0.01, 0.5, "Value", va="center", rotation="vertical", fontsize=14)

    plt.show()


def plot_single_time_series(
    dataset,
    freq,
    generate_datasets,
    X_orig,
    model,
    z,
    create_dataset_vae,
    transformations,
    num_series=4,
):
    """
    Plots a single plot for each time series, showing the original series, L-GTA,
    and Benchmark.
    """
    colors = {
        "original": "black",
        "l_gta": "darkorange",
        "benchmark": "dodgerblue",
    }

    for idx_s in range(num_series):
        for transformation_info in transformations:
            plt.figure(figsize=(12, 6))
            X_orig_plot, X_hat_transf, X_benchmark = generate_datasets(
                dataset,
                freq,
                model,
                z,
                create_dataset_vae,
                X_orig,
                transformation_info["transformation"],
                transformation_info["params"],
                transformation_info["parameters_benchmark"],
                transformation_info["version"],
            )

            plt.plot(
                X_orig_plot[:, idx_s],
                label="Original",
                color=colors["original"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )

            plt.plot(
                X_hat_transf[:, idx_s],
                label="L-GTA",
                color=colors["l_gta"],
                linewidth=1.5,
                alpha=0.8,
            )

            plt.plot(
                X_benchmark[:, idx_s],
                label="Benchmark",
                color=colors["benchmark"],
                linewidth=1.5,
                alpha=0.8,
            )

            plt.title(
                f"Original, L-GTA, and Benchmark using the {transformation_info['transformation']} Transformation",
                fontsize=16,
            )
            plt.xlabel("Time", fontsize=14)
            plt.ylabel("Value", fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle="--", linewidth=0.5)

            plt.tight_layout()
            plt.show()
