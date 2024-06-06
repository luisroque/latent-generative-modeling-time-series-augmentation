import numpy as np
import pandas as pd
from scipy.stats import iqr, skew, kurtosis


def process_transformations(input_dict):
    # Initialize an empty dictionary to hold processed results
    processed_results = {}

    # Iterate over each transformation type in the input dictionary
    for transformation, values in input_dict.items():
        # Unpack the tuple structures for Wasserstein distances and KL divergences
        wasserstein, reconstruction_and_kl_divergences, dataframe = values
        wasserstein_L_GTA, wasserstein_benchmark = wasserstein
        (
            reconstruction_error_orig,
            reconstruction_kl_l_gta,
            reconstruction_kl_benchmark,
        ) = reconstruction_and_kl_divergences

        # Calculate statistics for Wasserstein distances
        wasserstein_stats_l_gta = {
            "median": np.round(np.median(wasserstein_L_GTA), 3),
            "iqr": np.round(iqr(wasserstein_L_GTA), 3),
        }

        wasserstein_stats_benchmark = {
            "median": np.round(np.median(wasserstein_benchmark), 3),
            "iqr": np.round(iqr(wasserstein_benchmark), 3),
        }

        # Calculate statistics for KL divergences
        kl_stats_l_gta = {
            "median": np.round(reconstruction_kl_l_gta[1], 3),
            "iqr": np.round(reconstruction_kl_l_gta[2], 3),
        }

        kl_stats_benchmark = {
            "median": np.round(reconstruction_kl_benchmark[1], 3),
            "iqr": np.round(reconstruction_kl_benchmark[2], 3),
        }

        reconstruction_error = {
            "reconstruction_error": np.round(reconstruction_error_orig, 3),
            "reconstruction_error_lgta": np.round(reconstruction_kl_l_gta[0], 3),
            "reconstruction_error_benchmark": np.round(
                reconstruction_kl_benchmark[0], 3
            ),
        }

        # Extract MSE values from the DataFrame
        mse_values = {
            "Original": np.round(dataframe.loc["MSE", "Real"], 3),
            "L-GTA": np.round(dataframe.loc["MSE", "L-GTA"], 3),
            "Benchmark": np.round(dataframe.loc["MSE", "Benchmark"], 3),
        }

        # Compile all calculated values into the processed results dictionary
        processed_results[transformation] = {
            "wasserstein_stats_l_gta": wasserstein_stats_l_gta,
            "wasserstein_stats_benchmark": wasserstein_stats_benchmark,
            "kl_stats_l_gta": kl_stats_l_gta,
            "kl_stats_benchmark": kl_stats_benchmark,
            "reconstruction_error": reconstruction_error,
            "mse_values": mse_values,
        }

    return processed_results


def create_distance_metrics_dataset(data_dict):
    # Initialize a list to store rows
    distance_rows = []

    # Define the specific metrics to include (median and IQR)
    metrics_order = ["median", "iqr"]

    for transformation, stats in data_dict.items():
        for key in ["wasserstein_stats_l_gta", "wasserstein_stats_benchmark"]:
            model_prefix = "L-GTA" if "l_gta" in key else "Benchmark"
            stat_type = "Wasserstein" if "wasserstein" in key else "KL Divergence"
            for metric in metrics_order:
                stat_value = stats[key].get(metric)
                if stat_value is not None:
                    column_name = f"{model_prefix} - {transformation}"
                    metric_name = f"{stat_type} {metric.capitalize()}"
                    distance_rows.append(
                        {
                            "Metric": metric_name,
                            "Method-Transformation": column_name,
                            "Value": np.round(stat_value, 3),
                        }
                    )

    # Convert to DataFrame and pivot
    distance_df = pd.DataFrame(distance_rows)
    distance_metrics_df = distance_df.pivot_table(
        index="Metric", columns="Method-Transformation", values="Value"
    )
    stat_types = ["Wasserstein"]
    transformations_order = sorted(list(data_dict.keys()))
    metrics_order = [
        f"{stat_type} {metric.capitalize()}"
        for stat_type in stat_types
        for metric in metrics_order
    ]
    columns_order = [
        f"{model_prefix} - {transformation}"
        for transformation in transformations_order
        for model_prefix in ["L-GTA", "Benchmark"]
    ]

    # Reorder the DataFrame according to the desired order
    distance_metrics_df = distance_metrics_df.reindex(
        index=metrics_order, columns=columns_order
    )

    return distance_metrics_df


def create_reconstruction_error_percentage_dataset(data_dict):
    # Initialize a list to store rows
    reconstruction_rows = []

    for transformation, stats in data_dict.items():
        # Extract the original reconstruction error for reference
        original_error = stats["reconstruction_error"]["reconstruction_error"]

        # Calculate the percentage of the original for L-GTA
        lgta_error = stats["reconstruction_error"]["reconstruction_error_lgta"]
        lgta_percentage_of_original = np.round((lgta_error / original_error) * 100, 1)

        # Calculate the percentage of the original for Benchmark
        benchmark_error = stats["reconstruction_error"][
            "reconstruction_error_benchmark"
        ]
        benchmark_percentage_of_original = np.round(
            (benchmark_error / original_error) * 100, 1
        )

        # Append the calculated percentages
        reconstruction_rows.append(
            {
                "Model": "L-GTA",
                "Transformation": transformation,
                "Reconstruction Error % of Original": lgta_percentage_of_original,
            }
        )
        reconstruction_rows.append(
            {
                "Model": "Benchmark",
                "Transformation": transformation,
                "Reconstruction Error % of Original": benchmark_percentage_of_original,
            }
        )

    # Convert to DataFrame
    reconstruction_df = pd.DataFrame(reconstruction_rows)

    # Pivot the DataFrame to have models as rows and transformations as columns, showing the percentage of original reconstruction error
    reconstruction_error_percentage_df = reconstruction_df.pivot(
        index="Model",
        columns="Transformation",
        values="Reconstruction Error % of Original",
    )
    order = ["L-GTA", "Benchmark"]
    reconstruction_error_percentage_df = reconstruction_error_percentage_df.reindex(
        order
    )

    return reconstruction_error_percentage_df


def create_prediction_comparison_dataset(data_dict):
    # Initialize a list to store rows
    mse_rows = []

    for transformation, stats in data_dict.items():
        # Add MSE stats for Real, L-GTA, and Benchmark
        mse_values = stats["mse_values"]
        for model, mse_value in mse_values.items():
            mse_rows.append(
                {"Model": model, "Transformation": transformation, "MSE": mse_value}
            )

    # Convert to DataFrame and pivot
    mse_df = pd.DataFrame(mse_rows)
    prediction_comparison_df = mse_df.pivot(
        index="Model", columns="Transformation", values="MSE"
    )
    order = ["Original", "L-GTA", "Benchmark"]
    prediction_comparison_df = prediction_comparison_df.reindex(order)

    return prediction_comparison_df
