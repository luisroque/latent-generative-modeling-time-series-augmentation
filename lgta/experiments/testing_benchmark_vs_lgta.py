import matplotlib.pyplot as plt
import numpy as np
from lgta.model.create_dataset_versions_vae import (
    CreateTransformedVersionsCVAE,
)
from lgta.feature_engineering.feature_transformations import detemporalize
from lgta.visualization.comparison_analysis import (
    plot_transformations_with_generate_datasets,
    plot_series_comparisons,
    plot_single_time_series,
)
from lgta.model.generate_data import generate_datasets
from lgta.evaluation.evaluation_comparison import (
    standardize_and_calculate_residuals,
    analyze_transformations,
    plot_residuals_gradient,
)
from lgta.postprocessing.postprocessing_comparison import (
    process_transformations,
    create_prediction_comparison_dataset,
    create_distance_metrics_dataset,
    create_reconstruction_error_percentage_dataset,
)
from lgta.e2e.e2e_processing import e2e_transformation, compare_diff_magnitudes


dataset = "tourism_small"
freq = "Q"


create_dataset_vae = CreateTransformedVersionsCVAE(
    dataset_name=dataset, freq=freq
)
model, _, _ = create_dataset_vae.fit()
X_hat, z, _, _ = create_dataset_vae.predict(model)

######################
# Visualize transformations
######################

X_orig = create_dataset_vae.X_train_raw
X_hat_orig = X_hat

plt.plot(X_hat_orig[:, 0], label="reconstructed")
plt.plot(X_orig[:, 0], label="original")
plt.legend()


######################
# Residuals
######################

transformations_plot = [
    {
        "transformation": "jitter",
        "params": [0.5, 0.5, 0.5, 0.5],
        "parameters_benchmark": {
            "jitter": 0.5,
            "scaling": 0.1,
            "magnitude_warp": 0.1,
            "time_warp": 0.05,
        },
        "version": 5,
    },
    {
        "transformation": "magnitude_warp",
        "params": [0.1, 0.1, 0.1, 0.1],
        "parameters_benchmark": {
            "jitter": 0.5,
            "scaling": 0.1,
            "magnitude_warp": 0.1,
            "time_warp": 0.05,
        },
        "version": 4,
    },
]

# plot_transformations_with_generate_datasets(
#     dataset=dataset,
#     freq=freq,
#     generate_datasets=generate_datasets,
#     X_orig=X_orig,
#     model=model,
#     z=z,
#     create_dataset_vae=create_dataset_vae,
#     transformations=transformations_plot,
#     num_series=4,
# )

# plot_single_time_series(
#     dataset=dataset,
#     freq=freq,
#     generate_datasets=generate_datasets,
#     X_orig=X_orig,
#     model=model,
#     z=z,
#     create_dataset_vae=create_dataset_vae,
#     transformations=transformations_plot,
#     num_series=4,
# )

######################
# magnitude comparison
######################

compare_diff_magnitudes(
    dataset,
    freq,
    model,
    z,
    create_dataset_vae,
    X_orig,
)


######################
# e2e comparison
######################

# Parameters Tourism
transformations = [
    {
        "transformation": "jitter",
        "params": [0.5],
        "parameters_benchmark": {
            "jitter": 0.5,
            "scaling": 0.1,
            "magnitude_warp": 0.1,
            "time_warp": 0.05,
        },
        "version": 5,
    },
    {
        "transformation": "scaling",
        "params": [0.25],
        "parameters_benchmark": {
            "jitter": 0.375,
            "scaling": 0.1,
            "magnitude_warp": 0.1,
            "time_warp": 0.05,
        },
        "version": 4,
    },
    {
        "transformation": "magnitude_warp",
        "params": [0.1],
        "parameters_benchmark": {
            "jitter": 0.375,
            "scaling": 0.1,
            "magnitude_warp": 0.1,
            "time_warp": 0.05,
        },
        "version": 4,
    },
]

results = {}
for transformation in transformations:
    results[transformation["transformation"]] = e2e_transformation(
        dataset,
        freq,
        model,
        z,
        create_dataset_vae,
        transformation["transformation"],
        transformation["params"],
        transformation["parameters_benchmark"],
        transformation["version"],
        X_orig,
        X_hat,
    )


######################
# Comparison of metrics
######################

res_processed = process_transformations(results)

create_distance_metrics_dataset(res_processed)

create_reconstruction_error_percentage_dataset(res_processed)

create_prediction_comparison_dataset(res_processed)


######################
# Residuals Analysis
######################


transformation = "jitter"
params = [0.5]

parameters_benchmark = {
    "jitter": 0.5,
    "scaling": 0.1,
    "magnitude_warp": 0.1,
    "time_warp": 0.05,
}

version = 5

X_orig, X_lgta, X_benchmark = generate_datasets(
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
residuals_lgta, residuals_benchmark = standardize_and_calculate_residuals(
    X_orig, X_lgta, X_benchmark
)
# plot_series_comparisons(X_orig, X_lgta, X_benchmark, "jittering")
analyze_transformations(residuals_lgta, residuals_benchmark)

######################
# Residuals for diff magnitudes
######################

transformation = "jitter"
params = [0.5, 0.6, 0.7, 0.8, 0.9]

parameters_benchmark = [
    {
        "jitter": 0.5,
        "scaling": 0.1,
        "magnitude_warp": 0.1,
        "time_warp": 0.05,
    },
    {
        "jitter": 0.6,
        "scaling": 0.1,
        "magnitude_warp": 0.1,
        "time_warp": 0.05,
    },
    {
        "jitter": 0.7,
        "scaling": 0.1,
        "magnitude_warp": 0.1,
        "time_warp": 0.05,
    },
    {
        "jitter": 0.8,
        "scaling": 0.1,
        "magnitude_warp": 0.1,
        "time_warp": 0.05,
    },
    {
        "jitter": 0.9,
        "scaling": 0.1,
        "magnitude_warp": 0.1,
        "time_warp": 0.05,
    },
]

version = 5

residuals_lgta_all = []
residuals_benchmark_all = []

for i, param in enumerate(params):
    X_orig, X_lgta, X_benchmark = generate_datasets(
        dataset,
        freq,
        model,
        z,
        create_dataset_vae,
        X_orig,
        transformation,
        [param],
        parameters_benchmark[i],
        version,
    )
    residuals_lgta, residuals_benchmark = standardize_and_calculate_residuals(
        X_orig, X_lgta, X_benchmark
    )
    residuals_lgta_all.append(residuals_lgta)
    residuals_benchmark_all.append(residuals_benchmark)

plot_residuals_gradient(residuals_lgta_all, residuals_benchmark_all, params)
