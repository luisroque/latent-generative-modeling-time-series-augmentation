import matplotlib.pyplot as plt
import tensorflow as tf
from lgta.model.create_dataset_versions_vae import (
    CreateTransformedVersionsCVAE,
)
from lgta.feature_engineering.feature_transformations import detemporalize
from lgta.visualization.comparison_analysis import (
    plot_transformations_with_generate_datasets,
    plot_series_comparisons,
)
from lgta.visualization.model_visualization import plot_loss
from lgta.model.generate_data import generate_datasets
from lgta.evaluation.evaluation_comparison import (
    standardize_and_calculate_residuals,
    analyze_transformations,
)
from lgta.postprocessing.postprocessing_comparison import (
    process_transformations,
    create_prediction_comparison_dataset,
    create_distance_metrics_dataset,
    create_reconstruction_error_percentage_dataset,
)
from lgta.e2e.e2e_processing import e2e_transformation


dataset = "tourism"
freq = "M"
top = None

# For M5 dataset
# dataset = "m5"
# freq = "W"
# top = 500

# For police dataset
# dataset = "police"
# freq = "D"
# top = 500


create_dataset_vae = CreateTransformedVersionsCVAE(
    dataset_name=dataset, freq=freq, top=top, dynamic_feat_trig=True
)
model, history, _ = create_dataset_vae.fit()
plot_loss(history)
(dynamic_feat, X_inp, static_feat), _ = create_dataset_vae._feature_engineering(
    create_dataset_vae.n, val_steps=0
)
stacked_dynamic_feat = tf.stack(dynamic_feat, axis=-1)


######################
# Visualize transformations
######################

_, _, z = model.encoder.predict([X_inp, stacked_dynamic_feat])
z_modified = None

preds = model.decoder.predict([z, stacked_dynamic_feat])

preds = detemporalize(preds, create_dataset_vae.window_size)
X_hat = create_dataset_vae.scaler_target.inverse_transform(preds)

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
        "params": [2, 2, 2, 2],
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
        "params": [2, 2, 2, 2],
        "parameters_benchmark": {
            "jitter": 0.5,
            "scaling": 0.1,
            "magnitude_warp": 0.1,
            "time_warp": 0.05,
        },
        "version": 4,
    },
]

plot_transformations_with_generate_datasets(
    dataset,
    freq,
    generate_datasets,
    X_orig,
    model,
    z,
    dynamic_feat,
    static_feat,
    create_dataset_vae,
    transformations_plot,
    4,
)


######################
# e2e comparison
######################

# Parameters Tourism
transformations = [
    {
        "transformation": "jitter",
        "params": [1.5],
        "parameters_benchmark": {
            "jitter": 0.375,
            "scaling": 0.1,
            "magnitude_warp": 0.1,
            "time_warp": 0.05,
        },
        "version": 5,
    },
    {
        "transformation": "scaling",
        "params": [1.3],
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
        "params": [1.45],
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
        dynamic_feat,
        static_feat,
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
params = [1.5]

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
    dynamic_feat,
    static_feat,
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
plot_series_comparisons(X_orig, X_lgta, X_benchmark)
analyze_transformations(residuals_lgta, residuals_benchmark)
