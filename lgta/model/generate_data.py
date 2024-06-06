from lgta.transformations import ManipulateData
from lgta.feature_engineering.feature_transformations import detemporalize
from lgta.transformations.apply_transformations_benchmark import (
    apply_transformations_and_standardize,
)
from lgta.utils.helper import reshape_datasets, clip_datasets


def generate_synthetic_data(
    model, z, dynamic_feat, static_feat, create_dataset_vae, transformation, params
):
    """
    Generates synthetic data by applying a transformation to the latent space representation,
    then using the model's decoder to generate predictions. Optionally plots comparisons between
    original and transformed series.

    Parameters:
    - model: The trained model with an encoder and decoder.
    - z: Latent space representation to transform.
    - dynamic_feat: Dynamic features required by the model for prediction.
    - static_feat: Static features required by the model for prediction.
    - create_dataset_vae: An object containing utilities for scaling and inverse scaling of data.
    - transformation: String specifying the transformation to apply ('jitter', 'scaling', 'magnitude_warp', 'time_warp').
    - params: Parameters for the transformation.
    - plot_series: Boolean, if True, plots comparisons between original and synthetic series.

    Returns:
    - X_hat: The synthetic dataset after transformation and processing.
    """
    manipulate_data = ManipulateData(
        z, transformation, [param * 100 for param in params]
    )

    # Apply the specified transformation to the latent space representation
    z_modified = manipulate_data.apply_transf()

    # Generate predictions using the transformed latent representation
    preds = model.decoder.predict([z_modified] + dynamic_feat + static_feat)
    preds = detemporalize(preds, create_dataset_vae.window_size)

    # Inverse transform the predictions to get the synthetic dataset
    X_hat = create_dataset_vae.scaler_target.inverse_transform(preds)

    return X_hat


def generate_datasets(
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
):
    # Apply transformations and generate synthetic data
    X_hat = generate_synthetic_data(
        model, z, dynamic_feat, static_feat, create_dataset_vae, transformation, params
    )
    transformed_datasets_benchmark = apply_transformations_and_standardize(
        dataset, freq, parameters_benchmark, standardize=False
    )

    # Reshape and clip datasets
    X_orig, X_hat_transf, X_benchmark = reshape_datasets(
        X_orig,
        X_hat,
        transformed_datasets_benchmark,
        create_dataset_vae.n_features,
        transformation,
        version=version,
    )
    X_hat_transf, X_benchmark = clip_datasets(X_hat_transf, X_benchmark)

    return X_orig, X_hat_transf, X_benchmark
