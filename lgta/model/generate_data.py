import numpy as np
from lgta.transformations import ManipulateData
from lgta.feature_engineering.feature_transformations import detemporalize
from lgta.transformations.apply_transformations_benchmark import (
    apply_transformations_and_standardize,
)
from lgta.utils.helper import reshape_datasets, clip_datasets


def generate_synthetic_data(model, z, create_dataset_vae, transformation, params):
    """
    Generates synthetic data by applying a transformation to the latent space representation,
    then using the model's decoder to generate predictions.
    """
    all_preds = []

    for dynamic_feat, X_inp in create_dataset_vae.input_data:
        # apply the specified transformation to the latent space representation
        manipulate_data = ManipulateData(
            z, transformation, [param * 100 for param in params]
        )
        z_modified = manipulate_data.apply_transf()

        # generate predictions using the transformed latent representation
        preds = model.decoder.predict([z_modified, dynamic_feat])
        all_preds.append(preds)

    all_preds = np.concatenate(all_preds, axis=0)

    preds = detemporalize(all_preds, create_dataset_vae.window_size)
    X_hat = create_dataset_vae.scaler_target.inverse_transform(preds)

    return X_hat


def generate_datasets(
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
):
    # apply transformations and generate synthetic data
    X_hat = generate_synthetic_data(
        model, z, create_dataset_vae, transformation, params
    )
    transformed_datasets_benchmark = apply_transformations_and_standardize(
        dataset, freq, parameters_benchmark, standardize=False
    )

    # reshape and clip datasets
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
