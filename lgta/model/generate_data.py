import numpy as np
import torch
from lgta.transformations import ManipulateData
from lgta.feature_engineering.feature_transformations import detemporalize
from lgta.transformations.apply_transformations_benchmark import (
    apply_transformations_and_standardize,
)
from lgta.utils.helper import reshape_datasets, clip_datasets


def generate_synthetic_data(model, z, create_dataset_vae, transformation, params):
    """
    Generates synthetic data by applying a transformation to the latent space
    representation v' = T(v, eta), then decoding through the CVAE decoder.

    z has shape (n_windows, window_size, latent_dim) for per-timestep latent variables.
    """
    device = next(model.parameters()).device

    dynamic_features_np, _ = create_dataset_vae.input_data

    original_shape = z.shape
    z_2d = z.reshape(z.shape[0], -1)
    manipulate_data = ManipulateData(z_2d, transformation, list(params))
    z_modified = manipulate_data.apply_transf().reshape(original_shape)

    model.eval()
    with torch.no_grad():
        z_tensor = torch.tensor(z_modified, dtype=torch.float32, device=device)
        dyn_tensor = torch.tensor(
            dynamic_features_np[0], dtype=torch.float32, device=device
        )
        preds = model.decoder(z_tensor, dyn_tensor).cpu().numpy()

    preds = detemporalize(preds, create_dataset_vae.window_size)
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
    X_hat = generate_synthetic_data(
        model, z, create_dataset_vae, transformation, params
    )
    transformed_datasets_benchmark = apply_transformations_and_standardize(
        dataset, freq, parameters_benchmark, standardize=False
    )

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
