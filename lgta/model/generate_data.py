"""
Synthetic data generation through temporal latent space perturbation.
The CVAE produces a per-timestep latent code z of shape (n_win, W, d).
To apply transformations along true time, z is detemporalized into a
full latent time series (n_timesteps, d), transformed, re-temporalized,
and decoded back to data space.
"""

import numpy as np
import torch
from typing import Literal
from lgta.transformations import ManipulateData
from lgta.feature_engineering.feature_transformations import (
    detemporalize,
    temporalize,
)
from lgta.transformations.apply_transformations_benchmark import (
    apply_transformations_and_standardize,
)
from lgta.utils.helper import reshape_datasets, clip_datasets


def _normalize_latent(
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize z to unit scale so non-additive transforms are meaningful."""
    mu = z.mean(axis=0)
    std = z.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (z - mu) / std, mu, std


def generate_synthetic_data(
    model,
    z_mean: np.ndarray,
    create_dataset_vae,
    transformation: str,
    params: list[float],
    detemporalize_method: Literal["mean", "center"] = "mean",
    clip_to_unit_interval: bool = False,
) -> np.ndarray:
    """
    Generate synthetic data by perturbing the temporal latent code z_mean
    along the true time axis, then decoding through the CVAE decoder.

    z_mean has shape (n_windows, window_size, latent_dim). It is first
    detemporalized to (n_timesteps, latent_dim) so that transformations
    operate along real time. The perturbation delta is extracted, added
    to the original, re-temporalized, and decoded.
    """
    device = next(model.parameters()).device
    window_size = create_dataset_vae.window_size

    z_full = detemporalize(z_mean, window_size, method="mean")

    z_norm, _, _ = _normalize_latent(z_full)
    z_transf = ManipulateData(
        x=z_norm, transformation=transformation, parameters=list(params),
    ).apply_transf()
    z_modified = z_full + (z_transf - z_norm)

    z_windows = temporalize(z_modified, window_size)

    dynamic_features_np = create_dataset_vae.input_data[0][0]

    model.eval()
    with torch.no_grad():
        z_tensor = torch.tensor(z_windows, dtype=torch.float32, device=device)
        dyn_tensor = torch.tensor(
            dynamic_features_np, dtype=torch.float32, device=device
        )
        preds = model.decoder(z_tensor, dyn_tensor).cpu().numpy()

    preds = detemporalize(
        preds,
        window_size,
        method=detemporalize_method,
    )
    if clip_to_unit_interval:
        preds = np.clip(preds, 0.0, 1.0)
    return create_dataset_vae.scaler_target.inverse_transform(preds)


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
