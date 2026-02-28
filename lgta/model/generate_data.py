"""
Synthetic data generation through latent space perturbation. Applies
transformations (jitter, scaling, magnitude_warp, time_warp) to the CVAE
latent code z via ManipulateData, then decodes back to data space.
"""

import numpy as np
import torch
from typing import Literal
from lgta.transformations import ManipulateData
from lgta.feature_engineering.feature_transformations import detemporalize
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
    Generate synthetic data by perturbing the latent code z_mean with any
    registered transformation, then decoding through the CVAE decoder.

    Transformations are applied in normalized (unit-scale) space so that
    multiplicative and interpolation-based transforms produce meaningful
    perturbations. Only the perturbation delta is extracted and added to the
    original z_mean, preserving the decoder's expected input range.
    """
    device = next(model.parameters()).device

    z_norm, _, _ = _normalize_latent(z_mean)
    z_transf = ManipulateData(
        x=z_norm, transformation=transformation, parameters=list(params),
    ).apply_transf()
    z_modified = z_mean + (z_transf - z_norm)

    dynamic_features_np = create_dataset_vae.input_data[0][0]

    model.eval()
    with torch.no_grad():
        z_tensor = torch.tensor(z_modified, dtype=torch.float32, device=device)
        dyn_tensor = torch.tensor(
            dynamic_features_np, dtype=torch.float32, device=device
        )
        preds = model.decoder(z_tensor, dyn_tensor).cpu().numpy()

    preds = detemporalize(
        preds,
        create_dataset_vae.window_size,
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
