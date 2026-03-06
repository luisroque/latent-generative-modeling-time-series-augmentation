"""
Synthetic data generation through latent space perturbation.

In TEMPORAL mode the CVAE produces z of shape (n_win, W, d). The code
detemporalizes to (n_timesteps, d), transforms along true time,
re-temporalizes, and decodes. In GLOBAL mode z is (n_win, d); the
transform is applied directly and the decoder broadcasts z to (B, W, d)
internally.
"""

import numpy as np
import torch
from typing import Literal, Optional
from lgta.transformations import ManipulateData
from lgta.model.models import LatentMode
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
    latent_mode: LatentMode = LatentMode.TEMPORAL,
    z_log_var: Optional[np.ndarray] = None,
    sample_from_posterior: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate synthetic data by perturbing latent code then decoding.

    In TEMPORAL mode, z_mean is (n_windows, W, latent_dim) and is
    detemporalized before transforming. In GLOBAL mode, z_mean is
    (n_windows, latent_dim) and is transformed directly.

    If sample_from_posterior is True, z is drawn from N(z_mean, exp(0.5*z_log_var))
    before applying the transformation; z_log_var must be provided.
    """
    z_input = z_mean
    if sample_from_posterior:
        if z_log_var is None:
            raise ValueError("z_log_var is required when sample_from_posterior=True")
        gen = rng if rng is not None else np.random.default_rng()
        z_std = np.exp(z_log_var * 0.5)
        z_input = np.array(gen.normal(z_mean, z_std), dtype=np.float32)

    device = next(model.parameters()).device
    window_size = create_dataset_vae.window_size

    if latent_mode == LatentMode.GLOBAL:
        z_norm, _, _ = _normalize_latent(z_input)
        z_transf = ManipulateData(
            x=z_norm, transformation=transformation, parameters=list(params),
        ).apply_transf()
        z_modified = z_input + (z_transf - z_norm)
        z_decode = z_modified
    else:
        z_full = detemporalize(z_input, window_size, method="mean")
        z_norm, _, _ = _normalize_latent(z_full)
        z_transf = ManipulateData(
            x=z_norm, transformation=transformation, parameters=list(params),
        ).apply_transf()
        z_modified = z_full + (z_transf - z_norm)
        z_decode = temporalize(z_modified, window_size)

    dynamic_features_np = create_dataset_vae.input_data[0][0]

    model.eval()
    with torch.no_grad():
        z_tensor = torch.tensor(z_decode, dtype=torch.float32, device=device)
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
