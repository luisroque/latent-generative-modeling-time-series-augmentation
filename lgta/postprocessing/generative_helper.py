"""
Generation of new time series by applying transformation chains to the
CVAE temporal latent code and decoding. Transformations are applied in
detemporalized latent space (true time axis) then re-temporalized for
decoding, matching the pipeline in generate_synthetic_data.
"""

from typing import Literal, Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def generate_new_time_series(
    cvae,
    z_mean: np.ndarray,
    z_log_var: np.ndarray,
    window_size: int,
    dynamic_features_inp: np.ndarray,
    scaler_target,
    n: int,
    transformations: Optional[list[str]] = None,
    transf_params: Optional[list[float]] = None,
    device=None,
    sample_from_posterior: bool = False,
    detemporalize_method: Literal["mean", "center"] = "mean",
    clip_to_unit_interval: bool = False,
) -> np.ndarray:
    """
    Generate new time series by optionally sampling from the posterior,
    then applying transformations in the detemporalized latent space.

    z_mean has shape (n_windows, window_size, latent_dim). Transformations
    are applied along the true time axis by detemporalizing to
    (n_timesteps, latent_dim), transforming, then re-temporalizing.
    """
    from lgta.feature_engineering.feature_transformations import (
        detemporalize,
        temporalize,
    )
    from lgta.model.generate_data import _normalize_latent

    if sample_from_posterior:
        z_std = np.exp(z_log_var * 0.5)
        z_input = np.random.normal(z_mean, z_std)
    else:
        z_input = z_mean.copy()

    if transformations is not None:
        z_full = detemporalize(z_input, window_size, method="mean")
        z_norm, _, _ = _normalize_latent(z_full)

        for transformation, param in zip(transformations, transf_params):
            from lgta.transformations.manipulate_data import ManipulateData
            z_transf = ManipulateData(
                x=z_norm, transformation=transformation, parameters=[param],
            ).apply_transf()
            z_full = z_full + (z_transf - z_norm)
            z_norm, _, _ = _normalize_latent(z_full)

        z_input = temporalize(z_full, window_size)

    import torch
    if device is None:
        device = torch.device("cpu")

    cvae.eval()
    with torch.no_grad():
        z_tensor = torch.tensor(z_input, dtype=torch.float32, device=device)
        dyn_tensor = torch.tensor(
            dynamic_features_inp, dtype=torch.float32, device=device
        )
        preds = cvae.decoder(z_tensor, dyn_tensor).cpu().numpy()

    preds = detemporalize(preds, window_size, method=detemporalize_method)
    if clip_to_unit_interval:
        preds = np.clip(preds, 0.0, 1.0)
    return scaler_target.inverse_transform(preds)
