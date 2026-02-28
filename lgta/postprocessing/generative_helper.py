"""
Generation of new time series by applying transformation chains to the
CVAE latent code and decoding. Delegates to generate_synthetic_data for
the actual generation to maintain a single code path.
"""

from typing import Literal, Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from lgta.model.generate_data import generate_synthetic_data


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
    then applying a chain of transformations to the latent code.

    Kept for backward compatibility but delegates to generate_synthetic_data.
    """
    if sample_from_posterior:
        z_std = np.exp(z_log_var * 0.5)
        z_input = np.random.normal(z_mean, z_std)
    else:
        z_input = z_mean.copy()

    if transformations is not None:
        for transformation, param in zip(transformations, transf_params):
            from lgta.transformations.manipulate_data import ManipulateData
            z_input = ManipulateData(
                x=z_input, transformation=transformation, parameters=[param],
            ).apply_transf()

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

    from lgta.feature_engineering.feature_transformations import detemporalize
    preds = detemporalize(preds, window_size, method=detemporalize_method)
    if clip_to_unit_interval:
        preds = np.clip(preds, 0.0, 1.0)
    return scaler_target.inverse_transform(preds)
