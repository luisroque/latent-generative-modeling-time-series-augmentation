"""
Generation of new time series by sampling from the CVAE latent space and
optionally applying sequential transformation chains to the latent samples.

Follows the theory: v'_i = T_n(...T_2(T_1(v_i, eta_1), eta_2)..., eta_n)
"""

from typing import Optional
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from lgta.feature_engineering.feature_transformations import detemporalize
from lgta.transformations.manipulate_data import ManipulateData


def generate_new_time_series(
    cvae: nn.Module,
    z_mean: np.ndarray,
    z_log_var: np.ndarray,
    window_size: int,
    dynamic_features_inp: np.ndarray,
    scaler_target: MinMaxScaler,
    n_features: int,
    n: int,
    transformations: Optional[list[str]] = None,
    transf_params: Optional[list[float]] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Generate new time series by sampling per-timestep latent variables from
    the CVAE and optionally applying a chain of transformations.

    Args:
        cvae: A trained CVAE model.
        z_mean: Mean of latent distributions, shape (n_windows, window_size, latent_dim).
        z_log_var: Log-variance of latent distributions, same shape.
        window_size: Size of the rolling window.
        dynamic_features_inp: Dynamic features, shape (n_windows, window_size, n_dyn_features).
        scaler_target: Fitted scaler for inverse-transforming predictions.
        n_features: Number of output features.
        n: Total number of time points.
        transformations: List of transformation names to chain on latent samples.
        transf_params: Corresponding parameters for each transformation.
        device: Torch device for inference.

    Returns:
        Generated time series of shape (n, n_features).
    """
    if device is None:
        device = torch.device("cpu")

    latent_dim = z_mean.shape[-1]
    z_std = np.exp(z_log_var * 0.5)

    dec_pred = []

    cvae.eval()
    with torch.no_grad():
        for id_seq in range(n - window_size + 1):
            v = np.random.normal(z_mean[id_seq], z_std[id_seq])

            if transformations is not None:
                for transformation, param in zip(transformations, transf_params):
                    v = ManipulateData(
                        x=v, transformation=transformation, parameters=[param]
                    ).apply_transf()

            d_feat = dynamic_features_inp[id_seq : id_seq + 1, :, :]
            v_tensor = torch.tensor(
                v.reshape(1, window_size, latent_dim),
                dtype=torch.float32,
                device=device,
            )
            d_tensor = torch.tensor(d_feat, dtype=torch.float32, device=device)

            pred = cvae.decoder(v_tensor, d_tensor)
            dec_pred.append(pred.cpu().numpy())

    dec_pred_hat = detemporalize(np.squeeze(np.array(dec_pred)), window_size)
    dec_pred_hat = scaler_target.inverse_transform(dec_pred_hat)

    return dec_pred_hat
