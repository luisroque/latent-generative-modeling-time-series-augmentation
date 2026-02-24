"""
Generation of new time series by sampling from the CVAE latent space and
optionally applying sequential transformation chains to the latent samples.

Follows the theory: v'_i = T_n(...T_2(T_1(v_i, eta_1), eta_2)..., eta_n)
"""

from typing import Optional, Union
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from lgta.feature_engineering.feature_transformations import detemporalize
from lgta.transformations.manipulate_data import ManipulateData


def generate_new_time_series(
    cvae: keras.Model,
    z_mean: np.ndarray,
    z_log_var: np.ndarray,
    window_size: int,
    dynamic_features_inp: np.ndarray,
    scaler_target: MinMaxScaler,
    n_features: int,
    n: int,
    transformations: Optional[list[str]] = None,
    transf_params: Optional[list[float]] = None,
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

    Returns:
        Generated time series of shape (n, n_features).
    """
    latent_dim = z_mean.shape[-1]
    z_std = np.exp(z_log_var * 0.5)

    dec_pred = []

    for id_seq in range(n - window_size + 1):
        # v_t ~ N(mu_t, Sigma_t) — per-timestep sampling
        v = np.random.normal(z_mean[id_seq], z_std[id_seq])

        # Apply transformation chain: v' = T_n(...T_1(v, eta_1)..., eta_n)
        if transformations is not None:
            for transformation, param in zip(transformations, transf_params):
                v = ManipulateData(
                    x=v, transformation=transformation, parameters=[param]
                ).apply_transf()

        d_feat = dynamic_features_inp[id_seq : id_seq + 1, :, :]
        dec_pred.append(
            cvae.decoder.predict(
                [v.reshape(1, window_size, latent_dim), d_feat], verbose=0
            )
        )

    dec_pred_hat = detemporalize(np.squeeze(np.array(dec_pred)), window_size)
    dec_pred_hat = scaler_target.inverse_transform(dec_pred_hat)

    return dec_pred_hat
