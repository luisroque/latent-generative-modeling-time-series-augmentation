import numpy as np
import pandas as pd
from typing import Literal


DetemporalizeMethod = Literal["mean", "center"]


def temporalize(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Transforming the data to the following shape using a rolling window:
    from (n, s) to (n-window_size+1, window_size, s)

    :param data: input data to transform
    :param window_size: input window to consider on the transformation

    :return X: ndarray of the transformed features
    :return Y: ndarray of the transformed labels
    """

    X = []
    for i in range(len(data) - window_size + 1):
        row = [r for r in data[i : i + window_size]]
        X.append(row)
    return np.array(X)


def _detemporalize_mean(
    data: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Reconstruct the original time-series shape by averaging predictions from
    all overlapping windows that cover each timestep.

    Each timestep t appears in up to `window_size` different windows. Averaging
    all available predictions reduces variance by up to sqrt(window_size).
    """
    num_sequences, seq_len, num_features = data.shape
    num_data_points = num_sequences + window_size - 1

    output = np.zeros((num_data_points, num_features))
    counts = np.zeros((num_data_points, 1))

    for i in range(num_sequences):
        output[i : i + seq_len] += data[i]
        counts[i : i + seq_len] += 1

    output /= counts
    return output


def _detemporalize_center(
    data: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Reconstruct by selecting the most centered window prediction per timestep."""
    num_sequences, seq_len, num_features = data.shape
    num_data_points = num_sequences + window_size - 1
    center_idx = window_size // 2

    output = np.zeros((num_data_points, num_features))

    for t in range(num_data_points):
        # Pick a window index such that timestep t is as close as possible
        # to the center position inside that window.
        window_idx = t - center_idx
        window_idx = min(max(window_idx, 0), num_sequences - 1)
        local_idx = t - window_idx
        local_idx = min(max(local_idx, 0), seq_len - 1)
        output[t] = data[window_idx, local_idx]

    return output


def detemporalize(
    data: np.ndarray,
    window_size: int,
    method: DetemporalizeMethod = "mean",
) -> np.ndarray:
    """Reconstruct original time series from overlapping windows."""
    if method == "mean":
        return _detemporalize_mean(data, window_size)
    if method == "center":
        return _detemporalize_center(data, window_size)
    raise ValueError(f"Unknown detemporalize method '{method}'")


def combine_inputs_to_model(
    X_train: np.ndarray,
    dynamic_features: pd.DataFrame,
    window_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Combining the input features to the model: dynamic features, raw time series data and static features

    :param X_train: raw time series data
    :param dynamic_features: dynamic features already processed
    :param window_size: rolling window

    :return: dynamic features ready to be inputed by the model
    :return: raw time series features ready to be inputed by the model
    :return: static features ready to be inputed by the model

    """

    X_dyn = temporalize(dynamic_features.to_numpy(), window_size)

    return [X_dyn], [X_train]
