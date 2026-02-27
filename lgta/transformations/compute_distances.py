import numpy as np
from pathlib import Path
from dtw import dtw
from sklearn.preprocessing import StandardScaler

DEFAULT_DISTANCES_DIR = "assets/data/distances"


def _pairwise_dtw_series(X: np.ndarray) -> np.ndarray:
    """Compute pairwise DTW distances between all columns of X. X is (n_timesteps, n_series)."""
    n = X.shape[1]
    d = []
    for i in range(n):
        for j in range(i + 1, n):
            d.append(dtw(X[:, i], X[:, j]).distance)
    return np.array(d)


def compute_store_distances(
    dataset_name: str,
    data_orig: np.ndarray,
    data_transf: np.ndarray,
    transformations: list,
    versions: int,
    directory: str = DEFAULT_DISTANCES_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise DTW distances for original and each transformed version,
    then store them as .npy files. Returns (d_orig, d_transf).
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    n_series = data_transf.shape[4]
    n_d = n_series * (n_series - 1) // 2
    n_transf = len(transformations)

    if data_orig.ndim > 2:
        data_orig = data_orig.reshape(data_orig.shape[0], -1)
    if data_orig.shape[1] != n_series:
        data_orig = data_orig.T
    d_orig = _pairwise_dtw_series(data_orig)

    d_transf = np.zeros((n_transf, versions, n_d))
    for t in range(n_transf):
        for v in range(versions):
            x = data_transf[t, v]
            if x.ndim > 2:
                x = x[0].reshape(-1, x.shape[-1])
            else:
                x = x.reshape(-1, x.shape[-1])
            d_transf[t, v, :] = _pairwise_dtw_series(x)

    np.save(dir_path / f"{dataset_name}_distances_original.npy", d_orig)
    np.save(dir_path / f"{dataset_name}_distances_transformed.npy", d_transf)
    return d_orig, d_transf


def compute_pairwise_distance(X1, X2, fn):
    """
    Computes the pairwise Wasserstein distances between two sets of samples.

    Parameters:
    - X1: numpy.ndarray, first set of samples.
    - X2: numpy.ndarray, second set of samples.

    Returns:
    - distances: numpy.ndarray, Wasserstein distances between pairs.
    """
    # Ensure input is numpy array and in the right format (n_samples, n_features)
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)

    # Initialize an array to store distances
    distances = np.zeros(min(X1.shape[1], X2.shape[1]))

    # Compute the Wasserstein distance for each pair of samples
    for i in range(len(distances)):
        if fn == dtw:
            distances[i] = fn(X1[:, i], X2[:, i]).distance
        else:
            distances[i] = fn(X1[:, i], X2[:, i])

    return distances


def compute_distances(X_hat, X_orig, transformed_benchmark, fn):
    """
    Reshapes the datasets if necessary and computes pairwise Wasserstein distances.

    Parameters:
    - X_hat: numpy.ndarray, transformed dataset to compare with the original.
    - X_orig: numpy.ndarray, original dataset.
    - transformed_benchmark: numpy.ndarray, benchmark dataset to compare with the original.
    - n_samples: int, the number of samples expected in each dataset.

    Returns:
    - Tuple of numpy.ndarrays, (distances for X_hat vs. X_orig, distances for benchmark vs. X_orig).
    """
    scaler = StandardScaler()

    # Standardize both datasets
    X_orig_norm = scaler.fit_transform(X_orig)
    X_hat_norm = scaler.transform(X_hat)
    X_benchmark_norm = scaler.transform(transformed_benchmark)
    distances_xhat_vs_xorig = compute_pairwise_distance(X_hat_norm, X_orig_norm, fn)
    distances_benchmark_vs_xorig = compute_pairwise_distance(
        X_benchmark_norm, X_orig_norm, fn
    )

    return distances_xhat_vs_xorig, distances_benchmark_vs_xorig
