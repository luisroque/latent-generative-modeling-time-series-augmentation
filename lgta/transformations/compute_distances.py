import numpy as np
from dtw import dtw
from sklearn.preprocessing import StandardScaler


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
