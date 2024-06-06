import numpy as np


def reshape_datasets(
    X_orig, X_hat, X_benchmark, vae_n_samples, benchmark_transf, version
):
    """
    Reshape datasets according to VAE output and specific benchmark transformation,
    then ensure they are in the format (n_samples, n_features).

    Parameters:
    - X_orig: numpy.ndarray, original dataset.
    - X_hat: numpy.ndarray, dataset from VAE transformation.
    - transformed_benchmark: numpy.ndarray, benchmark transformed dataset.
    - vae_n_samples: int, the number of samples in the VAE dataset.
    - benchmark_transf: str, the type of benchmark transformation to apply ('jitter', 'scaling', 'magnitude_warp', 'time_warp').

    Returns:
    - X_hat_transf: Reshaped VAE transformed dataset.
    - transformed_benchmark_specific: Reshaped specific benchmark dataset.
    """
    transformations = {"jitter": 0, "scaling": 1, "magnitude_warp": 2, "time_warp": 3}

    # Check if benchmark_transf is a valid key
    if benchmark_transf not in transformations:
        raise ValueError(
            f"Invalid benchmark_transf: {benchmark_transf}. Must be one of {list(transformations.keys())}"
        )

    idx_bench_transf = transformations[benchmark_transf]

    transformed_benchmark_specific = X_benchmark[idx_bench_transf, version, 0, :, :]

    X_hat_transf = ensure_n_features_in_second_dim(np.squeeze(X_hat), vae_n_samples)
    transformed_benchmark_specific = ensure_n_features_in_second_dim(
        transformed_benchmark_specific, vae_n_samples
    )
    X_orig = ensure_n_features_in_second_dim(X_orig, vae_n_samples)

    return X_orig, X_hat_transf, transformed_benchmark_specific


def ensure_n_features_in_second_dim(X, n_features):
    """
    Ensures the array has n_samples as its second dimension. If not, the array is transposed.
    It assumes the total size of the array is divisible by n_samples to form a 2D array.

    Parameters:
    - X: numpy.ndarray, the dataset to check and possibly transpose.
    - n_samples: int, the expected number of samples.

    Returns:
    - numpy.ndarray, the array either unchanged or transposed to have n_samples as its second dimension.
    """
    if X.shape[1] != n_features:
        # Check if transposing results in the correct second dimension
        if X.shape[0] == n_features:
            return X.T
        else:
            raise ValueError(
                f"Array shape {X.shape} cannot be adjusted to have {n_features} as the second dimension by simple transposition."
            )
    return X


def clip_datasets(X_hat_transf, X_benchmark):
    """
    Clips values in X_hat_transf and X_benchmark to be above or equal to 0.

    Parameters:
    - X_hat_transf: numpy.ndarray, the L-GTA transformed dataset to clip.
    - X_benchmark: numpy.ndarray, the benchmark transformed dataset to clip.

    Returns:
    - tuple of numpy.ndarrays: The unchanged original dataset, and the clipped transformed datasets.
    """
    X_hat_transf_clipped = np.clip(X_hat_transf, a_min=0, a_max=None)
    X_benchmark_clipped = np.clip(X_benchmark, a_min=0, a_max=None)

    return X_hat_transf_clipped, X_benchmark_clipped
