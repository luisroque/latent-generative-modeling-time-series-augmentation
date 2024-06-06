import numpy as np
from lgta.transformations.create_dataset_versions import (
    CreateTransformedVersions,
)
from sklearn.preprocessing import StandardScaler


def apply_transformations_and_standardize(dataset, freq, parameters, standardize=False):
    """
    Optionally standardizes the dataset, applies transformations, and then inversely transforms the dataset.

    Parameters:
    - dataset: the dataset identifier.
    - freq: frequency or other relevant parameter for the dataset.
    - parameters: dictionary of transformation parameters.
    - standardize: boolean, whether to standardize the dataset before applying transformations.

    Returns:
    - Transformed and inversely transformed datasets.
    """
    ctv = CreateTransformedVersions(dataset, freq=freq)
    ctv.parameters = parameters

    scaler = StandardScaler()

    if standardize:
        # Standardize the data before the transformations
        ctv.y = scaler.fit_transform(ctv.y)

    transformed_datasets_benchmark = ctv._create_new_version(
        method="single_transf", save=False
    )

    if standardize:
        # Loop through the transformations, levels, and samples to inverse transform each slice
        inverse_transformed_datasets = np.empty_like(transformed_datasets_benchmark)

        for i in range(
            transformed_datasets_benchmark.shape[0]
        ):  # Loop through transformations
            for j in range(
                transformed_datasets_benchmark.shape[1]
            ):  # Loop through levels
                for k in range(
                    transformed_datasets_benchmark.shape[2]
                ):  # Loop through samples
                    # Apply inverse transformation to each slice
                    inverse_transformed_datasets[i, j, k] = scaler.inverse_transform(
                        transformed_datasets_benchmark[i, j, k]
                    )

        transformed_datasets_benchmark = inverse_transformed_datasets

    return transformed_datasets_benchmark
