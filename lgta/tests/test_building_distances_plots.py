import unittest
import numpy as np
import pathlib

from lgta.transformations.compute_distances import compute_store_distances
from lgta.visualization.visualize_ridge_distance import (
    build_df_ridge,
    plot_distances,
    load_distances,
    store_df_distances,
    load_df_distances,
)
from lgta.feature_engineering.get_data_distance import get_data


def _assets_exist(assets_transformed: pathlib.Path, dataset: str) -> bool:
    dataset_dir = assets_transformed / dataset
    return dataset_dir.exists() and any(dataset_dir.iterdir())


class TestBuildingDistancePlots(unittest.TestCase):
    def setUp(self):
        project_root = pathlib.Path(__file__).resolve().parent.parent.parent
        self.assets_transformed = project_root / "assets" / "data" / "transformed_datasets"
        self.assets_distances = project_root / "assets" / "data" / "distances"
        self.dataset = "tourism"
        self.versions = 2
        self.transformations = ["jitter", "scaling"]
        if not _assets_exist(self.assets_transformed, self.dataset):
            raise unittest.SkipTest(
                f"Assets not found at {self.assets_transformed / self.dataset}; "
                "run data pipeline to generate transformed datasets."
            )
        self.data_orig, self.data_transf = get_data(
            str(self.assets_transformed),
            str(self.assets_transformed),
            self.dataset,
            transformations=self.transformations,
            versions=self.versions,
        )
        self.s = self.data_transf.shape[4]
        self.d_orig, self.d_transf = compute_store_distances(
            self.dataset,
            self.data_orig,
            self.data_transf,
            self.transformations,
            self.versions,
            directory=str(self.assets_distances),
        )
        self.n_d = self.d_transf.shape[2]

    def test_compute_distances_shape(self):
        self.assertTrue(
            self.n_d
            == np.math.factorial(self.s)
            / (np.math.factorial(self.s - 2) * np.math.factorial(2))
        )

    def test_build_df_distances(self):
        df_ridge = build_df_ridge(
            self.d_transf, self.d_orig, self.n_d, self.transformations, self.versions
        )
        expected_rows = self.n_d * len(self.transformations)
        expected_cols = 2 + self.versions
        self.assertEqual(df_ridge.shape, (expected_rows, expected_cols))

    def test_store_load_data(self):
        d_transf_load, d_orig_load = load_distances(
            self.dataset, directory=str(self.assets_distances)
        )
        self.assertEqual(
            d_transf_load.shape,
            (len(self.transformations), self.versions, self.n_d),
        )

    def test_store_distances_df(self):
        df_ridge = build_df_ridge(
            self.d_transf, self.d_orig, self.n_d, self.transformations, self.versions
        )
        store_df_distances(df_ridge, self.dataset, directory=str(self.assets_distances))

    def test_plot_distances(self):
        df_ridge = load_df_distances(
            self.dataset, directory=str(self.assets_distances)
        )
        plot_distances(
            self.dataset,
            df_ridge,
            self.versions,
            x_range=[0, 10],
            show=False,
        )
