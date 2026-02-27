import unittest
import os
import numpy as np
from lgta.transformations.create_dataset_versions import CreateTransformedVersions
from lgta.transformations.compute_similarities_summary_metrics import (
    ComputeSimilarities,
)
from lgta.visualization.visualize_transformed_datasets import Visualizer


class TestCreateTransformedDatasets(unittest.TestCase):
    def setUp(self):
        self.dataset = "tourism_small"
        self.freq = "Q"
        self.transformed_datasets = CreateTransformedVersions(
            self.dataset, freq=self.freq
        )
        self.transformed_datasets.parameters = {
            "jitter": 0.5,
            "scaling": 0.1,
            "magnitude_warp": 0.05,
            "time_warp": 0.05,
        }
        self.transformed_datasets.create_new_version_single_transf()
        np.random.seed(0)

    def test_create_correct_number_transformed_datasets_single_transf(self):
        self.assertEqual(
            self.transformed_datasets.y_new_all.shape, (4, 6, 10, 36, 56)
        )

    def test_create_correct_number_transformed_datasets_FILES_single_transf(self):
        transformed_datasets = CreateTransformedVersions(self.dataset, freq=self.freq)
        transformed_datasets.create_new_version_single_transf()
        assets_dir = "./assets/data/transformed_datasets"
        self.assertTrue(os.path.isdir(assets_dir))
        file_count = len(
            [name for name in os.listdir(assets_dir) if not name.startswith(".")]
        )
        self.assertGreaterEqual(file_count, 1)

    def test_load_groups_transformed(self):
        transformed_datasets = CreateTransformedVersions(self.dataset, freq=self.freq)
        transformed_datasets.read_groups_transformed("jitter")
        self.assertEqual(
            transformed_datasets.y_loaded_transformed.shape, (6, 10, 36, 56)
        )

    def test_create_transformations_with_tourism_small_dataset(self):
        mean_sim_time_warp_version_1 = ComputeSimilarities(
            dataset=self.transformed_datasets.y,
            transf_dataset=self.transformed_datasets.y_new_all[3, 0, 9],
        ).compute_mean_similarity_elementwise()

        mean_sim_time_warp_version_6 = ComputeSimilarities(
            dataset=self.transformed_datasets.y,
            transf_dataset=self.transformed_datasets.y_new_all[3, 5, 9],
        ).compute_mean_similarity_elementwise()

        self.assertGreater(mean_sim_time_warp_version_6, mean_sim_time_warp_version_1)

    def test_create_transformations_compare_with_files(self):
        vi = Visualizer(self.dataset)
        vi._read_files(method="single_transf_time_warp")

        mean_sim = ComputeSimilarities(
            dataset=self.transformed_datasets.y_new_all[3, 5, 9][:, 10].reshape(-1, 1),
            transf_dataset=vi.y_new[5, 9][:, 10].reshape(-1, 1),
        ).compute_mean_similarity_elementwise()
        self.assertEqual(mean_sim, 0)
