import unittest
from lgta.visualization.visualize_transformed_datasets import Visualizer
from lgta.transformations.create_dataset_versions import CreateTransformedVersions


class TestVisualizeTransformedDatasets(unittest.TestCase):
    def setUp(self):
        self.dataset = "tourism_small"

    def test_read_files(self):
        transformed_datasets = CreateTransformedVersions(self.dataset, freq="Q")
        transformed_datasets.create_new_version_single_transf()
        vi = Visualizer(self.dataset)
        vi._read_files(method="single_transf_jitter")

        self.assertEqual(vi.y_new.shape, (6, 10, 36, 56))
        self.assertEqual(vi.y.shape, (36, 56))
