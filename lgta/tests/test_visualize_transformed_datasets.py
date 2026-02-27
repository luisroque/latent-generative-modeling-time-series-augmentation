import unittest
from lgta.visualization.visualize_transformed_datasets import Visualizer
from lgta.transformations.create_dataset_versions import CreateTransformedVersions


class TestCreateTransformedDatasets(unittest.TestCase):

    def setUp(self):
        self.dataset = "tourism"

    def test_read_files(self):
        transformed_datasets = CreateTransformedVersions(self.dataset, freq="M")
        transformed_datasets.create_new_version_single_transf()
        vi = Visualizer(self.dataset)
        vi._read_files(method="single_transf_jitter")

        self.assertTrue(vi.y_new.shape == (6, 10, 228, 304))
        self.assertTrue(vi.y.shape == (228, 304))
