import unittest
import numpy as np
import lgta as tsag


class TestImportTourismSmall(unittest.TestCase):
    def test_import_tourism_small_default(self):
        data = tsag.preprocessing.PreprocessDatasets(
            "tourism_small", freq="Q"
        ).apply_preprocess()
        self.assertEqual(data["train"]["data"].shape, (32, 56))

    def test_import_tourism_small_test_size(self):
        data = tsag.preprocessing.PreprocessDatasets(
            "tourism_small", freq="Q", test_size=10
        ).apply_preprocess()
        self.assertEqual(data["train"]["data"].shape, (32, 10))

    def test_import_tourism_small_has_required_keys(self):
        data = tsag.preprocessing.PreprocessDatasets(
            "tourism_small", freq="Q"
        ).apply_preprocess()
        self.assertIn("train", data)
        self.assertIn("predict", data)
        self.assertIn("data", data["train"])
        self.assertIn("data_matrix", data["predict"])
        self.assertIn("groups_idx", data["train"])
        self.assertIn("groups_names", data["train"])
        self.assertIn("seasonality", data)
        self.assertIn("h", data)
        self.assertIn("dates", data)

    def test_import_tourism_small_groups(self):
        data = tsag.preprocessing.PreprocessDatasets(
            "tourism_small", freq="Q"
        ).apply_preprocess()
        expected_groups = {"Purpose", "State", "CityType"}
        self.assertEqual(set(data["train"]["groups_names"].keys()), expected_groups)
        self.assertEqual(len(data["train"]["groups_names"]["Purpose"]), 4)
        self.assertEqual(len(data["train"]["groups_names"]["State"]), 7)
        self.assertEqual(len(data["train"]["groups_names"]["CityType"]), 2)

    def test_import_tourism_small_predict_shape(self):
        data = tsag.preprocessing.PreprocessDatasets(
            "tourism_small", freq="Q"
        ).apply_preprocess()
        self.assertEqual(data["predict"]["data_matrix"].shape, (36, 56))

    def test_import_tourism_small_seasonality_and_horizon(self):
        data = tsag.preprocessing.PreprocessDatasets(
            "tourism_small", freq="Q"
        ).apply_preprocess()
        self.assertEqual(data["seasonality"], 4)
        self.assertEqual(data["h"], 4)

    def test_import_unknown_dataset_raises(self):
        with self.assertRaises(ValueError):
            tsag.preprocessing.PreprocessDatasets(
                "nonexistent", freq="Q"
            ).apply_preprocess()
