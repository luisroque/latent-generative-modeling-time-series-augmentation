import unittest
import numpy as np
import lgta as tsag
from lgta.tests.conftest import skip_unless_dataset


class TestModel(unittest.TestCase):
    def test_import_tourism_default(self):
        skip_unless_dataset("tourism")
        self.data = tsag.preprocessing.PreprocessDatasets(
            "tourism", freq="M"
        ).apply_preprocess()
        self.assertTrue(self.data["train"]["data"].shape == (204, 304))

    def test_import_tourism(self):
        skip_unless_dataset("tourism")
        self.data = tsag.preprocessing.PreprocessDatasets(
            "tourism", freq="M", test_size=50
        ).apply_preprocess()
        self.assertTrue(self.data["train"]["data"].shape == (204, 50))

    def test_import_m5(self):
        skip_unless_dataset("m5")
        self.data = tsag.preprocessing.PreprocessDatasets(
            "m5", test_size=2, freq="W"
        ).apply_preprocess()
        self.assertTrue(self.data["train"]["data"].shape == (261, 2))

    def test_import_m5_daily(self):
        skip_unless_dataset("m5")
        self.data = tsag.preprocessing.PreprocessDatasets(
            "m5", test_size=2, top=5, freq="D", weekly_m5=False
        ).apply_preprocess()
        self.assertTrue(self.data["train"]["data"].shape == (1883, 2))

    def test_import_police(self):
        skip_unless_dataset("police")
        self.data = tsag.preprocessing.PreprocessDatasets(
            "police", top=2, freq="D"
        ).apply_preprocess()
        self.assertTrue(self.data["train"]["data"].shape == (304, 2))

    def test_import_tourism_50perc_data_small_test(self):
        skip_unless_dataset("tourism")
        self.data = tsag.preprocessing.PreprocessDatasets(
            "tourism", test_size=2, freq="M", sample_perc=0.5
        ).apply_preprocess()
        self.assertTrue(
            self.data["train"]["data"].shape
            == (int((228 - self.data["h"]) / 2) + 1, 2)
        )

    def test_import_tourism_50perc_data_x_values(self):
        skip_unless_dataset("tourism")
        self.data = tsag.preprocessing.PreprocessDatasets(
            "tourism", test_size=2, freq="M", sample_perc=0.5
        ).apply_preprocess()
        self.assertListEqual(
            self.data["predict"]["x_values"][-self.data["h"] :],
            list(np.arange(228 - self.data["h"], 228)),
        )

    def test_import_tourism_50perc_data(self):
        skip_unless_dataset("tourism")
        self.data = tsag.preprocessing.PreprocessDatasets(
            "tourism", test_size=50, freq="M", sample_perc=0.5
        ).apply_preprocess()
        self.assertTrue(
            self.data["train"]["data"].shape
            == (int((228 - self.data["h"]) / 2) + 1, 50)
        )

    def test_import_m5_50perc_data(self):
        skip_unless_dataset("m5")
        self.data = tsag.preprocessing.PreprocessDatasets(
            "m5", test_size=2, freq="W", sample_perc=0.5
        ).apply_preprocess()
        self.assertTrue(
            self.data["train"]["data"].shape == (int((275 - self.data["h"]) / 2) + 1, 2)
        )

    def test_import_police_50perc_data(self):
        skip_unless_dataset("police")
        self.data = tsag.preprocessing.PreprocessDatasets(
            "police", top=2, freq="D", sample_perc=0.5
        ).apply_preprocess()
        self.assertTrue(
            self.data["train"]["data"].shape == (int((334 - self.data["h"]) / 2) + 1, 2)
        )
