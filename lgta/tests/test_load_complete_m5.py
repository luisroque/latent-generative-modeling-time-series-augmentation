import unittest
from lgta.tests.conftest import skip_unless_dataset
from lgta.preprocessing import PreprocessDatasets


class TestLoadM5(unittest.TestCase):
    def test_load_m5_subsampled(self):
        skip_unless_dataset("m5")
        m5 = PreprocessDatasets(
            dataset="m5", freq="D", weekly_m5=False, top=10
        ).apply_preprocess()
        self.assertEqual(m5["train"]["s"], 10)
        self.assertIn("train", m5)
        self.assertIn("predict", m5)
        self.assertIn("data", m5["train"])
