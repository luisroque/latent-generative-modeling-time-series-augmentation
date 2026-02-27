import unittest
from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE


class TestVAETestSize(unittest.TestCase):
    def test_test_size_tourism_small_subset(self):
        vae = CreateTransformedVersionsCVAE(
            dataset_name="tourism_small", freq="Q", test_size=5
        )
        self.assertEqual(vae.dataset["train"]["data"].shape, (32, 5))

    def test_test_size_tourism_small_full(self):
        vae = CreateTransformedVersionsCVAE(
            dataset_name="tourism_small", freq="Q"
        )
        self.assertEqual(vae.dataset["train"]["data"].shape, (32, 56))
