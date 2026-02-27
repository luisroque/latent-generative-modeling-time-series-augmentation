import unittest
from lgta.tests.conftest import skip_unless_dataset
from lgta.model.create_dataset_versions_vae import (
    CreateTransformedVersionsCVAE,
)


class TestModel(unittest.TestCase):
    def test_test_size_tourism_small(self):
        skip_unless_dataset("tourism")
        self.create_dataset_vae = CreateTransformedVersionsCVAE(
            dataset_name="tourism", freq="M", test_size=2
        )

        self.assertTrue(
            self.create_dataset_vae.dataset["train"]["data"].shape == (204, 2)
        )

    def test_test_size_tourism(self):
        skip_unless_dataset("tourism")
        self.create_dataset_vae = CreateTransformedVersionsCVAE(
            dataset_name="tourism", freq="M", test_size=2
        )

        self.assertTrue(
            self.create_dataset_vae.dataset["train"]["data"].shape == (204, 2)
        )

    def test_test_size_m5(self):
        skip_unless_dataset("m5")
        self.create_dataset_vae = CreateTransformedVersionsCVAE(
            dataset_name="m5", freq="W", test_size=2, weekly_m5=True
        )

        self.assertTrue(
            self.create_dataset_vae.dataset["train"]["data"].shape == (261, 2)
        )

    def test_test_size_m5_daily(self):
        skip_unless_dataset("m5")
        self.create_dataset_vae = CreateTransformedVersionsCVAE(
            dataset_name="m5", freq="D", test_size=2, weekly_m5=False
        )

        self.assertTrue(
            self.create_dataset_vae.dataset["train"]["data"].shape == (1869, 2)
        )

    def test_test_size_police(self):
        skip_unless_dataset("police")
        self.create_dataset_vae = CreateTransformedVersionsCVAE(
            dataset_name="police", freq="D", test_size=2
        )

        self.assertTrue(
            self.create_dataset_vae.dataset["train"]["data"].shape == (304, 2)
        )
