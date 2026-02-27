import unittest
from lgta.preprocessing import PreprocessDatasets
from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE


class TestSyntheticDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = "synthetic"
        self.freq = "D"

    def test_synthetic_dataset_structure(self):
        groups = PreprocessDatasets(
            dataset=self.dataset, freq=self.freq
        ).apply_preprocess()

        self.assertIn("train", groups)
        self.assertIn("predict", groups)
        self.assertIn("groups_idx", groups["train"])
        self.assertIn("groups_n", groups["train"])
        self.assertIn("groups_names", groups["train"])
        self.assertIn("s", groups["train"])
        self.assertIn("n", groups["train"])
        self.assertEqual(groups["base_series"].shape, (3, 100, 1))

    def test_synthetic_dataset_CVAE(self):
        create_dataset_cvae = CreateTransformedVersionsCVAE(
            dataset_name=self.dataset,
            freq=self.freq,
        )

        model, _, _ = create_dataset_cvae.fit(epochs=2, latent_dim=3)
        (
            preds,
            z,
            z_mean,
            z_log_var,
        ) = create_dataset_cvae.predict(model)

        dec_pred_hat = create_dataset_cvae.generate_transformed_time_series(
            cvae=model,
            z_mean=z_mean,
            z_log_var=z_log_var,
            transformation="jitter",
        )

        self.assertEqual(dec_pred_hat.shape, (100, 60))
