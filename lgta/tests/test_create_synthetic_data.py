import unittest
from lgta.preprocessing import PreprocessDatasets
from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from lgta.model.models import LatentMode
from lgta.model.generate_data import generate_synthetic_data


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

        model, _, _ = create_dataset_cvae.fit(
            epochs=2, latent_dim=3, kl_anneal_epochs=1,
        )
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

    def test_global_latent_mode_smoke(self):
        """Train and generate with LatentMode.GLOBAL to verify the code path."""
        creator = CreateTransformedVersionsCVAE(
            dataset_name=self.dataset,
            freq=self.freq,
        )
        model, _, _ = creator.fit(
            epochs=2, latent_dim=3, kl_anneal_epochs=1,
            latent_mode=LatentMode.GLOBAL,
        )
        preds, z, z_mean, z_log_var = creator.predict(model)
        self.assertEqual(preds.shape[1], 60)
        self.assertEqual(z_mean.ndim, 2, "GLOBAL z_mean should be 2D (n_windows, d)")

        X_synth = generate_synthetic_data(
            model, z_mean, creator,
            transformation="jitter", params=[0.5],
            latent_mode=LatentMode.GLOBAL,
        )
        self.assertEqual(X_synth.shape[1], 60)
