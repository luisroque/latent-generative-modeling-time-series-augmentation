import unittest
from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from lgta.transformations.compute_similarities_summary_metrics import (
    compute_similarity_transformed_vs_original,
)


class TestEndToEndVAETourismSmall(unittest.TestCase):
    def setUp(self) -> None:
        self.create_dataset_vae = CreateTransformedVersionsCVAE(
            dataset_name="tourism_small", freq="Q", test_size=5
        )
        self.model, _, _ = self.create_dataset_vae.fit(
            epochs=1, load_weights=False, kl_anneal_epochs=1,
        )
        (
            self.preds,
            self.z,
            self.z_mean,
            self.z_log_var,
        ) = self.create_dataset_vae.predict(self.model)

    def test_compute_similarity(self):
        dec_pred_hat = self.create_dataset_vae.generate_transformed_time_series(
            cvae=self.model,
            z_mean=self.z_mean,
            z_log_var=self.z_log_var,
            transf_param=0.5,
        )
        similarity = compute_similarity_transformed_vs_original(
            dec_pred_hat, self.create_dataset_vae.X_train_raw
        )[0]
        self.assertLess(similarity, 50000)

    def test_create_correct_number_transformed_datasets(self):
        new_datasets = self.create_dataset_vae.generate_new_datasets(
            cvae=self.model,
            z_mean=self.z_mean,
            z_log_var=self.z_log_var,
            transf_param=[0.5, 1, 2, 5, 10, 15],
            n_versions=2,
            n_samples=3,
        )
        self.assertEqual(new_datasets.shape, (2, 3, 36, 5))
