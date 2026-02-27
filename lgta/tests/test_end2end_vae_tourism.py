import unittest
from lgta.tests.conftest import skip_unless_dataset
from lgta.model.create_dataset_versions_vae import (
    CreateTransformedVersionsCVAE,
)
from lgta.transformations.compute_similarities_summary_metrics import (
    compute_similarity_transformed_vs_original,
)


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        skip_unless_dataset("tourism")
        self.create_dataset_vae = CreateTransformedVersionsCVAE(
            dataset_name="tourism", freq="M", top=5
        )

        self.model, _, _ = self.create_dataset_vae.fit(epochs=1, load_weights=False)
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

        self.assertTrue(
            compute_similarity_transformed_vs_original(
                dec_pred_hat, self.create_dataset_vae.X_train_raw
            )[0]
            < 20
        )

    def test_create_correct_number_transformed_datasets(self):
        new_datasets = self.create_dataset_vae.generate_new_datasets(
            cvae=self.model,
            z_mean=self.z_mean,
            z_log_var=self.z_log_var,
            transf_param=[0.5, 1, 2, 5, 10, 15],
            n_versions=2,
            n_samples=3,
        )
        # shape (n_versions, n_samples, n_samples, n_series) — top=5 so 5 series
        self.assertTrue(new_datasets.shape == (2, 3, 228, 5))
