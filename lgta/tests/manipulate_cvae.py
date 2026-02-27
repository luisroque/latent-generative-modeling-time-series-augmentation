import unittest
from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from lgta.transformations.compute_similarities_summary_metrics import (
    compute_similarity_transformed_vs_original,
)


class TestModel(unittest.TestCase):
    """
    Experiments with the CVAE model's ability to generate new time series
    and compute similarity between the generated series and the original.
    """

    def setUp(self) -> None:
        self.dataset_name = "tourism_small"
        self.freq = "Q"

        self.create_dataset_vae = CreateTransformedVersionsCVAE(
            dataset_name=self.dataset_name,
            freq=self.freq,
        )
        self.epochs = 5
        self.load_weights = True
        self.model, _, _ = self.create_dataset_vae.fit(
            epochs=self.epochs, load_weights=self.load_weights
        )
        self.similarity_threshold = 50000

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
            transformation="magnitude_warp",
            transf_param=5,
        )

        similarity_score = compute_similarity_transformed_vs_original(
            dec_pred_hat, self.create_dataset_vae.X_train_raw
        )[0]

        self.assertLess(
            similarity_score,
            self.similarity_threshold,
            f"Similarity score {similarity_score} exceeds threshold {self.similarity_threshold}",
        )
