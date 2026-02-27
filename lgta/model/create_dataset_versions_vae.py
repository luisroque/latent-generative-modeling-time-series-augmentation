"""
Training pipeline for the CVAE model. Handles data preprocessing, model fitting,
prediction, and generation of transformed time series datasets.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from pathlib import Path
from typing import Optional, Union
from sklearn.preprocessing import MinMaxScaler
from lgta.model.models import CVAE, get_CVAE
from lgta.feature_engineering.static_features import (
    create_static_features,
)
from lgta.feature_engineering.dynamic_features import (
    create_dynamic_features,
)
from lgta.feature_engineering.feature_transformations import (
    temporalize,
    combine_inputs_to_model,
    detemporalize,
)
from lgta.postprocessing.generative_helper import generate_new_time_series
from lgta.visualization.model_visualization import (
    plot_generated_vs_original,
)
from lgta.preprocessing.pre_processing_datasets import (
    PreprocessDatasets as ppc,
)
from lgta import __version__


class InvalidFrequencyError(Exception):
    pass


class TimeSeriesDataset(TorchDataset):
    """PyTorch Dataset wrapping dynamic features and input data tensors."""

    def __init__(
        self, dynamic_features: np.ndarray, input_data: np.ndarray
    ):
        self.dynamic_features = torch.tensor(
            dynamic_features, dtype=torch.float32
        )
        self.input_data = torch.tensor(input_data, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.input_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dynamic_features[idx], self.input_data[idx]


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_transformations(
    transformation: Optional[Union[str, list[str]]],
    transf_param: Union[float, list[float]],
) -> tuple[Optional[list[str]], Optional[list[float]]]:
    """Convert single transformation/param to lists for uniform chaining logic."""
    if transformation is None:
        return None, None
    if isinstance(transformation, str):
        transformations = [transformation]
        transf_params = (
            [transf_param] if isinstance(transf_param, (int, float)) else transf_param
        )
    else:
        transformations = transformation
        transf_params = (
            transf_param if isinstance(transf_param, list) else [transf_param]
        )
    return transformations, transf_params


class CreateTransformedVersionsCVAE:
    """
    Class for creating transformed versions of the dataset using a Conditional Variational Autoencoder (CVAE).

    This class contains several methods to preprocess data, fit a CVAE, generate new time series, and
    save transformed versions of the dataset. It's designed to be used with time-series data.

    The class follows the Singleton design pattern ensuring that only one instance can exist.

    Args:
        dataset_name: Name of the dataset.
        freq: Frequency of the time series data.
        input_dir: Directory where the input data is located. Defaults to "./".
        transf_data: Type of transformation applied to the data. Defaults to "whole".
        top: Number of top series to select. Defaults to None.
        window_size: Window size for the sliding window. Defaults to 10.
        weekly_m5: If True, use the M5 competition's weekly grouping. Defaults to True.
        test_size: Size of the test set. If None, the size is determined automatically. Defaults to None.
        Below are parameters for the synthetic data creation:
            num_base_series_time_points: Number of base time points in the series. Defaults to 100.
            num_latent_dim: Dimension of the latent space. Defaults to 3.
            num_variants: Number of variants for the transformation. Defaults to 20.
            noise_scale: Scale of the Gaussian noise. Defaults to 0.1.
            amplitude: Amplitude of the time series data. Defaults to 1.0.
    """

    _instance = None

    def __new__(cls, *args, **kwargs) -> "CreateTransformedVersionsCVAE":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        dataset_name: str,
        freq: str,
        input_dir: str = "./",
        transf_data: str = "whole",
        top: int = None,
        window_size: int = 10,
        weekly_m5: bool = True,
        test_size: int = None,
        num_base_series_time_points: int = 100,
        num_latent_dim: int = 3,
        num_variants: int = 20,
        noise_scale: float = 0.1,
        amplitude: float = 1.0,
    ):
        self.dataset_name = dataset_name
        self.input_dir = input_dir
        self.transf_data = transf_data
        self.freq = freq
        self.top = top
        self.test_size = test_size
        self.weekly_m5 = weekly_m5
        self.num_base_series_time_points = num_base_series_time_points
        self.num_latent_dim = num_latent_dim
        self.num_variants = num_variants
        self.noise_scale = noise_scale
        self.amplitude = amplitude
        self.dataset = self._get_dataset()
        if window_size:
            self.window_size = window_size
        data = self.dataset["predict"]["data_matrix"]
        self.y = data
        self.n = data.shape[0]
        self.s = data.shape[1]
        self.n_features = self.s
        self.n_train = self.n - self.window_size + 1
        self.groups = list(self.dataset["train"]["groups_names"].keys())
        self.df = pd.DataFrame(data)
        self.df = pd.concat(
            [self.df, pd.DataFrame(self.dataset["dates"], columns=["Date"])], axis=1
        )[: self.n]
        self.df = self.df.set_index("Date")
        self.preprocess_freq()
        self.input_data = None
        self.device = _get_device()
        self._create_directories()
        self._save_original_file()

    def preprocess_freq(self):
        end_date = None

        if self.freq in ["Q", "QS"]:
            if self.freq == "Q":
                self.freq += "S"
            end_date = self.df.index[-1] + pd.DateOffset(months=self.window_size * 3)
        elif self.freq in ["M", "MS"]:
            if self.freq == "M":
                self.freq += "S"
            end_date = self.df.index[-1] + pd.DateOffset(months=self.window_size)
        elif self.freq == "W":
            end_date = self.df.index[-1] + pd.DateOffset(weeks=self.window_size)
        elif self.freq == "D":
            end_date = self.df.index[-1] + pd.DateOffset(days=self.window_size)
        else:
            raise InvalidFrequencyError(
                f"Invalid frequency - {self.freq}. Please use one of the defined frequencies: Q, QS, M, MS, W, or D."
            )

        ix = pd.date_range(
            start=self.df.index[0],
            end=end_date,
            freq=self.freq,
        )
        self.df_generate = self.df.copy()
        self.df_generate = self.df_generate.reindex(ix)

    def _get_dataset(self):
        ppc_args = {
            "dataset": self.dataset_name,
            "freq": self.freq,
            "input_dir": self.input_dir,
            "top": self.top,
            "test_size": self.test_size,
            "weekly_m5": self.weekly_m5,
            "num_base_series_time_points": self.num_base_series_time_points,
            "num_latent_dim": self.num_latent_dim,
            "num_variants": self.num_variants,
            "noise_scale": self.noise_scale,
            "amplitude": self.amplitude,
        }

        dataset = ppc(**ppc_args).apply_preprocess()
        return dataset

    def _create_directories(self):
        Path(f"{self.input_dir}data").mkdir(parents=True, exist_ok=True)
        Path(f"{self.input_dir}assets/data/transformed_datasets").mkdir(
            parents=True, exist_ok=True
        )
        Path(f"{self.input_dir}assets/model_weights").mkdir(parents=True, exist_ok=True)

    def _save_original_file(self):
        with open(
            f"{self.input_dir}assets/data/transformed_datasets/{self.dataset_name}_original.npy",
            "wb",
        ) as f:
            np.save(f, self.y)

    def _save_version_file(
        self,
        y_new: np.ndarray,
        version: int,
        sample: int,
        transformation: str,
        method: str = "single_transf",
    ) -> None:
        with open(
            f"{self.input_dir}assets/data/transformed_datasets/{self.dataset_name}_version_{version}_{sample}samples_{method}_{transformation}_{self.transf_data}.npy",
            "wb",
        ) as f:
            np.save(f, y_new)

    def _generate_static_features(self, n: int) -> None:
        self.static_features = create_static_features(self.groups, self.dataset)

    def _feature_engineering(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create static and dynamic features as well as apply preprocess to raw time series.

        Returns:
            Tuple of (dynamic_features_inp, X_inp) as numpy arrays.
        """
        self.X_train_raw = self.df.astype(np.float32).to_numpy()

        self.scaler_target = MinMaxScaler().fit(self.X_train_raw)
        X_train_raw_scaled = self.scaler_target.transform(self.X_train_raw)

        if n == self.n:
            self.dynamic_features = create_dynamic_features(self.df_generate, self.freq)
        else:
            self.dynamic_features = create_dynamic_features(self.df, self.freq)

        X_train = temporalize(X_train_raw_scaled, self.window_size)

        self.n_features_concat = X_train.shape[1] + self.dynamic_features.shape[1]

        (self.dynamic_features_inp, X_inp) = combine_inputs_to_model(
            X_train,
            self.dynamic_features,
            self.window_size,
        )

        self.input_data = (self.dynamic_features_inp, X_inp)
        return self.dynamic_features_inp[0], X_inp[0]

    def fit(
        self,
        epochs: int = 1000,
        batch_size: int = 5,
        patience: int = 30,
        latent_dim: int = 2,
        learning_rate: float = 0.001,
        hyper_tuning: bool = False,
        load_weights: bool = True,
    ) -> tuple[CVAE, Optional[dict[str, list[float]]], float]:
        """
        Training our CVAE on the dataset supplied.

        Returns:
            Tuple of (trained model, training history dict or None, best loss).
        """
        dynamic_features_np, X_inp_np = self._feature_engineering(self.n_train)

        n_main_features = X_inp_np.shape[-1]
        n_dyn_features = dynamic_features_np.shape[-1]

        encoder, decoder = get_CVAE(
            window_size=self.window_size,
            n_main_features=n_main_features,
            n_dyn_features=n_dyn_features,
            latent_dim=latent_dim,
        )

        cvae = CVAE(encoder, decoder, self.window_size)
        cvae = cvae.to(self.device)

        weights_folder = f"{self.input_dir}assets/model_weights"
        os.makedirs(weights_folder, exist_ok=True)
        weights_file = os.path.join(
            weights_folder, f"{self.dataset_name}_vae_weights.pt"
        )

        if os.path.exists(weights_file) and not hyper_tuning and load_weights:
            print("Loading existing weights...")
            cvae.load_state_dict(
                torch.load(weights_file, map_location=self.device, weights_only=True)
            )
            return cvae, None, 0.0

        dataset = TimeSeriesDataset(dynamic_features_np, X_inp_np)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(
            cvae.parameters(), lr=learning_rate, weight_decay=0.001
        )

        history: dict[str, list[float]] = {
            "loss": [],
            "reconstruction_loss": [],
            "kl_loss": [],
        }

        best_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        total_params = sum(p.numel() for p in cvae.parameters())
        trainable_params = sum(
            p.numel() for p in cvae.parameters() if p.requires_grad
        )
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")

        for epoch in range(epochs):
            cvae.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n_batches = 0

            for dyn_feat, x_inp in dataloader:
                dyn_feat = dyn_feat.to(self.device)
                x_inp = x_inp.to(self.device)

                optimizer.zero_grad()
                losses = cvae.compute_loss(dyn_feat, x_inp)
                losses["loss"].backward()
                optimizer.step()

                epoch_loss += losses["loss"].item()
                epoch_recon += losses["reconstruction_loss"].item()
                epoch_kl += losses["kl_loss"].item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl = epoch_kl / n_batches

            history["loss"].append(avg_loss)
            history["reconstruction_loss"].append(avg_recon)
            history["kl_loss"].append(avg_kl)

            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in cvae.state_dict().items()}
                torch.save(best_state, weights_file)
                print(
                    f"Epoch {epoch + 1}: loss improved to {avg_loss:.6f} - saving weights"
                )
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            cvae.load_state_dict(best_state)
            cvae = cvae.to(self.device)

        return cvae, history, best_loss

    def predict(
        self, cvae: CVAE
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict original time series using CVAE.

        Returns:
            Tuple containing the reconstructed time series (`X_hat_complete`),
            latent variables (`z`), mean of latent distribution (`z_mean`),
            and log variance of latent distribution (`z_log_var`).
        """
        dynamic_features_np, X_inp_np = self._feature_engineering(self.n_train)

        cvae.eval()
        with torch.no_grad():
            dyn_tensor = torch.tensor(
                dynamic_features_np, dtype=torch.float32, device=self.device
            )
            x_tensor = torch.tensor(
                X_inp_np, dtype=torch.float32, device=self.device
            )

            z_mean, z_log_var, z = cvae.encoder(dyn_tensor, x_tensor)
            preds = cvae.decoder(z, dyn_tensor)

            preds = preds.cpu().numpy()
            z_mean = z_mean.cpu().numpy()
            z_log_var = z_log_var.cpu().numpy()
            z = z.cpu().numpy()

        preds = detemporalize(preds, self.window_size)
        X_hat = self.scaler_target.inverse_transform(preds)
        X_hat_complete = np.concatenate(
            (self.X_train_raw[: self.window_size], X_hat), axis=0
        )

        return X_hat_complete, z, z_mean, z_log_var

    def generate_transformed_time_series(
        self,
        cvae: CVAE,
        z_mean: np.ndarray,
        z_log_var: np.ndarray,
        transformation: Optional[Union[str, list[str]]] = None,
        transf_param: Union[float, list[float]] = 0.5,
        plot_predictions: bool = True,
        n_series_plot: int = 8,
    ) -> np.ndarray:
        """
        Generate new time series by sampling from the CVAE latent space.

        Supports sequential chaining of transformations on latent samples:
        v' = T_n(...T_2(T_1(v, eta_1), eta_2)..., eta_n)
        """
        self._feature_engineering(self.n)

        transformations, transf_params = _normalize_transformations(
            transformation, transf_param
        )

        dec_pred_hat = generate_new_time_series(
            cvae=cvae,
            z_mean=z_mean,
            z_log_var=z_log_var,
            window_size=self.window_size,
            dynamic_features_inp=self.dynamic_features_inp[0],
            scaler_target=self.scaler_target,
            n_features=self.n_features,
            n=self.n,
            transformations=transformations,
            transf_params=transf_params,
            device=self.device,
        )

        if plot_predictions:
            plot_generated_vs_original(
                dec_pred_hat=dec_pred_hat,
                X_train_raw=self.X_train_raw,
                transformation=transformation,
                transf_param=transf_param,
                dataset_name=self.dataset_name,
                n_series=n_series_plot,
                model_version=__version__,
            )
        return dec_pred_hat

    def generate_new_datasets(
        self,
        cvae: CVAE,
        z_mean: np.ndarray,
        z_log_var: np.ndarray,
        transformation: Optional[str] = None,
        transf_param: list[float] = None,
        n_versions: int = 6,
        n_samples: int = 10,
        save: bool = True,
    ) -> np.ndarray:
        """
        Generate multiple dataset versions using different transformation magnitudes.
        """
        if transf_param is None:
            transf_param = [0.5, 2, 4, 10, 20, 50]
        y_new = np.zeros((n_versions, n_samples, self.n, self.s))
        s = 0
        for v in range(1, n_versions + 1):
            for s in range(1, n_samples + 1):
                y_new[v - 1, s - 1] = self.generate_transformed_time_series(
                    cvae=cvae,
                    z_mean=z_mean,
                    z_log_var=z_log_var,
                    transformation=transformation,
                    transf_param=transf_param[v - 1],
                )
            if save:
                self._save_version_file(y_new[v - 1], v, s, "vae")
        return y_new
