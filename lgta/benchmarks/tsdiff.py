"""
TSDiff benchmark (Kollovieh et al., NeurIPS 2023).

Uses the original implementation from amazon-science/unconditional-time-series-diffusion
via an adapter to our fit/generate API. Requires: pip install
'git+https://github.com/amazon-science/unconditional-time-series-diffusion.git'
"""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lgta.benchmarks.base import TimeSeriesGenerator

try:
    from uncond_ts_diff.configs import diffusion_small_config
    from uncond_ts_diff.model.diffusion.tsdiff import TSDiff
    from uncond_ts_diff.utils import linear_beta_schedule
except ImportError as e:
    raise ImportError(
        "TSDiff benchmark requires the uncond-ts-diff package. Install with: "
        "pip install 'git+https://github.com/amazon-science/unconditional-time-series-diffusion.git'"
    ) from e


def _fit_tsdiff(
    data: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> TSDiff:
    n_timesteps, n_series = data.shape
    backbone_params = diffusion_small_config["backbone_parameters"].copy()
    model = TSDiff(
        backbone_parameters=backbone_params,
        timesteps=diffusion_small_config["timesteps"],
        diffusion_scheduler=linear_beta_schedule,
        context_length=0,
        prediction_length=n_timesteps,
        use_lags=False,
        use_features=False,
        normalization="none",
        init_skip=True,
        lr=lr,
    )
    model = model.to(device)
    model.train()
    sequences = torch.from_numpy(
        data.T.reshape(n_series, n_timesteps, 1).astype(np.float32)
    ).to(device)
    dataset = TensorDataset(sequences)
    actual_batch = max(1, min(batch_size, n_series))
    loader = DataLoader(
        dataset,
        batch_size=actual_batch,
        shuffle=True,
        drop_last=n_series > actual_batch,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    torch.manual_seed(seed)
    for _ in range(epochs):
        for (batch,) in loader:
            t = torch.randint(
                0, model.timesteps, (batch.size(0),), device=device, dtype=torch.long
            )
            loss, _, _ = model.p_losses(batch, t, features=None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def _generate_tsdiff(
    model: TSDiff, n_timesteps: int, n_series: int, device: torch.device
) -> np.ndarray:
    model.eval()
    samples = model.sample_n(num_samples=n_series, return_lags=False)
    return np.asarray(samples, dtype=np.float64).T


class TSDiffGenerator(TimeSeriesGenerator):
    """TSDiff (Kollovieh et al., NeurIPS 2023) via the original implementation.
    Requires uncond-ts-diff to be installed."""

    def __init__(
        self,
        epochs: int = 500,
        batch_size: int = 64,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._model: Optional[TSDiff] = None

    def _fit(self, data: np.ndarray) -> None:
        self._model = _fit_tsdiff(
            data,
            self.device,
            self.seed,
            self.epochs,
            self.batch_size,
            self.lr,
        )

    @torch.no_grad()
    def _generate(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TSDiffGenerator has not been fitted; call fit(data) first.")
        return _generate_tsdiff(
            self._model,
            self._n_timesteps,
            self._n_series,
            self.device,
        )
