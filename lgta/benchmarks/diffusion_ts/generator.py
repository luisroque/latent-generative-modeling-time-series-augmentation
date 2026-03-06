"""
DiffusionTSGenerator: adapts the Diffusion-TS model (Yuan & Qiao, ICLR 2024)
to the TimeSeriesGenerator interface used by the downstream forecasting
experiment.  Training data of shape (n_timesteps, n_series) is split into
overlapping windows, and generated windows are stitched back to reconstruct
the full length.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


from lgta.benchmarks.base import TimeSeriesGenerator
from lgta.benchmarks.diffusion_ts.gaussian_diffusion import DiffusionTS
from lgta.benchmarks.diffusion_ts.lr_scheduler import ReduceLROnPlateauWithWarmup
from lgta.benchmarks.diffusion_ts.model_utils import (
    EMA,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)

_GRADIENT_ACCUMULATE_EVERY = 2
_EMA_DECAY = 0.995
_EMA_UPDATE_EVERY = 10
_WARMUP_STEPS = 500
_WARMUP_LR = 8e-4


class DiffusionTSGenerator(TimeSeriesGenerator):
    """Diffusion-TS benchmark (Yuan & Qiao, ICLR 2024).

    Uses an encoder-decoder transformer with disentangled temporal
    representations (trend + season decomposition) inside a denoising
    diffusion probabilistic framework.
    """

    def __init__(
        self,
        seq_length: int = 24,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        diffusion_steps: int = 500,
        sampling_steps: int | None = None,
        epochs: int = 5000,
        batch_size: int = 64,
        lr: float = 1e-5,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.seq_length = seq_length
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.diffusion_steps = diffusion_steps
        self.sampling_steps = sampling_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _fit(self, data: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)  # noqa: NPY002

        T, D = data.shape
        W = min(self.seq_length, T)
        self._actual_seq_length = W

        windows = _create_windows(data, W)

        self._window_min = windows.min(axis=(0, 1), keepdims=True)
        self._window_max = windows.max(axis=(0, 1), keepdims=True)
        windows_01 = (windows - self._window_min) / (
            self._window_max - self._window_min + 1e-8
        )
        windows_neg = normalize_to_neg_one_to_one(
            torch.from_numpy(windows_01.astype(np.float32))
        )

        self._diffusion = DiffusionTS(
            seq_length=W,
            feature_size=D,
            n_layer_enc=self.n_layers,
            n_layer_dec=self.n_layers,
            d_model=self.d_model,
            timesteps=self.diffusion_steps,
            sampling_timesteps=self.sampling_steps,
            loss_type="l1",
            beta_schedule="cosine",
            n_heads=self.n_heads,
            mlp_hidden_times=4,
            attn_pd=0.0,
            resid_pd=0.0,
            kernel_size=1,
            padding_size=0,
        ).to(self.device)

        self._ema = EMA(
            self._diffusion, beta=_EMA_DECAY, update_every=_EMA_UPDATE_EVERY
        ).to(self.device)

        opt = Adam(
            filter(lambda p: p.requires_grad, self._diffusion.parameters()),
            lr=self.lr,
            betas=(0.9, 0.96),
        )
        scheduler = ReduceLROnPlateauWithWarmup(
            opt,
            factor=0.5,
            patience=2000,
            min_lr=self.lr,
            threshold=0.1,
            threshold_mode="rel",
            warmup_lr=_WARMUP_LR,
            warmup=_WARMUP_STEPS,
            verbose=False,
        )

        n_windows = len(windows_neg)
        effective_bs = min(self.batch_size, n_windows)
        dl = DataLoader(
            TensorDataset(windows_neg),
            batch_size=effective_bs,
            shuffle=True,
            drop_last=(n_windows > effective_bs),
        )
        dl_iter = _cycle(dl)
        log_every = max(1, self.epochs // 10)

        for step in tqdm(range(self.epochs), desc="[Diffusion-TS] training"):
            total_loss = 0.0
            for _ in range(_GRADIENT_ACCUMULATE_EVERY):
                (batch,) = next(dl_iter)
                batch = batch.to(self.device)
                loss = self._diffusion(batch, target=batch)
                loss = loss / _GRADIENT_ACCUMULATE_EVERY
                loss.backward()
                total_loss += loss.item()

            clip_grad_norm_(self._diffusion.parameters(), 1.0)
            opt.step()
            scheduler.step(total_loss)
            opt.zero_grad()
            self._ema.update()

            if step % log_every == 0 or step == self.epochs - 1:
                print(
                    f"  [Diffusion-TS] step {step + 1}/{self.epochs}  "
                    f"loss={total_loss:.6f}"
                )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate(self) -> np.ndarray:
        T, D = self._n_timesteps, self._n_series
        W = self._actual_seq_length
        n_windows = max(1, math.ceil(T / W))

        samples = np.empty((0, W, D))
        remaining = n_windows
        while remaining > 0:
            bs = min(remaining, 256)
            batch = self._ema.ema_model.generate_mts(batch_size=bs)
            samples = np.concatenate([samples, batch.cpu().numpy()], axis=0)
            remaining -= bs

        samples_t = unnormalize_to_zero_to_one(torch.from_numpy(samples)).numpy()
        samples_t = (
            samples_t * (self._window_max - self._window_min + 1e-8) + self._window_min
        )

        full = samples_t.reshape(-1, D)[:T]
        return full


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _create_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    T, D = data.shape
    if T <= window_size:
        return data[np.newaxis, :, :]
    n_windows = T - window_size + 1
    return np.stack([data[i : i + window_size] for i in range(n_windows)], axis=0)


def _cycle(dl: DataLoader):
    while True:
        yield from dl
