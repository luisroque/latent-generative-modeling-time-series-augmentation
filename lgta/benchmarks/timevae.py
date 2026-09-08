"""
TimeVAE benchmark (Desai et al., 2021, arXiv:2111.08095).

PyTorch port of the *base* TimeVAE from the official implementation
(https://github.com/abudesai/timeVAE): the default ``timeVAE`` configuration
with the optional interpretable components disabled (trend_poly=0,
custom_seas=None, use_residual_conn=True).  The model is a Conv1D encoder
plus a decoder composed of a level component and a deconvolutional residual
branch.  Loss, scaling, and training protocol follow the official code:

  - reconstruction loss: summed squared error plus a time-axis mean term,
    weighted by reconstruction_wt=3.0; KL summed over batch and latent dims
  - MinMax scaling of the (samples, timesteps, features) tensor to [0, 1]
    (per timestep/feature across samples), applied on top of the pipeline's
    per-series standardisation
  - Adam(lr=1e-3), batch_size=16, up to 1000 epochs with EarlyStopping
    (min_delta 1e-2, patience 50) and ReduceLROnPlateau (factor 0.5,
    patience 30) on the training loss, as configured in the official
    fit_on_data

Unlike LGTA, TimeVAE has no temporal latent structure and no equivariance
regularisation—it uses a single global latent vector per series.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lgta.benchmarks.base import TimeSeriesGenerator


class _MinMaxScaler:
    """Official TimeVAE MinMax scaler: min/range over axis 0 of (N, T, D)."""

    def __init__(self) -> None:
        self.mini: np.ndarray | None = None
        self.range: np.ndarray | None = None

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.mini = np.min(data, axis=0)
        self.range = np.max(data, axis=0) - self.mini
        return self.transform(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mini) / (self.range + 1e-7)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.range + self.mini


class _Encoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        hidden_layer_sizes: list[int],
        latent_dim: int,
    ):
        super().__init__()
        convs = []
        in_ch = feat_dim
        for h in hidden_layer_sizes:
            # kernel 3, stride 2, padding 1: same output length as TF
            # Conv1D(padding='same', strides=2) (TF pads asymmetrically for
            # even input lengths; immaterial when training from scratch)
            convs.append(nn.Conv1d(in_ch, h, kernel_size=3, stride=2, padding=1))
            in_ch = h
        self.convs = nn.ModuleList(convs)
        t = seq_len
        for _ in hidden_layer_sizes:
            t = (t + 1) // 2
        self.final_len = t
        flat_dim = t * hidden_layer_sizes[-1]
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x.permute(0, 2, 1)
        for conv in self.convs:
            h = torch.relu(conv(h))
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    """Level component + deconvolutional residual branch (base TimeVAE)."""

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        hidden_layer_sizes: list[int],
        latent_dim: int,
        encoder_final_len: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.encoder_final_len = encoder_final_len
        self.hidden_last = hidden_layer_sizes[-1]

        self.level_fc1 = nn.Linear(latent_dim, feat_dim)
        self.level_fc2 = nn.Linear(feat_dim, feat_dim)

        self.res_fc = nn.Linear(latent_dim, encoder_final_len * self.hidden_last)
        deconvs = []
        in_ch = self.hidden_last
        for h in reversed(hidden_layer_sizes[:-1]):
            # kernel 3, stride 2, padding 1, output_padding 1: doubles the
            # length, like TF Conv1DTranspose(padding='same', strides=2)
            deconvs.append(
                nn.ConvTranspose1d(in_ch, h, 3, stride=2, padding=1, output_padding=1)
            )
            in_ch = h
        deconvs.append(
            nn.ConvTranspose1d(in_ch, feat_dim, 3, stride=2, padding=1, output_padding=1)
        )
        self.deconvs = nn.ModuleList(deconvs)
        out_len = encoder_final_len * (2 ** len(hidden_layer_sizes))
        self.res_out = nn.Linear(out_len * feat_dim, seq_len * feat_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        n = z.shape[0]
        level = torch.relu(self.level_fc1(z))
        level = self.level_fc2(level).unsqueeze(1)  # (N, 1, D), broadcast over T

        h = torch.relu(self.res_fc(z))
        h = h.view(n, self.encoder_final_len, self.hidden_last).permute(0, 2, 1)
        for deconv in self.deconvs:
            h = torch.relu(deconv(h))
        h = h.permute(0, 2, 1).flatten(1)
        residuals = self.res_out(h).view(n, self.seq_len, self.feat_dim)
        return residuals + level


class TimeVAEGenerator(TimeSeriesGenerator):
    """Base TimeVAE (official architecture/loss/training, PyTorch port)."""

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_layer_sizes: tuple[int, ...] = (50, 100, 200),
        reconstruction_wt: float = 3.0,
        batch_size: int = 16,
        max_epochs: int = 1000,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.reconstruction_wt = reconstruction_wt
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self._minmax = _MinMaxScaler()

    def _build(self, seq_len: int) -> None:
        self._enc = _Encoder(
            seq_len, 1, self.hidden_layer_sizes, self.latent_dim
        ).to(self.device)
        self._dec = _Decoder(
            seq_len, 1, self.hidden_layer_sizes, self.latent_dim, self._enc.final_len
        ).to(self.device)

    def _model_state(self) -> dict:
        return {
            "enc": self._enc.state_dict(),
            "dec": self._dec.state_dict(),
            "minmax_mini": self._minmax.mini,
            "minmax_range": self._minmax.range,
        }

    def _restore_model_state(self, state: dict) -> None:
        self._build(self._n_timesteps)
        self._enc.load_state_dict(state["enc"])
        self._dec.load_state_dict(state["dec"])
        self._minmax.mini = state["minmax_mini"]
        self._minmax.range = state["minmax_range"]

    def _loss(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self._enc(batch)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        recon = self._dec(z)

        # Official reconstruction loss: overall SSE + SSE of means over the
        # feature axis (reduce_sum, not mean, matching the TF code).
        recon_loss = ((batch - recon) ** 2).sum()
        recon_loss = recon_loss + ((batch.mean(dim=2) - recon.mean(dim=2)) ** 2).sum()

        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        total = self.reconstruction_wt * recon_loss + kl_loss
        return total, recon_loss, kl_loss

    def _fit(self, data: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        T, D = data.shape
        sequences = data.T.reshape(D, T, 1).astype(np.float32)
        sequences = self._minmax.fit_transform(sequences).astype(np.float32)
        tensor = torch.from_numpy(sequences).to(self.device)

        self._build(T)
        params = list(self._enc.parameters()) + list(self._dec.parameters())
        # eps=1e-7 matches the TF/Keras Adam default
        opt = torch.optim.Adam(params, lr=self.lr, eps=1e-7)

        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        # The official fit_on_data callbacks: EarlyStopping (min_delta 1e-2,
        # patience 50) and ReduceLROnPlateau (factor 0.5, patience 30,
        # min_delta 1e-4), each tracking its own best, as in Keras.
        es_best, es_wait = float("inf"), 0
        lr_best, lr_wait = float("inf"), 0
        log_every = max(1, self.max_epochs // 10)

        for epoch in range(self.max_epochs):
            total_sum = recon_sum = kl_sum = 0.0
            n_batches = 0
            for (batch,) in loader:
                total, recon_loss, kl_loss = self._loss(batch)
                opt.zero_grad()
                total.backward()
                opt.step()
                total_sum += total.item()
                recon_sum += recon_loss.item()
                kl_sum += kl_loss.item()
                n_batches += 1
            epoch_loss = total_sum / n_batches

            if epoch % log_every == 0 or epoch == self.max_epochs - 1:
                print(
                    f"  [TimeVAE] epoch {epoch+1}/{self.max_epochs}  "
                    f"recon={recon_sum / n_batches:.4f}  KL={kl_sum / n_batches:.4f}  "
                    f"loss={epoch_loss:.4f}  lr={opt.param_groups[0]['lr']:.2e}"
                )

            if epoch_loss < lr_best - 1e-4:
                lr_best, lr_wait = epoch_loss, 0
            else:
                lr_wait += 1
                if lr_wait >= 30:
                    for g in opt.param_groups:
                        g["lr"] *= 0.5
                    lr_wait = 0

            if epoch_loss < es_best - 1e-2:
                es_best, es_wait = epoch_loss, 0
            else:
                es_wait += 1
                if es_wait >= 50:
                    print(
                        f"  [TimeVAE] early stopping at epoch {epoch+1}  "
                        f"loss={epoch_loss:.4f}"
                    )
                    break

    @torch.no_grad()
    def _generate(self) -> np.ndarray:
        T, D = self._n_timesteps, self._n_series
        z = torch.randn(D, self.latent_dim, device=self.device)
        x = self._dec(z).cpu().numpy()  # (D, T, 1) in [0, 1] scale
        x = self._minmax.inverse_transform(x)
        return x.squeeze(-1).T
