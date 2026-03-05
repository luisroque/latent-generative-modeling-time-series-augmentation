"""
TimeVAE benchmark (Desai et al., ICLR 2022).

Standard LSTM-based variational autoencoder for time series. Unlike LGTA,
this has no temporal latent structure and no equivariance regularisation—it
uses a single global latent vector per series.
"""

import numpy as np
import torch
import torch.nn as nn

from lgta.benchmarks.base import TimeSeriesGenerator


class _Encoder(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden_dim, num_layers=2, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h, _) = self.rnn(x)
        h = h[-1]
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc(z)).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.rnn(h)
        return self.out(out)


class TimeVAEGenerator(TimeSeriesGenerator):
    """Vanilla LSTM-VAE. Learns per-series generation without temporal latent."""

    def __init__(
        self,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        epochs: int = 500,
        batch_size: int = 64,
        lr: float = 1e-3,
        kl_weight: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.kl_weight = kl_weight

    def _fit(self, data: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        T, D = data.shape
        sequences = torch.from_numpy(
            data.T.reshape(D, T, 1).astype(np.float32)
        ).to(self.device)

        self._enc = _Encoder(T, self.hidden_dim, self.latent_dim).to(self.device)
        self._dec = _Decoder(T, self.hidden_dim, self.latent_dim).to(self.device)
        opt = torch.optim.Adam(
            list(self._enc.parameters()) + list(self._dec.parameters()), lr=self.lr
        )
        log_every = max(1, self.epochs // 10)

        for epoch in range(self.epochs):
            idx = torch.randperm(D)[: self.batch_size]
            batch = sequences[idx]
            mu, logvar = self._enc(batch)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
            recon = self._dec(z)

            recon_loss = nn.functional.mse_loss(recon, batch)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + self.kl_weight * kl_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % log_every == 0 or epoch == self.epochs - 1:
                print(
                    f"  [TimeVAE] epoch {epoch+1}/{self.epochs}  "
                    f"recon={recon_loss.item():.4f}  KL={kl_loss.item():.4f}  "
                    f"loss={loss.item():.4f}"
                )

    @torch.no_grad()
    def _generate(self) -> np.ndarray:
        T, D = self._n_timesteps, self._n_series
        z = torch.randn(D, self.latent_dim, device=self.device)
        x = self._dec(z).cpu().numpy().squeeze(-1)
        return x.T
