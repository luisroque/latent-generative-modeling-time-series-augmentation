"""
Windowed TimeVAE benchmark.

Applies the base TimeVAE model (see lgta/benchmarks/timevae.py) to sliding
windows instead of whole series, following the official repository's FAQ
recommendation for multiple/long series: "generate windows from each series
and combine" (univariate windows, D=1, at least a few hundred samples).

Because the downstream pipeline requires a full synthetic panel
(n_timesteps, n_series), generation cannot use TimeVAE's native prior
sampling (prior windows carry no series identity or temporal position).
Instead, this variant encodes each real window, samples the latent from the
posterior z ~ N(mu, sigma), decodes, and reconstructs each series by
overlap-adding the decoded windows — the same window-and-reconstruct scheme
LGTA uses, but without LGTA's semantic latent-space transformations.
Repeated generate() calls draw fresh posterior samples, giving stochastic
variants.

Architecture, loss, scaling, and training protocol are identical to the
whole-series TimeVAE benchmark (official base TimeVAE defaults); the window
size defaults to 10 to match the LGTA CVAE, and training windows are
subsampled to max_train_windows to keep the official batch-16 protocol
tractable on large panels.
"""

import numpy as np
import torch

from lgta.benchmarks.base import TimeSeriesGenerator
from lgta.benchmarks.timevae import _Decoder, _Encoder, _MinMaxScaler


class TimeVAEWindowedGenerator(TimeSeriesGenerator):
    """Base TimeVAE trained on univariate sliding windows (posterior generation)."""

    def __init__(
        self,
        window_size: int = 10,
        latent_dim: int = 8,
        hidden_layer_sizes: tuple[int, ...] = (50, 100, 200),
        reconstruction_wt: float = 3.0,
        batch_size: int = 16,
        max_epochs: int = 1000,
        lr: float = 1e-3,
        max_train_windows: int = 5000,
        infer_batch: int = 4096,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.reconstruction_wt = reconstruction_wt
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.max_train_windows = max_train_windows
        self.infer_batch = infer_batch
        self._minmax = _MinMaxScaler()
        self._panel: np.ndarray | None = None  # standardised (T, D) train panel

    # ------------------------------------------------------------------

    def _build(self) -> None:
        w = self.window_size
        self._enc = _Encoder(w, 1, self.hidden_layer_sizes, self.latent_dim).to(
            self.device
        )
        self._dec = _Decoder(
            w, 1, self.hidden_layer_sizes, self.latent_dim, self._enc.final_len
        ).to(self.device)

    def _model_state(self) -> dict:
        return {
            "enc": self._enc.state_dict(),
            "dec": self._dec.state_dict(),
            "minmax_mini": self._minmax.mini,
            "minmax_range": self._minmax.range,
            "panel": self._panel,
        }

    def _restore_model_state(self, state: dict) -> None:
        self._build()
        self._enc.load_state_dict(state["enc"])
        self._dec.load_state_dict(state["dec"])
        self._minmax.mini = state["minmax_mini"]
        self._minmax.range = state["minmax_range"]
        self._panel = state["panel"]

    def _all_windows(self, data: np.ndarray) -> np.ndarray:
        """All sliding windows of every series: (D * (T - w + 1), w, 1)."""
        T, D = data.shape
        w = self.window_size
        # (D, T) -> (D, n_off, w) with stride-1 offsets, then stack series
        wins = np.lib.stride_tricks.sliding_window_view(data.T, w, axis=1)
        return wins.reshape(-1, w, 1).astype(np.float32)

    def _loss(self, batch: torch.Tensor):
        mu, logvar = self._enc(batch)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        recon = self._dec(z)
        recon_loss = ((batch - recon) ** 2).sum()
        recon_loss = recon_loss + ((batch.mean(dim=2) - recon.mean(dim=2)) ** 2).sum()
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        return self.reconstruction_wt * recon_loss + kl_loss, recon_loss, kl_loss

    def _fit(self, data: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._panel = data.astype(np.float32)

        windows = self._all_windows(self._panel)
        windows = self._minmax.fit_transform(windows).astype(np.float32)

        n = windows.shape[0]
        if n > self.max_train_windows:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(n, size=self.max_train_windows, replace=False)
            train_windows = windows[idx]
        else:
            train_windows = windows
        print(
            f"  [TimeVAE-W] training on {train_windows.shape[0]}/{n} windows "
            f"(window={self.window_size})"
        )

        tensor = torch.from_numpy(train_windows).to(self.device)
        self._build()
        params = list(self._enc.parameters()) + list(self._dec.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, eps=1e-7)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(tensor),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

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
                    f"  [TimeVAE-W] epoch {epoch+1}/{self.max_epochs}  "
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
                        f"  [TimeVAE-W] early stopping at epoch {epoch+1}  "
                        f"loss={epoch_loss:.4f}"
                    )
                    break

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate(self) -> np.ndarray:
        T, D = self._n_timesteps, self._n_series
        w = self.window_size
        n_off = T - w + 1

        windows = self._all_windows(self._panel)  # (D * n_off, w, 1)
        windows = self._minmax.transform(windows).astype(np.float32)

        decoded_parts: list[np.ndarray] = []
        for start in range(0, windows.shape[0], self.infer_batch):
            chunk = torch.from_numpy(windows[start : start + self.infer_batch]).to(
                self.device
            )
            mu, logvar = self._enc(chunk)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
            decoded_parts.append(self._dec(z).cpu().numpy())
        decoded = np.concatenate(decoded_parts, axis=0)  # (D * n_off, w, 1)
        decoded = self._minmax.inverse_transform(decoded)
        decoded = decoded.reshape(D, n_off, w)

        # Overlap-add reconstruction: window at offset i covers t = i .. i+w-1.
        acc = np.zeros((T, D), dtype=np.float64)
        cnt = np.zeros((T, 1), dtype=np.float64)
        for k in range(w):
            acc[k : k + n_off, :] += decoded[:, :, k].T
            cnt[k : k + n_off, 0] += 1.0
        return (acc / cnt).astype(np.float32)
