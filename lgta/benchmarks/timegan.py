"""
TimeGAN benchmark (Yoon et al., NeurIPS 2019).

PyTorch implementation aligned with the original TimeGAN codebase:
https://github.com/jsyoon0823/TimeGAN

Uses MinMax scaling, uniform Z, recovery with sigmoid, supervisor with
num_layers-1, and the original loss/training schedule (E_loss, G_loss with
gamma, two-moment loss, D only when loss > 0.15, generator 2x per step).
"""

import numpy as np
import torch
import torch.nn as nn

from lgta.benchmarks.base import TimeSeriesGenerator


def _minmax_scale(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-Max normalizer as in original TimeGAN (utils / timegan.py)."""
    min_val = np.min(data, axis=0, keepdims=True)
    data_centered = data - min_val
    max_val = np.max(data, axis=0, keepdims=True)
    norm_data = data_centered / (max_val + 1e-7)
    return norm_data.astype(np.float32), min_val.squeeze(0), max_val.squeeze(0)


def _minmax_inverse(norm_data: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
    """Inverse MinMax for denormalization."""
    return norm_data * (max_val + 1e-7) + min_val


class _Embedder(nn.Module):
    """Embedding network: original feature space -> latent space (sigmoid on output)."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        return torch.sigmoid(self.fc(h))


class _Recovery(nn.Module):
    """Recovery network: latent -> original space. Original uses sigmoid on output."""

    def __init__(self, hidden_dim: int, output_dim: int, n_layers: int):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h)
        return torch.sigmoid(self.fc(out))


class _Supervisor(nn.Module):
    """Next-step predictor in latent space. Original uses num_layers-1 RNN."""

    def __init__(self, hidden_dim: int, n_layers: int):
        super().__init__()
        n_sup = max(1, n_layers - 1)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, n_sup, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h)
        return torch.sigmoid(self.fc(out))


class _Generator(nn.Module):
    """Generator: Z -> latent embedding (sigmoid)."""

    def __init__(self, noise_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.rnn = nn.GRU(noise_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(z)
        return torch.sigmoid(self.fc(out))


class _Discriminator(nn.Module):
    """Discriminator on latent space (logits, no sigmoid)."""

    def __init__(self, hidden_dim: int, n_layers: int):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h)
        return self.fc(out)


class TimeGANGenerator(TimeSeriesGenerator):
    """
    TimeGAN aligned with jsyoon0823/TimeGAN.

    Uses MinMax scaling internally (overrides base StandardScaler), iterations
    instead of epochs, gamma=1 for D_loss_fake_e and G_loss_U_e, two-moment
    loss (G_loss_V), conditional D update (only when D_loss > 0.15), and
    generator trained twice per joint step. z_dim = feature dim; Z is uniform [0,1].
    """

    def __init__(
        self,
        hidden_dim: int = 24,
        n_layers: int = 3,
        iterations: int = 2000,
        batch_size: int = 128,
        lr: float = 1e-3,
        gamma: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.iterations = iterations
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self._min_val: np.ndarray
        self._max_val: np.ndarray

    def fit(self, data: np.ndarray) -> "TimeGANGenerator":
        """Fit using MinMax scaling as in original TimeGAN; then _fit."""
        self._n_timesteps, self._n_series = data.shape
        data_norm, self._min_val, self._max_val = _minmax_scale(data)
        self._fit(data_norm)
        return self

    def generate(self) -> np.ndarray:
        """Generate and inverse MinMax transform to match original scale."""
        data_scaled = self._generate()
        return _minmax_inverse(data_scaled, self._min_val, self._max_val)

    def _fit(self, data: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        T, D = data.shape
        dim = D
        z_dim = dim
        sequences = data.T.reshape(D, T, 1).astype(np.float32)
        X = torch.from_numpy(sequences).to(self.device)
        no, seq_len, _ = sequences.shape
        ori_time = [seq_len] * no
        max_seq_len = seq_len

        self._emb = _Embedder(1, self.hidden_dim, self.n_layers).to(self.device)
        self._rec = _Recovery(self.hidden_dim, 1, self.n_layers).to(self.device)
        self._sup = _Supervisor(self.hidden_dim, self.n_layers).to(self.device)
        self._gen = _Generator(z_dim, self.hidden_dim, self.n_layers).to(self.device)
        self._dis = _Discriminator(self.hidden_dim, self.n_layers).to(self.device)

        self._train_embedder(X, ori_time, max_seq_len)
        self._train_supervisor_only(X, ori_time, max_seq_len, z_dim)
        self._train_joint(X, ori_time, max_seq_len, z_dim, no)

    def _random_z(self, batch_size: int, seq_len: int, z_dim: int) -> torch.Tensor:
        """Random Z uniform in [0, 1] as in original random_generator."""
        return torch.rand(batch_size, seq_len, z_dim, device=self.device, dtype=torch.float32)

    def _train_embedder(self, X: torch.Tensor, ori_time: list[int], max_seq_len: int) -> None:
        """Phase 1: embedder + recovery. E_loss0 = 10*sqrt(MSE(X, X_tilde))."""
        opt = torch.optim.Adam(
            list(self._emb.parameters()) + list(self._rec.parameters()), lr=self.lr
        )
        n = X.size(0)
        log_every = max(1, self.iterations // 10)
        for itt in range(self.iterations):
            idx = torch.randperm(n, device=self.device)[: self.batch_size]
            batch = X[idx]
            h = self._emb(batch)
            x_tilde = self._rec(h)
            e_loss_t0 = nn.functional.mse_loss(x_tilde, batch)
            e_loss0 = 10.0 * torch.sqrt(e_loss_t0 + 1e-8)
            opt.zero_grad()
            e_loss0.backward()
            opt.step()
            if itt % log_every == 0 or itt == self.iterations - 1:
                print(f"  [TimeGAN phase 1/3 embedder] iter {itt+1}/{self.iterations}  E_loss0={e_loss0.item():.4f}")

    def _train_supervisor_only(
        self, X: torch.Tensor, ori_time: list[int], max_seq_len: int, z_dim: int
    ) -> None:
        """Phase 2: G_loss_S = MSE(H[:,1:,:], supervisor(H[:,:-1,:])) on real H. Original trains only supervisor (G in var_list but no gradient from G_loss_S)."""
        opt_sup = torch.optim.Adam(self._sup.parameters(), lr=self.lr)
        n = X.size(0)
        log_every = max(1, self.iterations // 10)
        for itt in range(self.iterations):
            idx = torch.randperm(n, device=self.device)[: self.batch_size]
            batch = X[idx]
            with torch.no_grad():
                h_real = self._emb(batch)
            h_hat_supervise = self._sup(h_real[:, :-1, :])
            g_loss_s = nn.functional.mse_loss(h_hat_supervise, h_real[:, 1:, :])
            opt_sup.zero_grad()
            g_loss_s.backward()
            opt_sup.step()
            if itt % log_every == 0 or itt == self.iterations - 1:
                print(f"  [TimeGAN phase 2/3 supervisor] iter {itt+1}/{self.iterations}  G_loss_S={g_loss_s.item():.4f}")

    def _train_joint(
        self, X: torch.Tensor, ori_time: list[int], max_seq_len: int, z_dim: int, no: int
    ) -> None:
        """Phase 3: joint training. D_loss includes gamma*D_fake_e; G includes gamma*G_U_e, 100*sqrt(G_S), 100*G_V; E_loss = 10*sqrt(E_T0)+0.1*G_S. D only when > 0.15; G twice per step."""
        opt_gs = torch.optim.Adam(
            list(self._gen.parameters()) + list(self._sup.parameters()), lr=self.lr
        )
        opt_d = torch.optim.Adam(self._dis.parameters(), lr=self.lr)
        opt_e = torch.optim.Adam(
            list(self._emb.parameters()) + list(self._rec.parameters()), lr=self.lr
        )
        n = X.size(0)
        log_every = max(1, self.iterations // 10)

        for itt in range(self.iterations):
            step_d_loss = torch.tensor(0.0, device=self.device)
            step_g_loss_u = torch.tensor(0.0, device=self.device)
            step_g_loss_s = torch.tensor(0.0, device=self.device)
            step_g_loss_v = torch.tensor(0.0, device=self.device)
            step_e_loss_t0 = torch.tensor(0.0, device=self.device)

            for _ in range(2):
                idx = torch.randperm(n, device=self.device)[: self.batch_size]
                batch = X[idx]
                B, T_len, _ = batch.shape
                z = self._random_z(B, T_len, z_dim)

                h_real = self._emb(batch)
                e_hat = self._gen(z)
                h_hat = self._sup(e_hat)
                x_hat = self._rec(h_hat)
                h_hat_supervise = self._sup(h_real[:, :-1, :])
                g_loss_s = nn.functional.mse_loss(h_real[:, 1:, :], h_hat_supervise)

                y_fake = self._dis(h_hat)
                y_real = self._dis(h_real)
                y_fake_e = self._dis(e_hat)
                g_loss_u = nn.functional.binary_cross_entropy_with_logits(
                    y_fake, torch.ones_like(y_fake)
                )
                g_loss_u_e = nn.functional.binary_cross_entropy_with_logits(
                    y_fake_e, torch.ones_like(y_fake_e)
                )
                m1_hat = x_hat.mean(dim=0)
                m1 = batch.mean(dim=0)
                v1_hat = x_hat.var(dim=0) + 1e-6
                v1 = batch.var(dim=0) + 1e-6
                g_loss_v1 = (torch.sqrt(v1_hat) - torch.sqrt(v1)).abs().mean()
                g_loss_v2 = (m1_hat - m1).abs().mean()
                g_loss_v = g_loss_v1 + g_loss_v2
                g_loss = (
                    g_loss_u
                    + self.gamma * g_loss_u_e
                    + 100.0 * torch.sqrt(g_loss_s + 1e-8)
                    + 100.0 * g_loss_v
                )
                opt_gs.zero_grad()
                g_loss.backward()
                opt_gs.step()

                x_tilde = self._rec(self._emb(batch))
                e_loss_t0 = nn.functional.mse_loss(x_tilde, batch)
                e_loss = 10.0 * torch.sqrt(e_loss_t0 + 1e-8) + 0.1 * g_loss_s.detach()
                opt_e.zero_grad()
                e_loss.backward()
                opt_e.step()

                step_g_loss_u = g_loss_u.detach()
                step_g_loss_s = g_loss_s.detach()
                step_g_loss_v = g_loss_v.detach()
                step_e_loss_t0 = e_loss_t0.detach()

            idx = torch.randperm(n, device=self.device)[: self.batch_size]
            batch = X[idx]
            B, T_len, _ = batch.shape
            z = self._random_z(B, T_len, z_dim)
            with torch.no_grad():
                h_real = self._emb(batch)
                e_hat = self._gen(z)
                h_hat = self._sup(e_hat)
                y_fake = self._dis(h_hat)
                y_real = self._dis(h_real)
                y_fake_e = self._dis(e_hat)
                d_loss_real = nn.functional.binary_cross_entropy_with_logits(
                    y_real, torch.ones_like(y_real)
                )
                d_loss_fake = nn.functional.binary_cross_entropy_with_logits(
                    y_fake, torch.zeros_like(y_fake)
                )
                d_loss_fake_e = nn.functional.binary_cross_entropy_with_logits(
                    y_fake_e, torch.zeros_like(y_fake_e)
                )
                check_d_loss = (d_loss_real + d_loss_fake + self.gamma * d_loss_fake_e).item()
            if check_d_loss > 0.15:
                z = self._random_z(B, T_len, z_dim)
                h_real = self._emb(batch)
                e_hat = self._gen(z)
                h_hat = self._sup(e_hat)
                y_fake = self._dis(h_hat)
                y_real = self._dis(h_real)
                y_fake_e = self._dis(e_hat)
                d_loss_real = nn.functional.binary_cross_entropy_with_logits(
                    y_real, torch.ones_like(y_real)
                )
                d_loss_fake = nn.functional.binary_cross_entropy_with_logits(
                    y_fake, torch.zeros_like(y_fake)
                )
                d_loss_fake_e = nn.functional.binary_cross_entropy_with_logits(
                    y_fake_e, torch.zeros_like(y_fake_e)
                )
                d_loss = d_loss_real + d_loss_fake + self.gamma * d_loss_fake_e
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                step_d_loss = d_loss.detach()

            if itt % log_every == 0 or itt == self.iterations - 1:
                print(
                    f"  [TimeGAN phase 3/3 joint] iter {itt+1}/{self.iterations}  "
                    f"D={step_d_loss.item():.4f}  G_U={step_g_loss_u.item():.4f}  "
                    f"G_S={step_g_loss_s.item():.4f}  G_V={step_g_loss_v.item():.4f}  "
                    f"E_T0={step_e_loss_t0.item():.4f}"
                )

    @torch.no_grad()
    def _generate(self) -> np.ndarray:
        T, D = self._n_timesteps, self._n_series
        z = torch.rand(D, T, D, device=self.device, dtype=torch.float32)
        h = self._sup(self._gen(z))
        x = self._rec(h)
        out = x.cpu().numpy().squeeze(-1)
        return out.T
