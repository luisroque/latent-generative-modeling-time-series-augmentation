"""
L-GTA model architecture: Conditional Variational Autoencoder (CVAE) with
Bi-LSTM encoder/decoder and Variational Multi-Head Attention (VMHA).

Encoder: Bi-LSTM(z) -> h_t, then VMHA(h_t, c) -> phi -> (mu_t, Sigma_t) -> v_t
Decoder: Bi-LSTM(v_t, c) -> psi(y_t) -> z_tilde_t
"""

import torch
import torch.nn as nn
from .helper import Sampling


class PositionalEmbedding(nn.Module):
    """Learned positional embedding added to input sequences."""

    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.empty(max_seq_len, embed_dim)
        )
        nn.init.xavier_uniform_(self.pos_embedding.unsqueeze(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        seq_len = inputs.shape[1]
        return inputs + self.pos_embedding[:seq_len, :].unsqueeze(0)


class VariationalMultiHeadAttention(nn.Module):
    """VMHA(h_t, c): Enriches Bi-LSTM hidden states h_t with condition c through
    self-attention (long-range temporal dependencies) and cross-attention
    (condition-aware representations). Preserves per-timestep temporal structure
    for subsequent variational sampling.
    """

    def __init__(
        self,
        h_dim: int,
        c_dim: int,
        num_heads: int,
        ff_dim: int,
        max_seq_len: int,
        rate: float = 0.3,
    ):
        super().__init__()
        self.pos_embed = PositionalEmbedding(max_seq_len, h_dim)
        self.c_proj = nn.Linear(c_dim, h_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=h_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=h_dim, num_heads=num_heads, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(h_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, h_dim),
        )

        self.ln_self = nn.LayerNorm(h_dim, eps=1e-6)
        self.ln_cross = nn.LayerNorm(h_dim, eps=1e-6)
        self.ln_ffn = nn.LayerNorm(h_dim, eps=1e-6)

        self.drop_self = nn.Dropout(rate)
        self.drop_cross = nn.Dropout(rate)
        self.drop_ffn = nn.Dropout(rate)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.c_proj.weight)
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(
        self, h: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        c_proj = self.c_proj(c)
        h = self.pos_embed(h)

        self_out, _ = self.self_attn(h, h, h)
        self_out = self.drop_self(self_out)
        h = self.ln_self(h + self_out)

        cross_out, _ = self.cross_attn(query=h, key=c_proj, value=c_proj)
        cross_out = self.drop_cross(cross_out)
        h = self.ln_cross(h + cross_out)

        ffn_out = self.ffn(h)
        ffn_out = self.drop_ffn(ffn_out)
        return self.ln_ffn(h + ffn_out)


class Encoder(nn.Module):
    """Bi-LSTM encoder with VMHA and per-timestep variational sampling."""

    def __init__(
        self,
        window_size: int,
        n_main_features: int,
        n_dyn_features: int,
        latent_dim: int,
        num_heads: int = 8,
        ff_dim: int = 256,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_main_features,
            hidden_size=n_main_features,
            batch_first=True,
            bidirectional=True,
        )
        self.batch_norm = nn.BatchNorm1d(window_size)

        self.vmha = VariationalMultiHeadAttention(
            h_dim=n_main_features,
            c_dim=n_dyn_features,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_seq_len=window_size,
        )

        self.dropout = nn.Dropout(0.3)

        self.dense_latent = nn.Linear(n_main_features, latent_dim)
        self.batch_norm_latent = nn.BatchNorm1d(window_size)

        self.z_mean_layer = nn.Linear(latent_dim, latent_dim)
        self.z_log_var_layer = nn.Linear(latent_dim, latent_dim)

        self.sampling = Sampling()
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.dense_latent.weight)
        nn.init.xavier_uniform_(self.z_mean_layer.weight)
        nn.init.xavier_uniform_(self.z_log_var_layer.weight)

    def forward(
        self, dyn_inp: torch.Tensor, inp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(inp)
        # Average forward and backward directions (merge_mode="ave")
        h = (h[:, :, : inp.shape[-1]] + h[:, :, inp.shape[-1] :]) / 2.0

        h = self.batch_norm(h)

        enc = self.vmha(h, dyn_inp)
        enc = self.dropout(enc)

        enc = torch.relu(self.dense_latent(enc))
        enc = self.batch_norm_latent(enc)

        z_mean = self.z_mean_layer(enc)
        z_log_var = self.z_log_var_layer(enc)

        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    """Bi-LSTM decoder that reconstructs time series from latent + condition."""

    def __init__(
        self,
        window_size: int,
        n_main_features: int,
        n_dyn_features: int,
        latent_dim: int,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=latent_dim + n_dyn_features,
            hidden_size=n_main_features,
            batch_first=True,
            bidirectional=True,
        )
        self.output_layer = nn.Linear(n_main_features, n_main_features)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(
        self, z: torch.Tensor, dyn_inp: torch.Tensor
    ) -> torch.Tensor:
        dec = torch.cat([z, dyn_inp], dim=-1)

        dec, _ = self.lstm(dec)
        # Average forward and backward directions (merge_mode="ave")
        hidden_size = dec.shape[-1] // 2
        dec = (dec[:, :, :hidden_size] + dec[:, :, hidden_size:]) / 2.0

        return self.output_layer(dec)


class CVAE(nn.Module):
    """Conditional Variational Autoencoder combining encoder and decoder."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        window_size: int,
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.window_size = window_size
        self.kl_weight = kl_weight

    def forward(
        self, dynamic_features: torch.Tensor, inp_data: torch.Tensor
    ) -> torch.Tensor:
        z_mean, z_log_var, z = self.encoder(dynamic_features, inp_data)
        return self.decoder(z, dynamic_features)

    def compute_loss(
        self, dynamic_features: torch.Tensor, inp_data: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        z_mean, z_log_var, z = self.encoder(dynamic_features, inp_data)
        pred = self.decoder(z, dynamic_features)

        reconstruction_loss = torch.mean((inp_data - pred) ** 2)
        kl_loss = -0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        kl_loss = kl_loss.sum(dim=[1, 2]).mean()
        total_loss = reconstruction_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


def get_CVAE(
    window_size: int,
    n_main_features: int,
    n_dyn_features: int,
    latent_dim: int,
    num_heads: int = 8,
    ff_dim: int = 256,
) -> tuple[Encoder, Decoder]:
    encoder = Encoder(
        window_size=window_size,
        n_main_features=n_main_features,
        n_dyn_features=n_dyn_features,
        latent_dim=latent_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
    )
    decoder = Decoder(
        window_size=window_size,
        n_main_features=n_main_features,
        n_dyn_features=n_dyn_features,
        latent_dim=latent_dim,
    )
    return encoder, decoder
