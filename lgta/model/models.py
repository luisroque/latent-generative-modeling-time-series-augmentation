"""
L-GTA model architecture: Conditional Variational Autoencoder (CVAE) with
a per-timestep temporal latent code.

Encoder: concat(x, c) -> BiLSTM -> LN -> SelfAttn+PosEmbed (no pooling)
         -> Dense(relu) -> LN -> (mu, log_var) -> z   [shape: (B, W, d)]
Decoder: concat(z, c) -> BiLSTM -> z_skip -> FC -> x_hat

The encoder produces a per-timestep latent code z of shape (B, W, latent_dim)
so that temporal transformations applied in latent space preserve their
character through decoding. Self-attention provides global context across
timesteps while retaining each timestep's individual latent identity.
"""

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from .helper import Sampling
from .equivariance import sample_perturbation, compute_equivariance_loss


class EncoderType(str, Enum):
    FULL = "full"
    SIMPLE = "simple"


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


class TemporalSelfAttention(nn.Module):
    """Self-attention with positional encoding that preserves per-timestep outputs.

    Each timestep attends to all others for global context but retains its
    own representation, enabling the latent space to carry temporal structure.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        max_seq_len: int,
        rate: float = 0.3,
    ):
        super().__init__()
        self.pos_embed = PositionalEmbedding(max_seq_len, embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.drop1 = nn.Dropout(rate)
        self.drop2 = nn.Dropout(rate)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_embed(x)
        attn_out, _ = self.self_attn(x, x, x)
        attn_out = self.drop1(attn_out)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        ffn_out = self.drop2(ffn_out)
        x = self.ln2(x + ffn_out)
        return x


class Encoder(nn.Module):
    """BiLSTM encoder with self-attention producing a per-timestep latent code.

    Concatenates dynamic features with the raw input, processes through BiLSTM
    and self-attention (no temporal pooling), then projects each timestep
    independently to (z_mean, z_log_var). Output z has shape (B, W, latent_dim).
    """

    def __init__(
        self,
        window_size: int,
        n_main_features: int,
        n_dyn_features: int,
        latent_dim: int,
        num_heads: int = 8,
        ff_dim: int = 256,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.n_main_features = n_main_features

        self.lstm = nn.LSTM(
            input_size=n_main_features + n_dyn_features,
            hidden_size=n_main_features,
            batch_first=True,
            bidirectional=True,
        )
        self.ln_lstm = nn.LayerNorm(n_main_features)

        self.transformer = TemporalSelfAttention(
            embed_dim=n_main_features,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_seq_len=window_size,
            rate=dropout_rate,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.dense_latent = nn.Linear(n_main_features, latent_dim)
        self.ln_latent = nn.LayerNorm(latent_dim)

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
        x = torch.cat([dyn_inp, inp], dim=-1)
        h, _ = self.lstm(x)
        h = (h[:, :, : self.n_main_features] + h[:, :, self.n_main_features :]) / 2.0
        h = self.ln_lstm(h)

        h = self.transformer(h)
        h = self.dropout(h)

        enc = F.relu(self.dense_latent(h))
        enc = self.ln_latent(enc)

        z_mean = self.z_mean_layer(enc)
        z_log_var = self.z_log_var_layer(enc)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class SimpleEncoder(nn.Module):
    """BiLSTM encoder without self-attention producing a per-timestep latent code.

    Simpler alternative to Encoder that skips self-attention. The BiLSTM still
    provides cross-timestep context, and each timestep is projected
    independently to (z_mean, z_log_var). Output z has shape (B, W, latent_dim).
    """

    def __init__(
        self,
        window_size: int,
        n_main_features: int,
        n_dyn_features: int,
        latent_dim: int,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.n_main_features = n_main_features

        self.lstm = nn.LSTM(
            input_size=n_main_features + n_dyn_features,
            hidden_size=n_main_features,
            batch_first=True,
            bidirectional=True,
        )
        self.ln_lstm = nn.LayerNorm(n_main_features)

        self.dropout = nn.Dropout(dropout_rate)
        self.dense_latent = nn.Linear(n_main_features, latent_dim)
        self.ln_latent = nn.LayerNorm(latent_dim)

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
        x = torch.cat([dyn_inp, inp], dim=-1)
        h, _ = self.lstm(x)
        h = (h[:, :, : self.n_main_features] + h[:, :, self.n_main_features :]) / 2.0
        h = self.ln_lstm(h)

        h = self.dropout(h)

        enc = F.relu(self.dense_latent(h))
        enc = self.ln_latent(enc)

        z_mean = self.z_mean_layer(enc)
        z_log_var = self.z_log_var_layer(enc)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    """BiLSTM decoder that reconstructs temporal output from a temporal latent code.

    Receives z of shape (B, W, latent_dim) with per-timestep latent values,
    concatenates with dynamic features, processes through BiLSTM, and applies
    a dense projection to reconstruct the window. A linear skip connection
    (z_proj) projects z per-timestep into the LSTM hidden-state space,
    guaranteeing a direct path from latent perturbations to the output.
    """

    def __init__(
        self,
        window_size: int,
        n_main_features: int,
        n_dyn_features: int,
        latent_dim: int,
        dropout_rate: float = 0.3,
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.n_main_features = n_main_features

        self.lstm = nn.LSTM(
            input_size=latent_dim + n_dyn_features,
            hidden_size=n_main_features,
            batch_first=True,
            bidirectional=True,
        )

        self.z_proj = nn.Linear(latent_dim, n_main_features)

        flat_dim = window_size * n_main_features
        fc = nn.Linear(flat_dim, flat_dim)
        self.fc = nn.utils.spectral_norm(fc) if spectral_norm else fc
        self._spectral_norm = spectral_norm
        self._init_weights()

    def _init_weights(self) -> None:
        if not self._spectral_norm:
            nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.z_proj.weight, gain=0.1)
        nn.init.zeros_(self.z_proj.bias)

    def forward(
        self, z: torch.Tensor, dyn_inp: torch.Tensor
    ) -> torch.Tensor:
        dec_inp = torch.cat([z, dyn_inp], dim=-1)

        h, _ = self.lstm(dec_inp)
        h = (h[:, :, : self.n_main_features] + h[:, :, self.n_main_features :]) / 2.0

        z_skip = self.z_proj(z)
        h = h + z_skip

        h_flat = h.reshape(h.shape[0], -1)
        out_flat = self.fc(h_flat)
        return out_flat.reshape(-1, self.window_size, self.n_main_features)


class CVAE(nn.Module):
    """Conditional Variational Autoencoder with temporal latent code and
    optional equivariance regularization for the decoder."""

    def __init__(
        self,
        encoder: Encoder | SimpleEncoder,
        decoder: Decoder,
        window_size: int,
        kl_weight: float = 1.0,
        free_bits: float = 0.0,
        equiv_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.window_size = window_size
        self.kl_weight = kl_weight
        self.free_bits = free_bits
        self.equiv_weight = equiv_weight

    def forward(
        self, dynamic_features: torch.Tensor, inp_data: torch.Tensor
    ) -> torch.Tensor:
        z_mean, z_log_var, z = self.encoder(dynamic_features, inp_data)
        return self.decoder(z_mean, dynamic_features)

    def compute_loss(
        self, dynamic_features: torch.Tensor, inp_data: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        z_mean, z_log_var, z = self.encoder(dynamic_features, inp_data)
        pred = self.decoder(z, dynamic_features)

        reconstruction_loss = torch.mean((inp_data - pred) ** 2)
        kl_per_dim = -0.5 * (
            1 + z_log_var - z_mean.pow(2) - z_log_var.exp()
        )
        if self.free_bits > 0.0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        kl_loss = torch.mean(kl_per_dim.sum(dim=-1))

        equiv_loss = torch.tensor(0.0, device=pred.device)
        if self.equiv_weight > 0.0:
            perturbation = sample_perturbation(
                batch_size=z_mean.shape[0],
                window_size=self.window_size,
                device=z_mean.device,
            )
            equiv_loss = compute_equivariance_loss(
                self.decoder, z_mean, dynamic_features, perturbation,
            )

        total_loss = (
            reconstruction_loss
            + self.kl_weight * kl_loss
            + self.equiv_weight * equiv_loss
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "equiv_loss": equiv_loss,
        }


def _valid_num_heads(embed_dim: int, preferred: int = 8) -> int:
    """Return the largest divisor of *embed_dim* that is <= *preferred*."""
    for h in range(preferred, 0, -1):
        if embed_dim % h == 0:
            return h
    return 1


def get_CVAE(
    window_size: int,
    n_main_features: int,
    n_dyn_features: int,
    latent_dim: int,
    num_heads: int = 8,
    ff_dim: int = 256,
    dropout_rate: float = 0.3,
    encoder_type: EncoderType = EncoderType.FULL,
    spectral_norm: bool = False,
) -> tuple[Encoder | SimpleEncoder, Decoder]:
    if encoder_type == EncoderType.SIMPLE:
        encoder: Encoder | SimpleEncoder = SimpleEncoder(
            window_size=window_size,
            n_main_features=n_main_features,
            n_dyn_features=n_dyn_features,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
        )
    else:
        num_heads = _valid_num_heads(n_main_features, num_heads)
        encoder = Encoder(
            window_size=window_size,
            n_main_features=n_main_features,
            n_dyn_features=n_dyn_features,
            latent_dim=latent_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
        )
    decoder = Decoder(
        window_size=window_size,
        n_main_features=n_main_features,
        n_dyn_features=n_dyn_features,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        spectral_norm=spectral_norm,
    )
    return encoder, decoder
