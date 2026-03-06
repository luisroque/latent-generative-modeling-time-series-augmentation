"""
Utility modules for the Diffusion-TS interpretable diffusion model.
Ported from https://github.com/Y-debug-sys/Diffusion-TS
"""

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn


def exists(x: object) -> bool:
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return t


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def normalize_to_neg_one_to_one(x: torch.Tensor) -> torch.Tensor:
    return x * 2 - 1


def unnormalize_to_zero_to_one(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) * 0.5


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe
        return self.dropout(x)


class Transpose(nn.Module):
    def __init__(self, shape: tuple) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(*self.shape)


class Conv_MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, resid_pdrop: float = 0.0) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_dim, out_dim, 3, stride=1, padding=1),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x).transpose(1, 2)


class GELU2(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.sigmoid(1.702 * x)


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        label_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.emb(timestep)
        if label_emb is not None:
            emb = emb + label_emb
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class EMA(nn.Module):
    """Exponential moving average of a model's parameters. Used at sampling time."""

    def __init__(
        self,
        model: nn.Module,
        beta: float = 0.995,
        update_every: int = 10,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.update_every = update_every
        self._step = 0
        self._source_model = model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

    def update(self) -> None:
        self._step += 1
        if self._step % self.update_every != 0:
            return
        with torch.no_grad():
            for ema_p, p in zip(
                self.ema_model.parameters(),
                self._source_model.parameters(),
            ):
                ema_p.data.mul_(self.beta).add_(p.data, alpha=1.0 - self.beta)
