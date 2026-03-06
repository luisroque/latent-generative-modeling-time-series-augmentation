"""
Encoder-decoder Transformer with disentangled temporal representations.
The decoder decomposes time series into trend (polynomial regressor) and
seasonal (Fourier layer) components at every block.
Ported from https://github.com/Y-debug-sys/Diffusion-TS
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import nn

from lgta.benchmarks.diffusion_ts.model_utils import (
    AdaLayerNorm,
    Conv_MLP,
    GELU2,
    LearnablePositionalEncoding,
    Transpose,
)


class TrendBlock(nn.Module):
    """Polynomial regressor for the trend component."""

    def __init__(
        self, in_dim: int, out_dim: int, in_feat: int, out_feat: int, act: nn.Module
    ) -> None:
        super().__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1),
        )
        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack(
            [lin_space ** float(p + 1) for p in range(trend_poly)], dim=0
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        return trend_vals.transpose(1, 2)


class FourierLayer(nn.Module):
    """Model seasonality via inverse DFT with top-k frequency selection."""

    def __init__(self, d_model: int, low_freq: int = 1, factor: int = 1) -> None:
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq : -1]
            f = torch.fft.rfftfreq(t)[self.low_freq : -1]
        else:
            x_freq = x_freq[:, self.low_freq :]
            f = torch.fft.rfftfreq(t)[self.low_freq :]

        x_freq, index_tuple = self._topk_freq(x_freq)
        f = repeat(f, "f -> b f d", b=x_freq.size(0), d=x_freq.size(2)).to(
            x_freq.device
        )
        f = rearrange(f[index_tuple], "b f d -> b f () d").to(x_freq.device)
        return self._extrapolate(x_freq, f, t)

    def _extrapolate(
        self, x_freq: torch.Tensor, f: torch.Tensor, t: int
    ) -> torch.Tensor:
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_range = rearrange(
            torch.arange(t, dtype=torch.float), "t -> () () t ()"
        ).to(x_freq.device)

        amp = rearrange(x_freq.abs(), "b f d -> b f () d")
        phase = rearrange(x_freq.angle(), "b f d -> b f () d")
        x_time = amp * torch.cos(2 * math.pi * f * t_range + phase)
        return reduce(x_time, "b f t d -> b t d", "sum")

    def _topk_freq(
        self, x_freq: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        length = x_freq.shape[1]
        top_k = max(1, int(self.factor * math.log(length)))
        _values, indices = torch.topk(
            x_freq.abs(), top_k, dim=1, largest=True, sorted=True
        )
        mesh_a, mesh_b = torch.meshgrid(
            torch.arange(x_freq.size(0)),
            torch.arange(x_freq.size(2)),
            indexing="ij",
        )
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        # Advanced indexing on complex tensors lacks MPS backward support;
        # decompose into real/imag, index, then recombine.
        x_real = x_freq.real[index_tuple]
        x_imag = x_freq.imag[index_tuple]
        x_freq = torch.complex(x_real, x_imag)
        return x_freq, index_tuple


class FullAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()
        hs = C // self.n_head
        k = self.key(x).view(B, T, self.n_head, hs).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, hs).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        att = att.mean(dim=1, keepdim=False)
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        condition_embd: int,
        n_head: int,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()
        _, T_E, _ = encoder_output.size()
        hs = C // self.n_head
        k = self.key(encoder_output).view(B, T_E, self.n_head, hs).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, hs).transpose(1, 2)
        v = self.value(encoder_output).view(B, T_E, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        att = att.mean(dim=1, keepdim=False)
        y = self.resid_drop(self.proj(y))
        return y, att


class EncoderBlock(nn.Module):
    def __init__(
        self,
        n_embd: int = 1024,
        n_head: int = 16,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_hidden_times: int = 4,
        activate: str = "GELU",
    ) -> None:
        super().__init__()
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
            n_embd=n_embd, n_head=n_head,
            attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop,
        )
        act = nn.GELU() if activate == "GELU" else GELU2()
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        mask: torch.Tensor | None = None,
        label_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer: int = 14,
        n_embd: int = 1024,
        n_head: int = 16,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        mlp_hidden_times: int = 4,
        block_activate: str = "GELU",
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                EncoderBlock(
                    n_embd=n_embd, n_head=n_head,
                    attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop,
                    mlp_hidden_times=mlp_hidden_times, activate=block_activate,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(
        self,
        input: torch.Tensor,
        t: torch.Tensor,
        padding_masks: torch.Tensor | None = None,
        label_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = input
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        n_channel: int,
        n_feat: int,
        n_embd: int = 1024,
        n_head: int = 16,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_hidden_times: int = 4,
        activate: str = "GELU",
        condition_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn1 = FullAttention(
            n_embd=n_embd, n_head=n_head,
            attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop,
        )
        self.attn2 = CrossAttention(
            n_embd=n_embd, condition_embd=condition_dim,
            n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop,
        )
        self.ln1_1 = AdaLayerNorm(n_embd)

        act = nn.GELU() if activate == "GELU" else GELU2()
        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, act=act)
        self.seasonal = FourierLayer(d_model=n_embd)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )
        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        timestep: torch.Tensor,
        mask: torch.Tensor | None = None,
        label_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a, _att = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        a, _att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a
        x1, x2 = self.proj(x).chunk(2, dim=1)
        trend_out, season_out = self.trend(x1), self.seasonal(x2)
        x = x + self.mlp(self.ln2(x))
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend_out, season_out


class Decoder(nn.Module):
    def __init__(
        self,
        n_channel: int,
        n_feat: int,
        n_embd: int = 1024,
        n_head: int = 16,
        n_layer: int = 10,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_hidden_times: int = 4,
        block_activate: str = "GELU",
        condition_dim: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = n_embd
        self.n_feat = n_feat
        self.blocks = nn.Sequential(
            *[
                DecoderBlock(
                    n_feat=n_feat, n_channel=n_channel, n_embd=n_embd,
                    n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop,
                    mlp_hidden_times=mlp_hidden_times, activate=block_activate,
                    condition_dim=condition_dim,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        enc: torch.Tensor,
        padding_masks: torch.Tensor | None = None,
        label_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, _ = x.shape
        mean: list[torch.Tensor] = []
        season = torch.zeros((b, c, self.d_model), device=x.device)
        trend = torch.zeros((b, c, self.n_feat), device=x.device)
        for block_idx in range(len(self.blocks)):
            x, residual_mean, residual_trend, residual_season = self.blocks[
                block_idx
            ](x, enc, t, mask=padding_masks, label_emb=label_emb)
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)
        mean_cat = torch.cat(mean, dim=1)
        return x, mean_cat, trend, season


class Transformer(nn.Module):
    def __init__(
        self,
        n_feat: int,
        n_channel: int,
        n_layer_enc: int = 5,
        n_layer_dec: int = 14,
        n_embd: int = 1024,
        n_heads: int = 16,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_hidden_times: int = 4,
        block_activate: str = "GELU",
        max_len: int = 2048,
        conv_params: list[int | None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop)

        if conv_params is None or conv_params[0] is None:
            if n_feat < 32 and n_channel < 64:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 5, 2
        else:
            kernel_size, padding = conv_params

        self.combine_s = nn.Conv1d(
            n_embd, n_feat, kernel_size=kernel_size, stride=1,
            padding=padding, padding_mode="circular", bias=False,
        )
        self.combine_m = nn.Conv1d(
            n_layer_dec, 1, kernel_size=1, stride=1,
            padding=0, padding_mode="circular", bias=False,
        )

        self.encoder = Encoder(
            n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop,
            mlp_hidden_times, block_activate,
        )
        self.pos_enc = LearnablePositionalEncoding(
            n_embd, dropout=resid_pdrop, max_len=max_len,
        )
        self.decoder = Decoder(
            n_channel, n_feat, n_embd, n_heads, n_layer_dec,
            attn_pdrop, resid_pdrop, mlp_hidden_times,
            block_activate, condition_dim=n_embd,
        )
        self.pos_dec = LearnablePositionalEncoding(
            n_embd, dropout=resid_pdrop, max_len=max_len,
        )

    def forward(
        self,
        input: torch.Tensor,
        t: torch.Tensor,
        padding_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.emb(input)
        inp_enc = self.pos_enc(emb)
        enc_cond = self.encoder(inp_enc, t, padding_masks=padding_masks)

        inp_dec = self.pos_dec(emb)
        output, mean, trend, season = self.decoder(
            inp_dec, t, enc_cond, padding_masks=padding_masks,
        )

        res = self.inverse(output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        season_error = (
            self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m
        )
        trend = self.combine_m(mean) + res_m + trend
        return trend, season_error
