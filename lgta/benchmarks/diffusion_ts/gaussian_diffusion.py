"""
Gaussian diffusion process for the Diffusion-TS model.
Implements the denoising diffusion probabilistic model with Fourier-based
loss and the ability to predict x_0 directly (instead of noise).
Ported from https://github.com/Y-debug-sys/Diffusion-TS
"""

import math
from functools import partial

import torch
import torch.nn.functional as F
from einops import reduce
from torch import nn
from tqdm.auto import tqdm

from lgta.benchmarks.diffusion_ts.model_utils import default, extract, identity
from lgta.benchmarks.diffusion_ts.transformer import Transformer


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DiffusionTS(nn.Module):
    """Denoising diffusion model with an interpretable Transformer backbone."""

    def __init__(
        self,
        seq_length: int,
        feature_size: int,
        n_layer_enc: int = 3,
        n_layer_dec: int = 6,
        d_model: int | None = None,
        timesteps: int = 1000,
        sampling_timesteps: int | None = None,
        loss_type: str = "l1",
        beta_schedule: str = "cosine",
        n_heads: int = 4,
        mlp_hidden_times: int = 4,
        eta: float = 0.0,
        attn_pd: float = 0.0,
        resid_pd: float = 0.0,
        kernel_size: int | None = None,
        padding_size: int | None = None,
        use_ff: bool = True,
        reg_weight: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.eta = eta
        self.use_ff = use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)

        self.model = Transformer(
            n_feat=feature_size,
            n_channel=seq_length,
            n_layer_enc=n_layer_enc,
            n_layer_dec=n_layer_dec,
            n_heads=n_heads,
            attn_pdrop=attn_pd,
            resid_pdrop=resid_pd,
            mlp_hidden_times=mlp_hidden_times,
            max_len=seq_length,
            n_embd=d_model,
            conv_params=[kernel_size, padding_size],
            **kwargs,
        )

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.sampling_timesteps = default(sampling_timesteps, self.num_timesteps)
        assert self.sampling_timesteps <= self.num_timesteps
        self.fast_sampling = self.sampling_timesteps < self.num_timesteps

        def _register(name: str, val: torch.Tensor) -> None:
            self.register_buffer(name, val.to(torch.float32))

        _register("betas", betas)
        _register("alphas_cumprod", alphas_cumprod)
        _register("alphas_cumprod_prev", alphas_cumprod_prev)

        _register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        _register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        _register("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        _register("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        _register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        _register("posterior_variance", posterior_variance)
        _register(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        _register(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        _register(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        _register(
            "loss_weight",
            torch.sqrt(alphas)
            * torch.sqrt(1.0 - alphas_cumprod)
            / betas
            / 100,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def predict_noise_from_start(
        self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def output(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        padding_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        trend, season = self.model(x, t, padding_masks=padding_masks)
        return trend + season

    def model_predictions(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        clip_x_start: bool = False,
        padding_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if padding_masks is None:
            padding_masks = torch.ones(
                x.shape[0], self.seq_length, dtype=bool, device=x.device
            )
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )
        x_start = self.output(x, t, padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(
        self, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, x_start = self.model_predictions(x, t)
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(
        self, x: torch.Tensor, t: int, clip_denoised: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long
        )
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x) if t > 0 else 0.0
        pred_series = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_series, x_start

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img, _ = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def fast_sample(
        self, shape: tuple[int, ...], clip_denoised: bool = True
    ) -> torch.Tensor:
        batch = shape[0]
        device = self.betas.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(
                img, time_cond, clip_x_start=clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * (
                (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
            ).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img

    def generate_mts(self, batch_size: int = 16) -> torch.Tensor:
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn((batch_size, self.seq_length, self.feature_size))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        target: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        padding_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start

        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.output(x, t, padding_masks)

        train_loss = self.loss_fn(model_out, target, reduction="none")

        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm="forward")
            fft2 = torch.fft.fft(target.transpose(1, 2), norm="forward")
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(
                torch.real(fft1), torch.real(fft2), reduction="none"
            ) + self.loss_fn(
                torch.imag(fft1), torch.imag(fft2), reduction="none"
            )
            train_loss = train_loss + self.ff_weight * fourier_loss

        train_loss = reduce(train_loss, "b ... -> b (...)", "mean")
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        b, c, n = x.shape
        device = x.device
        assert n == self.feature_size, f"number of variables must be {self.feature_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x, t=t, **kwargs)
