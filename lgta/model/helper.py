import torch
import torch.nn as nn


class Sampling(nn.Module):
    """Reparameterization trick: samples z from N(z_mean, exp(z_log_var/2)).

    Works with any tensor shape (2D for global latent, 3D for per-timestep latent).
    """

    def forward(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon
