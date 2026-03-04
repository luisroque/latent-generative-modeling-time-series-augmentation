"""
Equivariance regularization for the CVAE decoder.

Generates perturbation fields that can be applied consistently to both
latent and data spaces. The equivariance loss encourages
decode(T(z)) ≈ T(decode(z)), so that transformations applied in latent
space produce the same effect as if applied in data space.

In TEMPORAL mode, fields have shape (B, W, 1) and carry temporal
structure (drift, trend). In GLOBAL mode, all four perturbation types
are available but their fields collapse to (B, 1) since the latent
code has no temporal axis -- drift and trend degenerate into generic
per-sample additive noise, which is the expected behaviour that the
ablation study should demonstrate.
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .models import LatentMode


class PerturbationType(str, Enum):
    JITTER = "jitter"
    SCALING = "scaling"
    DRIFT = "drift"
    TREND = "trend"


@dataclass
class TemporalPerturbation:
    """A perturbation field with its application mode."""

    field: torch.Tensor
    is_multiplicative: bool


def _sample_temporal(
    ptype: PerturbationType,
    batch_size: int,
    window_size: int,
    sigma: float,
    device: torch.device,
) -> TemporalPerturbation:
    """Perturbation with per-timestep structure: (B, W, 1)."""
    if ptype == PerturbationType.JITTER:
        field = sigma * torch.randn(batch_size, window_size, 1, device=device)
        return TemporalPerturbation(field=field, is_multiplicative=False)

    if ptype == PerturbationType.SCALING:
        factor = 1.0 + sigma * torch.randn(batch_size, 1, 1, device=device)
        return TemporalPerturbation(field=factor, is_multiplicative=True)

    if ptype == PerturbationType.DRIFT:
        steps = (
            torch.randn(batch_size, window_size, 1, device=device)
            * sigma / (window_size ** 0.5)
        )
        field = torch.cumsum(steps, dim=1)
        return TemporalPerturbation(field=field, is_multiplicative=False)

    ramp = torch.linspace(-0.5, 0.5, window_size, device=device).reshape(1, -1, 1)
    slopes = torch.randn(batch_size, 1, 1, device=device)
    field = sigma * slopes * ramp
    return TemporalPerturbation(field=field, is_multiplicative=False)


def _sample_global(
    ptype: PerturbationType,
    batch_size: int,
    sigma: float,
    device: torch.device,
) -> TemporalPerturbation:
    """Perturbation collapsed to a per-sample scalar: (B, 1).

    Drift and trend degenerate into plain additive noise because the
    global latent code carries no temporal dimension for cumulative or
    linear structure to act on.
    """
    if ptype == PerturbationType.SCALING:
        factor = 1.0 + sigma * torch.randn(batch_size, 1, device=device)
        return TemporalPerturbation(field=factor, is_multiplicative=True)

    field = sigma * torch.randn(batch_size, 1, device=device)
    return TemporalPerturbation(field=field, is_multiplicative=False)


def sample_perturbation(
    batch_size: int,
    window_size: int,
    device: torch.device,
    sigma_range: tuple[float, float] = (0.1, 1.0),
    latent_mode: "LatentMode | None" = None,
) -> TemporalPerturbation:
    """Sample a random perturbation field.

    In TEMPORAL mode (default), the field has shape (B, W, 1) and can
    carry temporal structure. In GLOBAL mode, all perturbation types
    are available but their fields collapse to (B, 1).
    """
    from .models import LatentMode

    ptype = random.choice(list(PerturbationType))
    sigma = random.uniform(*sigma_range)

    if latent_mode == LatentMode.GLOBAL:
        return _sample_global(ptype, batch_size, sigma, device)
    return _sample_temporal(ptype, batch_size, window_size, sigma, device)


def _apply_perturbation(
    tensor: torch.Tensor, perturbation: TemporalPerturbation
) -> torch.Tensor:
    field = perturbation.field
    while field.dim() < tensor.dim():
        field = field.unsqueeze(1)
    if perturbation.is_multiplicative:
        return tensor * field
    return tensor + field


def compute_equivariance_loss(
    decoder: nn.Module,
    z_mean: torch.Tensor,
    dynamic_features: torch.Tensor,
    perturbation: TemporalPerturbation,
) -> torch.Tensor:
    """Compute ||decode(T(z)) - T(decode(z))||².

    Gradients flow only through the decoder (z_mean is detached).
    The target T(decode(z)) is also detached so that the loss
    exclusively shapes how the decoder maps perturbations.
    """
    z_base = z_mean.detach()

    with torch.no_grad():
        x_base = decoder(z_base, dynamic_features)
        x_target = _apply_perturbation(x_base, perturbation)

    z_perturbed = _apply_perturbation(z_base, perturbation)
    x_from_perturbed_z = decoder(z_perturbed, dynamic_features)

    return torch.mean((x_from_perturbed_z - x_target) ** 2)
