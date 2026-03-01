"""
Equivariance regularization for the CVAE decoder.

Generates temporal perturbation fields that can be applied consistently
to both latent and data spaces. The equivariance loss encourages
decode(T(z)) ≈ T(decode(z)), so that transformations applied in latent
space produce the same effect as if applied in data space.

Perturbation fields have shape (B, W, 1) and broadcast to any feature
dimension, ensuring the same temporal pattern affects all features
identically. Different perturbation types capture different temporal
structures (i.i.d., cumulative, linear).
"""

import random
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn


class PerturbationType(str, Enum):
    JITTER = "jitter"
    SCALING = "scaling"
    DRIFT = "drift"
    TREND = "trend"


@dataclass
class TemporalPerturbation:
    """A temporal perturbation field with its application mode."""

    field: torch.Tensor
    is_multiplicative: bool


def sample_perturbation(
    batch_size: int,
    window_size: int,
    device: torch.device,
    sigma_range: tuple[float, float] = (0.1, 1.0),
) -> TemporalPerturbation:
    """Sample a random temporal perturbation field of shape (B, W, 1).

    The field broadcasts across the feature dimension so the same
    temporal pattern is applied to all latent dims / data dims.
    """
    ptype = random.choice(list(PerturbationType))
    sigma = random.uniform(*sigma_range)

    if ptype == PerturbationType.JITTER:
        field = sigma * torch.randn(
            batch_size, window_size, 1, device=device
        )
        return TemporalPerturbation(field=field, is_multiplicative=False)

    if ptype == PerturbationType.SCALING:
        factor = 1.0 + sigma * torch.randn(
            batch_size, 1, 1, device=device
        )
        return TemporalPerturbation(field=factor, is_multiplicative=True)

    if ptype == PerturbationType.DRIFT:
        steps = torch.randn(
            batch_size, window_size, 1, device=device
        ) * sigma / (window_size ** 0.5)
        field = torch.cumsum(steps, dim=1)
        return TemporalPerturbation(field=field, is_multiplicative=False)

    ramp = torch.linspace(
        -0.5, 0.5, window_size, device=device
    ).reshape(1, -1, 1)
    slopes = torch.randn(batch_size, 1, 1, device=device)
    field = sigma * slopes * ramp
    return TemporalPerturbation(field=field, is_multiplicative=False)


def _apply_perturbation(
    tensor: torch.Tensor, perturbation: TemporalPerturbation
) -> torch.Tensor:
    if perturbation.is_multiplicative:
        return tensor * perturbation.field
    return tensor + perturbation.field


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
