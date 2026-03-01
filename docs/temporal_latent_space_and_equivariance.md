# Temporal Latent Space with Equivariant Decoder Training

## Hypothesis

L-GTA applies value-space transformations (jitter, scaling, magnitude_warp, drift, trend)
in the CVAE latent code before decoding. While this produces controlled augmentations with
perfect monotonic response (Wasserstein distance scales with sigma), the **specific character**
of each transformation is lost through decoding. Jitter residuals look identical to drift
residuals after the decoder processes them.

**Root cause**: the original encoder mean-pools across the temporal dimension
(`TransformerPooling.forward` returned `x.mean(dim=1)`), collapsing each window into a single
global vector. This destroys per-timestep identity in the latent space. Combined with the
decoder's BiLSTM non-linearity, any transformation applied to the collapsed `z` produces
a generic perturbation pattern regardless of the transformation type.

**Proposed solution** (two parts):

1. **Temporal Latent Space** — restructure the encoder to preserve per-timestep latent codes
   (`z` shape changes from `(B, latent_dim)` to `(B, W, latent_dim)`) so that
   transformations can be applied along the true time axis.

2. **Equivariant Decoder Training** — add a regularization term during training that
   encourages `decode(T(z)) ≈ T(decode(z))`, teaching the decoder to faithfully transmit
   the character of latent-space perturbations into data space.

## Implementation

### 1. Temporal Latent Space (architecture changes)

**`lgta/model/models.py`**

| Component | Before | After |
|---|---|---|
| `TransformerPooling` | Self-attention → `x.mean(dim=1)` → global vector | Renamed to `TemporalSelfAttention`; returns full `(B, W, embed_dim)` |
| `Encoder.forward` | Projects pooled vector to `(B, latent_dim)` | Projects each timestep independently → `(B, W, latent_dim)` |
| `SimpleEncoder.forward` | `h.mean(dim=1)` → global vector | Keeps `(B, W, n_main)` through projection |
| `Decoder.forward` | `z.unsqueeze(1).expand(...)` to broadcast global z | Receives `(B, W, latent_dim)` directly; no broadcasting |
| `Decoder.z_proj` (skip) | `z_proj(z).unsqueeze(1).expand(...)` | `z_proj(z)` applied per-timestep |
| `CVAE.compute_loss` (KL) | `sum(dim=1)` over latent_dim | `sum(dim=-1)` to handle `(B, W, latent_dim)` |

**`lgta/model/generate_data.py`**

The generation pipeline now:
1. Detemporalizes `z_mean` from `(n_windows, W, latent_dim)` to `(n_timesteps, latent_dim)`
2. Applies transformations along the true time axis
3. Re-temporalizes back to `(n_windows, W, latent_dim)`
4. Decodes and detemporalizes the output

### 2. Equivariant Decoder Training (new regularization)

**`lgta/model/equivariance.py`** (new file)

Generates temporal perturbation fields of shape `(B, W, 1)` that broadcast to any feature
dimension. Four perturbation types match the transformation registry:

| Type | Field generation | Mode |
|---|---|---|
| Jitter | `σ · N(0,1)` i.i.d. per timestep | Additive |
| Scaling | `1 + σ · N(0,1)` single scalar | Multiplicative |
| Drift | `cumsum(σ/√W · N(0,1))` | Additive |
| Trend | `σ · slopes · linspace(-0.5, 0.5)` | Additive |

The equivariance loss:
```
L_equiv = ||decode(T(z)) − T(decode(z))||²
```

- `z_mean` is detached from the encoder (only decoder gets gradients)
- `T(decode(z))` is computed with `no_grad` (treated as a fixed target)
- A random perturbation type and sigma are sampled each training step
- Added to the total loss: `L = L_recon + β_kl · L_kl + β_eq · L_equiv`

### 3. Transformation Changes

- **Removed**: `time_warp` (incompatible with latent-space augmentation since it affects
  temporal distances)
- **Added**: `drift` (cumulative random walk) and `trend` (linear ramp injection)
- Both are value-space transformations that have distinct, measurable statistical signatures

## Results

### Monotonic Response — Preserved ✓

All 5 transformations maintain Spearman ρ = 1.0000 across all configurations
(with and without equivariance training).

### Transformation Signature Differentiation

Comparison of L-GTA fingerprint metrics **before** and **after** equivariant training
(`equiv_weight=1.0`), against Direct augmentation as ground truth:

**Autocorrelation of residuals** (high = temporally correlated perturbation):

| Transform | Before (L-GTA) | After (L-GTA) | Direct |
|---|---|---|---|
| jitter | −0.24 | **0.08** | −0.03 |
| scaling | −0.25 | **0.08** | −0.04 |
| magnitude_warp | −0.19 | −0.22 | −0.04 |
| drift | −0.23 | **0.31** | 0.87 |
| trend | −0.23 | **0.32** | 1.00 |

**Linearity (R²) of residuals** (high = linear trend in perturbation):

| Transform | Before (L-GTA) | After (L-GTA) | Direct |
|---|---|---|---|
| jitter | 0.01 | **0.01** | 0.03 |
| scaling | 0.01 | **0.04** | 0.04 |
| magnitude_warp | 0.02 | 0.01 | 0.02 |
| drift | 0.01 | **0.30** | 0.42 |
| trend | 0.02 | **0.43** | 1.00 |

**Key finding**: before equivariance, all L-GTA values were homogeneous (autocorrelation
clustered at ~−0.23, linearity at ~0.01). After equivariance, the values are clearly
differentiated — drift/trend show high autocorrelation and linearity, matching the
direction of the Direct baseline.

### Reconstruction Quality — Tradeoff

| Metric | Without Equiv | With Equiv (weight=1.0) |
|---|---|---|
| Final training loss | 0.000551 | 0.034525 |
| Reconstruction MSE (data space) | 36.4 | 472.0 |
| Training epochs | 541 | 306 |

The equivariance constraint competes with reconstruction. The weight is tunable —
lower values (e.g., 0.1–0.5) recover reconstruction quality while retaining partial
signature differentiation.

## Configuration

The equivariance weight is configured through the training pipeline:

```python
model, _, _ = vae_creator.fit(
    equiv_weight=1.0,  # 0.0 disables equivariance (default)
    ...
)
```

## Files Changed

| File | Change |
|---|---|
| `lgta/model/models.py` | Temporal latent architecture + equivariance integration |
| `lgta/model/equivariance.py` | New — perturbation sampling and equivariance loss |
| `lgta/model/generate_data.py` | Temporal latent generation pipeline |
| `lgta/model/create_dataset_versions_vae.py` | `equiv_weight` parameter in training loop |
| `lgta/postprocessing/generative_helper.py` | Temporal latent handling for generation |
| `lgta/transformations/manipulate_data.py` | Removed `time_warp`, added `drift` and `trend` |
| `lgta/transformations/create_dataset_versions.py` | Updated transformation list |
| `lgta/experiments/monotonic_response_experiment.py` | Signature analysis + equiv_weight config |
| `lgta/experiments/transformation_signatures.py` | New — fingerprint computation and plotting |
| `lgta/tests/test_manipulate_data.py` | Tests for `drift` and `trend` |
| `lgta/tests/test_create_transformed_dataset_files.py` | Updated for new transformations |
| `lgta/utils/helper.py` | Updated transformation index mapping |
