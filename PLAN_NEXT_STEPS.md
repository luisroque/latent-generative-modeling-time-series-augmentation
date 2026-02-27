# Next Steps: Fixing Posterior Collapse on Real Data

## Context

KL annealing, larger latent dim (8), and reconstruction-based early stopping
have been implemented and validated on the synthetic dataset, where L-GTA
achieves perfect monotonic controllability (Spearman rho = 1.0).

On the tourism dataset, training now runs fully (1000 epochs, recon 0.186 ->
0.034), but the KL term collapses to zero by ~epoch 650. The encoder's
posterior becomes identical to the prior, so the latent space carries no
information and sigma perturbations have no effect (rho = -0.2, flat curve).

## Root Cause

The decoder can reconstruct reasonably well using only the dynamic features
(conditioning signal). The optimizer finds it cheaper to collapse the posterior
to the prior (zero KL cost) than to maintain an informative latent space.

## Proposed Fixes (ordered by priority)

### 1. Free Bits (KL minimum floor)

Enforce a per-dimension minimum KL contribution so the latent space cannot
collapse entirely. In `compute_loss`:

```python
kl_per_dim = -0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)  # e.g. free_bits=0.1
kl_loss = kl_per_dim.sum(dim=[1, 2]).mean()
```

This is the most targeted fix -- it directly prevents collapse while allowing
the model to freely allocate capacity across latent dimensions.

### 2. Cyclical KL Annealing

Instead of a single linear ramp from 0 to 1, cycle the KL weight between 0
and 1 multiple times during training. Each cycle forces the encoder to
re-encode information into the latent space. Implementation:

```python
cycle = epoch % cycle_length
kl_weight = min(1.0, cycle / (cycle_length * ratio)) * kl_weight_max
```

This is complementary to free bits and can be used together.

### 3. Reduce Conditioning Capacity

The decoder currently receives 8 dynamic features (day, week, month, quarter
sin/cos encodings). If those are too predictive of the target, the latent
is ignored. Options:

- Drop some dynamic features (e.g., keep only quarter for quarterly data)
- Add dropout to the conditioning input
- Reduce the conditioning dimension via a bottleneck layer

### 4. Latent Dimension Tuning

With `latent_dim=8`, some dimensions may be unused. Try `latent_dim=16` to
give the model more room, or monitor per-dimension KL to see which dimensions
are active vs collapsed.

## Validation Protocol

For each fix, run the monotonic response experiment on both datasets:

1. **Synthetic** (sanity check): Confirm rho stays close to 1.0
2. **Tourism** (target): Check that KL remains above zero at convergence and
   that the Wasserstein curve shows monotonic response to sigma

Key metrics to track:
- Final KL loss (should be > 0, ideally > 0.01)
- Spearman rho (target: > 0.8)
- Visual quality in the series comparison grid
