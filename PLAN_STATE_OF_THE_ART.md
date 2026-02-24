# Elevating L-GTA to State-of-the-Art

A plan to elevate L-GTA from its current state (comparison against weak direct-transformation baselines on 3 datasets) to a state-of-the-art paper with strong baselines, rigorous evaluation, ablation studies, and a compelling controllability narrative.

---

## Diagnosis: Current Weaknesses

After reviewing the entire codebase, here are the critical gaps preventing this work from being truly compelling:

1. **Weak baselines** -- L-GTA is only compared against direct transformations (jitter, scaling, etc. applied to raw data). No comparison against generative baselines like TimeGAN, Diffusion-TS, or score-based models. TimeGAN code exists in `lgta/benchmarks/timegan.py` but is unused in evaluation.
2. **Limited datasets** -- Only 3 real-world datasets (tourism/M/304 series, M5/W/500 series, police/D/500 series). Missing standard benchmarks that reviewers expect.
3. **No ablation study** -- No evidence for the contribution of VMHA, Bi-LSTM, or the latent-space-augmentation approach individually.
4. **Underdeveloped controllability story** -- The unique selling point (controlled augmentation via latent manipulation) is stated but not demonstrated quantitatively. No interpolation experiments, no monotonicity analysis of transformation parameters.
5. **Simplistic downstream evaluation** -- TSTR uses a 2-layer GRU (`lgta/evaluation/evaluation_comparison.py:219`); no established forecasting models.
6. **Model training details** -- Fixed `kl_weight=1.0` in `CVAE` (`lgta/model/models.py:153`), no KL annealing schedule, no hyperparameter sensitivity analysis.

---

## Plan: Six Pillars

### Pillar 1: Stronger Baselines (Highest Impact)

Add at least 2 generative baselines to the comparison:

- **TimeGAN** -- Already partially implemented in `lgta/benchmarks/timegan.py`. Integrate it into the evaluation pipeline by generating data with TimeGAN and passing it through the same `MetricsAggregator` and `EvaluationPipeline`.
- **Diffusion-TS or TSGM** -- Pick one recent diffusion-based method. The `tsgm` library (pip-installable) provides score-based generation. Add it as a second generative baseline.
- Keep direct transformations as a third "naive" baseline.

This transforms the narrative from "L-GTA vs jittering" to "L-GTA vs the best generative models", which is vastly more publishable.

### Pillar 2: More Datasets (Forecasting Benchmarks)

Add datasets from established forecasting benchmarks:

- **ETTh1** (Electricity Transformer Temperature, hourly, 7 features, ~17k timesteps) -- the standard long-horizon forecasting benchmark. Available via HuggingFace or direct CSV download.
- **Electricity** (370 clients, hourly consumption) -- widely used in forecasting literature.
- **Weather** (21 meteorological features, 10-minute intervals) -- tests generalization to high-frequency multivariate data.

This gives 5-6 forecasting datasets across different frequencies and domains (tourism/retail/crime/energy/weather), all evaluated on the same downstream task: forecasting.

Implementation: update `PreprocessDatasets` in `lgta/preprocessing/pre_processing_datasets.py` with new dataset loaders.

### Pillar 3: Ablation Study

Design 4 ablated variants to isolate each component's contribution:

| Variant                      | Encoder        | Attention | Augmentation Space |
| ---------------------------- | -------------- | --------- | ------------------ |
| Full L-GTA                   | Bi-LSTM + VMHA | Yes       | Latent             |
| No VMHA                      | Bi-LSTM only   | No        | Latent             |
| No Bi-LSTM                   | MLP + VMHA     | Yes       | Latent             |
| Direct latent (no transform) | Bi-LSTM + VMHA | Yes       | Random sample only |

Implementation: create variants of `get_CVAE` in `lgta/model/models.py` that skip components. Run each through the same `EvaluationPipeline`.

### Pillar 4: Controllability Demonstration (Unique Differentiator)

This is the narrative backbone. The hypothesis is well-grounded in two architectural properties: (a) the decoder is a smooth Lipschitz-continuous function (Bi-LSTM + Dense), so small latent perturbations produce bounded output perturbations, and (b) the CVAE KL regularization enforces a structured latent manifold where nearby points decode to similar outputs. Together, these make monotonic response the *expected* behavior within a practical sigma range.

#### Why the hypothesis should hold

- The decoder is a smooth function. The Bi-LSTM + Dense decoder has no hard discontinuities. Neural networks are generally Lipschitz continuous, meaning small perturbations in latent space z map to bounded perturbations in output space. So increasing sigma in the latent transformation should produce a roughly monotonic increase in output deviation.
- The CVAE objective explicitly encourages a structured latent space. The KL term regularizes the latent distribution toward a standard normal, creating a smooth manifold where nearby points decode to similar outputs.
- Direct augmentation lacks this guarantee. Jittering raw data can produce clipping, extreme spikes, or negative values. These artifacts make the response to sigma non-monotonic and unpredictable. L-GTA's decoder acts as a "learned constraint" that keeps outputs on the data manifold.

#### Risks to watch for

- At very high sigma values (>2.0), the perturbed z lands in out-of-distribution regions where the decoder has not been trained. Outputs will degrade. This is not a failure -- it defines the useful operating range. The paper should present this as "controllable within a practical sigma range" and show where the boundary is.
- Posterior collapse could weaken the effect. The current `kl_weight=1.0` is fixed, and if the KL term dominates, the latent space collapses and perturbations become meaningless. This is why KL annealing (Pillar 6) directly supports the controllability story.
- The response may be monotonic but not linear, which is fine. The claim should be "predictable and monotonic", not "linear".

#### Experiments

- **Monotonic response curve**: Sweep sigma from 0.05 to 2.0 in ~15 steps. At each sigma, generate N=10 augmented datasets and measure mean Wasserstein distance from original. Plot the curve for both L-GTA (latent augmentation) and direct augmentation. L-GTA should produce a smooth, monotonically increasing curve; direct augmentation should be noisier and may plateau or become erratic at high sigma due to artifacts (clipping, negative values). **This is the single most important figure in the paper.**
- **Controllability score**: Define as Spearman rank correlation between sigma and measured deviation. Report L-GTA score (expected ~0.95+) vs direct augmentation score (expected lower, ~0.7-0.85). This gives a single number to cite.
- **Composability**: Chain 2 transformations (e.g., jitter then magnitude_warp) using the existing chaining logic in `lgta/postprocessing/generative_helper.py`. Show L-GTA preserves coherence (low spectral Wasserstein distance) while direct chaining degrades more rapidly.
- **Latent space visualization**: t-SNE/PCA of latent z for original, low-sigma augmented, and high-sigma augmented. Show that augmented points spread outward from the original cluster proportionally to sigma, rather than scattering randomly.

### Pillar 5: Improved Downstream Forecasting Evaluation

Focus entirely on forecasting as the downstream task (consistent with the paper's domain: financial, retail, climate, health time series).

Strengthen the TSTR framework in `lgta/evaluation/evaluation_comparison.py`:

- **Better forecasting models**: Replace the simple 2-layer GRU (`RNN_regression` at line 219) with 2-3 models of increasing complexity:
  - **Linear baseline** (DLinear or simple MLP) -- fast, shows augmentation helps even simple models.
  - **GRU** (keep existing, properly tuned) -- current baseline.
  - **PatchTST or N-BEATS** (via neuralforecast or pytorch-forecasting) -- a strong modern forecaster that reviewers will recognize.
- **Multiple forecast horizons**: Test h=1 (one-step-ahead), h=6, and h=12 (or dataset-appropriate horizons) to show augmentation benefits persist across horizons.
- **Three training regimes** (this is the key table in the paper):
  - **TR/TR**: Train on Real, Test on Real (upper bound baseline).
  - **TSTR**: Train on Synthetic only, Test on Real (pure augmentation quality test).
  - **TRTR**: Train on Real + Synthetic, Test on Real (the practical augmentation use case -- most important for practitioners). This shows L-GTA-augmented data *improves* forecasting beyond what real data alone achieves.
- **Metrics**: Report MSE, MAE, and MASE (scale-independent) to be thorough.

### Pillar 6: Model Refinements

Small but important improvements:

- **KL annealing**: Add a beta schedule to `CVAE.train_step` in `lgta/model/models.py` that ramps `kl_weight` from 0 to 1 over training. This prevents posterior collapse and typically improves latent space quality. This directly supports the controllability story (Pillar 4).
- **Hyperparameter sensitivity**: Run a grid over `latent_dim` in {2, 4, 8, 16} and `num_heads` in {4, 8} to show robustness or find optimal settings.
- **Reproducibility**: Add random seed control throughout the pipeline to ensure reproducible results.

---

## Narrative Structure

The paper story becomes:

1. **Problem**: Time series augmentation with direct transformations is uncontrolled and introduces artifacts. Generative models (TimeGAN, diffusion) generate data but offer no mechanism to *control* the degree or nature of augmentation.
2. **Insight**: If you learn a structured latent space and apply transformations there, the decoder acts as a learned constraint that ensures outputs remain on the data manifold. Crucially, the transformation parameter sigma becomes a *knob* that predictably controls augmentation intensity.
3. **Method**: L-GTA = Bi-LSTM + VMHA CVAE with latent-space augmentation.
4. **Why it works (ablation)**: Each component contributes; removing any degrades quality.
5. **Controllability (key differentiator)**: Show the monotonic response curve and controllability score. Unlike TimeGAN/diffusion which sample randomly, L-GTA lets practitioners dial augmentation intensity to match their needs.
6. **Results**: L-GTA beats direct augmentation on fidelity metrics, matches or beats generative baselines, and uniquely offers controllability that others lack.
7. **Downstream forecasting utility**: Augmented data improves forecasting across 5-6 datasets in the TRTR regime. L-GTA augmentation provides the best or near-best improvement over the TR/TR baseline.

---

## Implementation Priority and Effort

Each pillar is ordered by impact-to-effort ratio:

| Priority | Pillar                   | Effort   | Rationale                                                        |
| -------- | ------------------------ | -------- | ---------------------------------------------------------------- |
| 1        | Baselines (Pillar 1)     | ~3-4 days | Highest impact -- reviewers always ask "why no comparison with X?" |
| 2        | Controllability (Pillar 4) | ~2-3 days | This is the unique contribution, must be demonstrated quantitatively |
| 3        | Downstream (Pillar 5)    | ~2-3 days | Strengthens claims of practical forecasting utility              |
| 4        | Ablation (Pillar 3)      | ~2 days  | Standard requirement for architecture papers                     |
| 5        | Datasets (Pillar 2)      | ~2-3 days | More datasets = more convincing                                  |
| 6        | Model refinements (Pillar 6) | ~1-2 days | Quick wins (KL annealing alone can meaningfully improve results) |
