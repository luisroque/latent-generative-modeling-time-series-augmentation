# Comprehensive Evaluation Framework for LGTA vs Benchmark

## Overview

This evaluation framework provides an **objective, statistically rigorous** comparison between the LGTA (Latent Generative Time-series Augmentation) approach and direct benchmark transformations.

## Key Features

### 1. **Multi-Dimensional Metrics** (5 Dimensions)

#### 🎯 **Realism** - How realistic are the augmented series?
- **Wasserstein Distance**: Measures distribution distance between original and augmented data
- **Kolmogorov-Smirnov Test**: Statistical test for distribution similarity
- **Moment Preservation**: Preservation of mean, std, skewness, kurtosis

#### 🌈 **Diversity** - Are augmented series diverse yet meaningful?
- **Latent Space Coverage**: How well augmented data covers the original data's feature space
- **Intra-Series Diversity**: Diversity among multiple augmentations of the same series

#### 🎨 **Novelty** - Is augmented data novel, not just copied? ⭐ **NEW**
- **Minimum Distance to Originals**: How far augmented series are from original series (prevents memorization)
- **Duplication Ratio**: Percentage of augmented series that are near-duplicates of originals
- **Internal Diversity**: Average pairwise distance between augmented series (prevents mode collapse)
- **Novelty Score**: Combined metric balancing distance from originals and internal diversity
- **Degenerate Solution Detection**: Automatically detects if augmentation is just copying data

**Why this matters**: Without novelty metrics, simply copying the original data would score perfectly on realism metrics but provide no value. Novelty metrics ensure augmentation adds genuine value.

#### 📊 **Statistical Consistency** - Are key time series properties preserved?
- **Autocorrelation Preservation**: Maintains temporal dependencies
- **Spectral Similarity**: Preserves frequency domain characteristics
- **Trend Preservation**: Maintains long-term trends

#### 🎨 **Smoothness/Quality** - Are transformations smooth without artifacts?
- **Temporal Smoothness**: First and second-order differences
- **Smoothness Ratios**: Comparison of smoothness between original and augmented

#### 🚀 **Downstream Utility** - Do they improve model performance?
- **TSTR (Train on Synthetic, Test on Real)**: Forecasting performance using augmented data

### 2. **Statistical Hypothesis Testing**

For each metric, we perform **three complementary statistical tests**:

- **Paired t-test** (parametric): Tests if mean differences are significant
- **Wilcoxon signed-rank test** (non-parametric): Robust to outliers
- **Bootstrap test**: Distribution-free approach with resampling

**Decision Rule**: Use majority voting across tests to determine the winner for each metric.

### 3. **Multiple Experimental Runs**

- Each evaluation is repeated **N times** (default: 10)
- Provides confidence intervals and statistical power
- Ensures results are robust and not due to random chance

### 4. **Automated Reporting**

- **JSON reports**: Complete results with all metrics
- **Visualizations**: Box plots comparing methods
- **Summary statistics**: Mean, median, std, min, max for all metrics
- **Human-readable output**: Clear winner declarations with confidence levels

## How to Use

### Basic Usage

```python
from lgta.evaluation.evaluation_pipeline import run_evaluation_pipeline, EvaluationConfig
from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE

# 1. Train your VAE model
create_dataset_vae = CreateTransformedVersionsCVAE(
    dataset_name="tourism", freq="M", top=None
)
model, _, _ = create_dataset_vae.fit()
X_hat, z, _, _ = create_dataset_vae.predict(model)
X_orig = create_dataset_vae.X_train_raw

# 2. Configure evaluation
config = EvaluationConfig(
    dataset_name="tourism",
    freq="M",
    transformation_type="jitter",
    transformation_params=[0.5],
    benchmark_params={
        "jitter": 0.5,
        "scaling": 0.1,
        "magnitude_warp": 0.1,
        "time_warp": 0.05,
    },
    benchmark_version=5,
    n_repetitions=10,  # Number of runs for robustness
)

# 3. Run evaluation
report = run_evaluation_pipeline(
    config=config,
    model=model,
    z=z,
    create_dataset_vae=create_dataset_vae,
    X_orig=X_orig,
)
```

### Using the Improved Testing Script

```python
# Run comprehensive evaluation for all transformations
from lgta.experiments.testing_benchmark_vs_lgta_improved import (
    run_comprehensive_evaluation
)

reports = run_comprehensive_evaluation(
    dataset="tourism",
    freq="M",
    top=None,
    n_repetitions=10,
)
```

### Comparing Across Magnitudes

```python
from lgta.experiments.testing_benchmark_vs_lgta_improved import (
    compare_transformation_magnitudes
)

results = compare_transformation_magnitudes(
    dataset="tourism",
    transformation="jitter",
    magnitudes=[0.3, 0.5, 0.7, 0.9],
    n_repetitions=5,
)
```

## Understanding the Output

### Console Output

```
================================================================================
OVERALL WINNER: LGTA
Confidence: 80.0%
Wins: LGTA=8, Benchmark=2, Ties=0
================================================================================

KEY METRICS COMPARISON
--------------------------------------------------------------------------------

realism.wasserstein_mean:
  LGTA:      0.1234
  Benchmark: 0.1567
  Difference: -21.24%

diversity.latent_coverage:
  LGTA:      0.8921
  Benchmark: 0.8234
  Difference: +8.34%

novelty.mean_min_distance:
  LGTA:      0.4521
  Benchmark: 0.0234
  Difference: +1832.1%

novelty.duplication_ratio:
  LGTA:      0.0123
  Benchmark: 0.7845
  Difference: -98.4%

[...]

⚠️  WARNING: Benchmark appears to be memorizing/copying original data.
    Duplication ratio: 78.5%
```

### Generated Files

After running an evaluation, you'll find in `assets/results/{dataset}_{transformation}/`:

1. **`evaluation_report.json`**: Complete results including:
   - Configuration parameters
   - All metric values with statistics
   - Statistical test results
   - Category-level winners
   - Downstream performance results

2. **Visualizations** (one PNG per category):
   - `comparison_realism.png`: Realism metrics (Wasserstein, KS test, moment errors)
   - `comparison_novelty.png`: Novelty metrics (distance to originals, duplication, diversity)
   - `comparison_diversity.png`: Diversity metrics (latent coverage)
   - `comparison_consistency.png`: Consistency metrics (ACF, spectral, trend)
   - `comparison_smoothness.png`: Smoothness metrics (temporal differences)
   - `category_summary.png`: Bar chart showing winners per category

Each category plot shows:
- Box plots for each metric
- Color coding: Green = winning method, Red = losing method
- Direction indicators: ↓ (lower is better) or ↑ (higher is better)
- Mean values displayed on each plot
- Overall category winner in the title

## Interpretation Guide

### Metric Interpretation

| Metric | Lower is Better | Higher is Better | Ideal Range |
|--------|----------------|-----------------|-------------|
| **Realism** |
| Wasserstein Distance | ✅ | | 0-1 |
| KS Statistic | ✅ | | 0-0.3 |
| Moment Errors | ✅ | | 0-0.2 |
| **Diversity & Novelty** |
| Latent Coverage | | ✅ | 0.7-1.0 |
| Min Distance to Originals ⭐ | | ✅ | 0.1-1.0 |
| Duplication Ratio ⭐ | ✅ | | 0-0.1 |
| Internal Diversity ⭐ | | ✅ | 0.5-2.0 |
| Novelty Score ⭐ | | ✅ | 0.5-1.0 |
| **Consistency** |
| Autocorrelation Error | ✅ | | 0-0.3 |
| Spectral Distance | ✅ | | 0-0.5 |
| Trend Correlation | | ✅ | 0.7-1.0 |
| **Utility** |
| TSTR MSE | ✅ | | Dataset-dependent |

⭐ = New metrics that prevent memorization/copying

### Winner Determination

An overall winner is declared based on:

1. **Metric-level winners**: For each metric, the method that performs significantly better (p < 0.05) across majority of tests
2. **Overall winner**: Method that wins more metrics
3. **Confidence**: Proportion of metrics won

**Example**:
- If LGTA wins 8 metrics, Benchmark wins 2, Ties 0
- Overall Winner: **LGTA**
- Confidence: **80%** (8/10)

## Solving the Memorization Problem

### The Challenge

Without novelty metrics, simply **copying the original data** would achieve perfect scores on most metrics:
- ✅ Wasserstein Distance = 0 (perfect match!)
- ✅ Moment Preservation = perfect
- ✅ Autocorrelation = perfect
- **But this provides ZERO value for augmentation!**

### The Solution

We now balance **TWO competing objectives**:

1. **Realism**: Generated data should look like real data
2. **Novelty**: Generated data should be different enough to add value

```
Good Augmentation = High Realism + High Novelty

Bad Solutions:
- Identity Copy = High Realism + Zero Novelty ❌
- Random Noise = Zero Realism + High Novelty ❌
```

### How It Works

The evaluation now includes **automatic detection** of degenerate solutions:

```
⚠️  WARNING: Augmentation appears to be memorizing/copying original data.
    Duplication ratio: 78.5%
```

When this happens:
- Realism metrics may look "perfect"
- Novelty metrics will be terrible
- Overall assessment will favor the non-memorizing method
- User is explicitly warned

See `SOLVING_MEMORIZATION_PROBLEM.md` for detailed explanation.

## Advantages of This Approach

### 1. **Objectivity**
- No manual interpretation needed
- Statistical tests provide p-values and effect sizes
- Automated decision-making based on evidence

### 2. **Robustness**
- Multiple runs prevent cherry-picking
- Multiple statistical tests reduce false positives
- Confidence intervals quantify uncertainty

### 3. **Comprehensiveness**
- Evaluates multiple aspects (not just one metric)
- Includes both statistical properties and downstream utility
- Time-series specific metrics

### 4. **Reproducibility**
- Fully automated pipeline
- All parameters logged
- JSON output for further analysis

## Customization

### Adding New Metrics

To add a new metric:

1. Add to appropriate class in `metrics.py`:

```python
class RealismMetrics:
    @staticmethod
    def my_new_metric(X_orig: np.ndarray, X_augmented: np.ndarray) -> float:
        # Your implementation
        return metric_value
```

2. Add to `MetricsAggregator.compute_all_metrics()`:

```python
new_metric = self.realism.my_new_metric(X_orig, X_aug)
results[method]["realism"]["my_new_metric"] = float(new_metric)
```

3. Add to metrics_config in `evaluation_pipeline.py`:

```python
metrics_config = {
    "realism.my_new_metric": True,  # True if lower is better
    # ...
}
```

### Adjusting Statistical Tests

Modify the significance level:

```python
config = EvaluationConfig(
    # ...
    alpha=0.01,  # More stringent (default: 0.05)
)
```

### Understanding Novelty Metrics

If you want to check if your augmentation is memorizing:

```python
# After running evaluation
report = run_evaluation_pipeline(...)

# Check duplication ratio
lgta_dup = report['metrics_summary']['lgta']['novelty.duplication_ratio']['mean']
bench_dup = report['metrics_summary']['benchmark']['novelty.duplication_ratio']['mean']

if lgta_dup > 0.5:
    print("⚠️  LGTA is copying data (>50% near-duplicates)")
    
if bench_dup > 0.5:
    print("⚠️  Benchmark is copying data (>50% near-duplicates)")

# Check minimum distance to originals
lgta_dist = report['metrics_summary']['lgta']['novelty.mean_min_distance']['mean']

if lgta_dist < 0.05:
    print("⚠️  LGTA augmentations are too close to originals")
```

## Best Practices

1. **Use adequate repetitions**: At least 10 runs for reliable statistics
2. **Check assumptions**: Examine distributions before interpreting tests
3. **Consider practical significance**: Small p-values don't always mean practical importance
4. **Look at multiple metrics**: Don't rely on a single metric
5. **Save all results**: Keep JSON reports for later analysis

## Troubleshooting

### Issue: All metrics show "tie"
- **Cause**: Methods are too similar or sample size too small
- **Solution**: Increase n_repetitions or use larger magnitude differences

### Issue: NaN values in metrics
- **Cause**: Insufficient data or computational issues
- **Solution**: Check data quality and metric implementations

### Issue: Inconsistent results across runs
- **Cause**: High variance in augmentation process
- **Solution**: Increase n_repetitions for more stable estimates

## References

- **Wasserstein Distance**: Optimal transport theory for comparing distributions
- **KS Test**: Non-parametric test for distribution equality
- **TSTR Framework**: Train on Synthetic, Test on Real paradigm
- **Effect Sizes**: Cohen's d for parametric, rank-biserial for non-parametric

