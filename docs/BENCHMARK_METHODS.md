# Benchmark Methods for Augmentation and Forecasting

Reference table of baseline and comparison methods used in the downstream forecasting task, with publication year and implementation links.

| Method | Year | Venue | GitHub / implementation |
|--------|------|--------|-------------------------|
| **LGTA** | — | — | [This repository](https://github.com/luisroque/latent-generative-modeling-time-series-augmentation) |
| **TimeGAN** | 2019 | NeurIPS | [jsyoon0823/TimeGAN](https://github.com/jsyoon0823/TimeGAN) (original); our PyTorch reimplementation in `lgta/benchmarks/timegan.py` |
| **TimeVAE** | 2022 | ICLR | [abudesai/timeVAE](https://github.com/abudesai/timeVAE) |
| **TSDiff** | 2023 | NeurIPS | [amazon-science/unconditional-time-series-diffusion](https://github.com/amazon-science/unconditional-time-series-diffusion); adapter in `lgta/benchmarks/tsdiff.py` |
| **Benchmark** (direct transformations) | — | — | In-repo: `lgta/transformations/apply_transformations_benchmark.py` (jitter, magnitude warp, scaling, time warp) |

### TSDiff and avoiding dependency clashes

We use the **original implementation** via an adapter. Its dependencies (e.g. `torch~=1.13`, GluonTS) can clash with the main project, so TSDiff is **optional at import time**:

- **Default (main env):** Install only `requirements.txt`. The downstream forecasting experiment runs without TSDiff; other benchmarks (TimeGAN, TimeVAE, etc.) run as usual.
- **Full benchmarks including TSDiff:** Use the **lgta-tsdiff** env (minimal deps, no dtaidistance) so versions don’t clash:

```bash
conda create -n lgta-tsdiff python=3.12 -y && conda activate lgta-tsdiff
pip install -r requirements-tsdiff.txt
pip install -e .
python -m lgta.experiments.downstream_forecasting
```

`requirements-tsdiff.txt` installs [unconditional-time-series-diffusion](https://github.com/amazon-science/unconditional-time-series-diffusion) and the rest of the minimal deps needed for the experiment; the default generator list includes TSDiff only when that package is available.

## References

- **TimeGAN:** Yoon, J., Jarrett, D., & van der Schaar, M. (2019). *Time-series Generative Adversarial Networks.* NeurIPS.
- **TimeVAE:** Desai, A., Freeman, C., Wang, Z., & Beaver, I. (2022). *TimeVAE: A Variational Auto-Encoder for Multivariate Time Series Generation.* ICLR.
- **TSDiff:** Kollovieh, M., Ansari, A. F., et al. (2023). *Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting.* NeurIPS.
