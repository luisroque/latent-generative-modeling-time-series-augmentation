# Benchmark Methods for Augmentation and Forecasting

Reference table of baseline and comparison methods used in the downstream forecasting task, with publication year and implementation links.

| Method | Year | Venue | GitHub / implementation |
|--------|------|--------|-------------------------|
| **LGTA** | — | — | [This repository](https://github.com/luisroque/latent-generative-modeling-time-series-augmentation) |
| **TimeGAN** | 2019 | NeurIPS | [jsyoon0823/TimeGAN](https://github.com/jsyoon0823/TimeGAN) (original); our PyTorch reimplementation in `lgta/benchmarks/timegan.py` |
| **TimeVAE** | 2022 | ICLR | [abudesai/timeVAE](https://github.com/abudesai/timeVAE) |
| **Benchmark** (direct transformations) | — | — | In-repo: `lgta/transformations/apply_transformations_benchmark.py` (jitter, magnitude warp, scaling, time warp) |

## References

- **TimeGAN:** Yoon, J., Jarrett, D., & van der Schaar, M. (2019). *Time-series Generative Adversarial Networks.* NeurIPS.
- **TimeVAE:** Desai, A., Freeman, C., Wang, Z., & Beaver, I. (2022). *TimeVAE: A Variational Auto-Encoder for Multivariate Time Series Generation.* ICLR.
