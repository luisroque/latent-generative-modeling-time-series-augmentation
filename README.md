# L-GTA: Latent Generative Temporal Augmentation

Data augmentation is becoming increasingly important across time series forecasting, classification, and anomaly detection. We introduce **Latent Generative Temporal Augmentation (L-GTA)**, a generative approach based on a Conditional Variational Autoencoder with a Bi-LSTM backbone and temporal self-attention. The model learns a latent representation per timestep and applies controlled perturbations (jittering, magnitude warping, drift, etc.). An equivariance objective encourages consistency between latent- and data-space transformations, so that augmented samples show predictable, interpretable transformation signatures. We evaluate L-GTA on real-world datasets against SOTA generative methods (TimeGAN, TimeVAE, Diffusion-TS) and direct transformation baselines. Across downstream forecasting, distribution fidelity, and controllability of transformation intensity, L-GTA consistently outperforms competing approaches—reducing prediction error by up to 26% vs the strongest generative baseline and 27% relative to using the original data without augmentation.

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
```

## Features

- **L-GTA**: CVAE with Bi-LSTM encoder, temporal self-attention, optional dynamic time features; per-timestep latent code, equivariance objective, latent-space transformations.
- **Experiments**: Downstream forecasting (vs TimeGAN, TimeVAE, Diffusion-TS, direct), component ablation (signatures, controllability), synthesis quality.

## Run

```bash
bash scripts/run_all_experiments.sh
# or: --datasets tourism wiki2
```

Individual modules:

```bash
python -m lgta.experiments.downstream_forecasting --output-dir assets/results/downstream_forecasting [--dataset tourism] [--all-datasets] [--method lgta]
python -m lgta.experiments.component_ablation --output-dir assets/results/component_ablation [--dataset tourism] [--all-datasets]
python -m lgta.experiments.synthesis_quality --output-dir assets/results/synthesis_quality [--dataset tourism] [--all-datasets]
```

## Code

```python
from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE

cvae = CreateTransformedVersionsCVAE(dataset_name="tourism", freq="M", use_dynamic_features=True)
model, _, _ = cvae.fit()
# cvae.generate_transformed_time_series(...) for latent-space transforms
```

## Tests

```bash
python -m pytest lgta/tests -v
```

## Results

The study is organized around four research questions:

- **Q1:** Do latent-space manipulations in L-GTA produce synthetic series with controllable and interpretable transformation signatures?
- **Q2:** How does L-GTA compare with direct augmentation and SOTA methods in downstream forecasting utility and computational efficiency?
- **Q3:** How does L-GTA synthetic data compare with direct and SOTA augmentation in terms of fidelity and diversity?
- **Q4:** Which components of L-GTA contribute most to controllability and reconstruction quality?

Evaluation covers downstream forecasting, distribution fidelity (e.g. Wasserstein, reconstruction), and controllability. L-GTA reduces prediction error by up to 26% vs the strongest generative baseline and 27% vs original data without augmentation. Figures and tables under `assets/results/`.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
