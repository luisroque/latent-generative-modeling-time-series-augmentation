"""
Simple example demonstrating the improved evaluation pipeline.

This script shows how to:
1. Run evaluation for all transformation types (jitter, scaling, magnitude_warp, time_warp)
2. Use the same params for the model (LGTA) and benchmark for fair comparison
3. Understand the output and access results
"""

from pathlib import Path
from typing import Dict, List

from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from lgta.evaluation.evaluation_pipeline import (
    run_evaluation_pipeline,
    EvaluationConfig,
)

TRANSFORMATION_TYPES: List[str] = [
    "jitter",
    "scaling",
    "magnitude_warp",
    "time_warp",
]

# Single param per transformation; used for both LGTA and benchmark so they match.
PARAMS_BY_TRANSFORMATION: Dict[str, float] = {
    "jitter": 1.0,
    "scaling": 1.0,
    "magnitude_warp": 1.0,
    "time_warp": 1.0,
}


def _benchmark_params_for_all_transformations() -> Dict[str, float]:
    """Params for benchmark: same values as PARAMS_BY_TRANSFORMATION for each key."""
    return dict(PARAMS_BY_TRANSFORMATION)


def quick_evaluation_example():
    """
    Run evaluation for all transformation types with matching model and benchmark params.
    """
    print("\n" + "=" * 80)
    print("QUICK EVALUATION EXAMPLE — ALL TRANSFORMATIONS")
    print("=" * 80)
    print("\nRunning evaluation for each transformation with n_repetitions=3")
    print("Model and benchmark use the same param per transformation.\n")

    dataset = "tourism"
    freq = "M"
    top = None
    base_output = Path("assets/results/quick_example")

    # Step 1: Train VAE once
    print("[1/2] Training VAE model...")
    create_dataset_vae = CreateTransformedVersionsCVAE(
        dataset_name=dataset, freq=freq, top=top
    )
    model, _, _ = create_dataset_vae.fit()
    X_hat, z, _, _ = create_dataset_vae.predict(model)
    X_orig = create_dataset_vae.X_train_raw
    print(f"   ✓ Model trained. Data shape: {X_orig.shape}")

    # Step 2: Run evaluation for each transformation
    print("\n[2/2] Running evaluation for each transformation...")
    benchmark_params = _benchmark_params_for_all_transformations()
    reports: Dict[str, dict] = {}

    for transformation_type in TRANSFORMATION_TYPES:
        param = PARAMS_BY_TRANSFORMATION[transformation_type]
        config = EvaluationConfig(
            dataset_name=dataset,
            freq=freq,
            transformation_type=transformation_type,
            transformation_params=[param],
            benchmark_params=benchmark_params,
            benchmark_version=0,
            n_repetitions=3,
            output_dir=base_output / transformation_type,
        )
        print(f"\n--- {transformation_type} (param={param}) ---")
        report = run_evaluation_pipeline(
            config=config,
            model=model,
            z=z,
            create_dataset_vae=create_dataset_vae,
            X_orig=X_orig,
        )
        reports[transformation_type] = report

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY BY TRANSFORMATION")
    print("=" * 80)
    for transformation_type in TRANSFORMATION_TYPES:
        report = reports[transformation_type]
        winner = report["statistical_tests"]["overall_winner"]
        param = PARAMS_BY_TRANSFORMATION[transformation_type]
        print(f"\n{transformation_type} (param={param}): winner = {winner}")
    print("\n" + "=" * 80)
    print("✅ Example complete!")
    print("=" * 80 + "\n")

    return reports


if __name__ == "__main__":
    reports = quick_evaluation_example()
