import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from lgta.transformations.apply_transformations_benchmark import (
    apply_transformations_and_standardize,
)
from lgta.model.generate_data import generate_datasets

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

dataset = "tourism"
freq = "M"
top = None

transformations = [
    {
        "transformation": "jitter",
        "params": [0.5],
        "parameters_benchmark": {
            "jitter": 0.5,
            "scaling": 0.1,
            "magnitude_warp": 0.1,
            "time_warp": 0.05,
        },
        "version": 5,
    },
    {
        "transformation": "magnitude_warp",
        "params": [0.1],
        "parameters_benchmark": {
            "jitter": 0.375,
            "scaling": 0.1,
            "magnitude_warp": 0.1,
            "time_warp": 0.05,
        },
        "version": 4,
    },
]

create_dataset_vae = CreateTransformedVersionsCVAE(
    dataset_name=dataset, freq=freq, top=top, dynamic_feat_trig=False
)

model, _, _ = create_dataset_vae.fit()
X_hat, z, _, _ = create_dataset_vae.predict(model)

X_orig = create_dataset_vae.X_train_raw


def prepare_rnn_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


window_size = 12
X_orig_rnn, y_orig_rnn = prepare_rnn_data(X_orig, window_size)
X_combined = np.concatenate([X_orig, X_orig], axis=1)
X_combined_rnn, y_combined_rnn = prepare_rnn_data(X_combined, window_size)

X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X_orig_rnn, y_orig_rnn, test_size=0.1, random_state=SEED, shuffle=False
)
X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(
    X_combined_rnn, y_combined_rnn, test_size=0.1, random_state=SEED, shuffle=False
)


def build_rnn(input_shape, output_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(
                128,
                activation="relu",
                return_sequences=True,
                input_shape=input_shape,
            ),
            tf.keras.layers.LSTM(
                64,
                activation="relu",
                return_sequences=False,
            ),
            tf.keras.layers.Dense(output_dim),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_and_evaluate_rnn(input_shape, X_train, y_train, X_test, y_test, num_runs=5):
    results = []
    for _ in range(num_runs):
        rnn = build_rnn(input_shape, y_train.shape[1])
        rnn.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)
        y_pred = rnn.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results.append(mse)
    return np.median(results), np.std(results)


results_summary = []

mse_orig, std_orig = train_and_evaluate_rnn(
    (window_size, X_orig.shape[1]),
    X_train_orig,
    y_train_orig,
    X_test_orig,
    y_test_orig,
    num_runs=5,
)
mse_comb, std_comb = train_and_evaluate_rnn(
    (window_size, X_combined.shape[1]),
    X_train_comb,
    y_train_comb[:, : y_orig_rnn.shape[1]],
    X_test_comb,
    y_test_orig,
    num_runs=5,
)


for transformation in transformations:
    mse_comb_hat_results = []
    mse_comb_bench_results = []
    for run in range(30):
        transformed_datasets_benchmark = apply_transformations_and_standardize(
            dataset, freq, transformation["parameters_benchmark"], standardize=False
        )
        X_orig, X_hat_transf, X_benchmark = generate_datasets(
            dataset,
            freq,
            model,
            z,
            create_dataset_vae,
            X_orig,
            transformation["transformation"],
            transformation["params"],
            transformation["parameters_benchmark"],
            transformation["version"],
        )

        X_combined_hat = np.concatenate([X_orig, X_hat_transf], axis=1)
        X_combined_bench = np.concatenate([X_orig, X_benchmark], axis=1)

        X_combined_hat_rnn, y_combined_hat_rnn = prepare_rnn_data(
            X_combined_hat, window_size
        )
        X_combined_bench_rnn, y_combined_bench_rnn = prepare_rnn_data(
            X_combined_bench, window_size
        )

        X_train_comb_hat, X_test_comb_hat, y_train_comb_hat, y_test_comb_hat = (
            train_test_split(
                X_combined_hat_rnn,
                y_combined_hat_rnn,
                test_size=0.1,
                random_state=SEED,
                shuffle=False,
            )
        )
        X_train_comb_bench, X_test_comb_bench, y_train_comb_bench, y_test_comb_bench = (
            train_test_split(
                X_combined_bench_rnn,
                y_combined_bench_rnn,
                test_size=0.1,
                random_state=SEED,
                shuffle=False,
            )
        )

        mse_comb_hat, _ = train_and_evaluate_rnn(
            (window_size, X_combined_hat.shape[1]),
            X_train_comb_hat,
            y_train_comb_hat[:, : y_orig_rnn.shape[1]],
            X_test_comb_hat,
            y_test_orig,
            num_runs=1,
        )
        mse_comb_bench, _ = train_and_evaluate_rnn(
            (window_size, X_combined_bench.shape[1]),
            X_train_comb_bench,
            y_train_comb_bench[:, : y_orig_rnn.shape[1]],
            X_test_comb_bench,
            y_test_orig,
            num_runs=1,
        )

        mse_comb_hat_results.append(mse_comb_hat)
        mse_comb_bench_results.append(mse_comb_bench)

    results_summary.append(
        {
            "transformation": transformation["transformation"],
            "mse_comb_hat": np.median(mse_comb_hat_results),
            "std_comb_hat": np.std(mse_comb_hat_results),
            "mse_comb_bench": np.median(mse_comb_bench_results),
            "std_comb_bench": np.std(mse_comb_bench_results),
        }
    )

print("Final Results:")
print(f"MSE Original: {mse_orig:.4f} ± {std_orig:.4f}")
print(f"MSE Combined (original repeated): {mse_comb:.4f} ± {std_comb:.4f}")
for result in results_summary:
    print(
        f"Transformation: {result['transformation']}\n"
        f"  MSE Combined (L-GTA): {result['mse_comb_hat']:.4f} ± {result['std_comb_hat']:.4f}\n"
        f"  MSE Combined (benchmark): {result['mse_comb_bench']:.4f} ± {result['std_comb_bench']:.4f}"
    )
