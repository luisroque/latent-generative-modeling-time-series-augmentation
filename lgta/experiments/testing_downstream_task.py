"""
Downstream task experiment: trains an LSTM forecaster on original data,
original+augmented (L-GTA), and original+benchmark-augmented data to
compare MSE performance.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lgta.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from lgta.transformations.apply_transformations_benchmark import (
    apply_transformations_and_standardize,
)
from lgta.model.generate_data import generate_datasets

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

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
    dataset_name=dataset, freq=freq, top=top
)

model, _, _ = create_dataset_vae.fit()
X_hat, z, _, _ = create_dataset_vae.predict(model)

X_orig = create_dataset_vae.X_train_raw


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_rnn_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


class DownstreamLSTM(nn.Module):
    """Two-layer LSTM forecaster for the downstream evaluation task."""

    def __init__(self, input_size: int, output_dim: int):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size, hidden_size=128, batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=128, hidden_size=64, batch_first=True
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = torch.relu(out)
        out, _ = self.lstm2(out)
        out = torch.relu(out[:, -1, :])
        return self.fc(out)


def build_rnn(input_shape, output_dim):
    input_size = input_shape[1]
    return DownstreamLSTM(input_size, output_dim)


def train_and_evaluate_rnn(input_shape, X_train, y_train, X_test, y_test, num_runs=5):
    device = _get_device()
    results = []
    for _ in range(num_runs):
        rnn = build_rnn(input_shape, y_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        loader = DataLoader(train_ds, batch_size=32, shuffle=False)

        rnn.train()
        for _ in range(200):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(rnn(bx), by)
                loss.backward()
                optimizer.step()

        rnn.eval()
        with torch.no_grad():
            y_pred = rnn(
                torch.tensor(X_test, dtype=torch.float32, device=device)
            ).cpu().numpy()
        mse = mean_squared_error(y_test, y_pred)
        results.append(mse)
    return np.median(results), np.std(results)


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
