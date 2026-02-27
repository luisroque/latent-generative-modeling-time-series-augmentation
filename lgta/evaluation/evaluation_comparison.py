"""
Evaluation utilities for comparing original, L-GTA synthetic, and benchmark
datasets using dimensionality reduction, KL divergence, TSTR (Train Synthetic
Test Real), and residual analysis.
"""

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import seaborn as sns
from scipy.stats import iqr, skew, kurtosis
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import f
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


###########################
# PCA and TSNE
###########################


def calculate_reconstruction_error(X, n_components):
    """Calculates the reconstruction error for PCA reduced and reconstructed data."""
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    return mean_squared_error(X, X_reconstructed)


def perform_dimensionality_reduction(X_orig, X_synth, n_components=2):
    """
    Performs PCA and t-SNE dimensionality reduction on the original and synthetic datasets.

    Parameters:
    - X_orig: numpy.ndarray, original dataset.
    - X_synth: numpy.ndarray, synthetic dataset.
    - n_components: int, number of dimensions for PCA and t-SNE.

    Returns:
    - pca_real: DataFrame, PCA results for the original dataset.
    - pca_synth: DataFrame, PCA results for the synthetic dataset.
    - tsne_results: DataFrame, t-SNE results combining both datasets.
    """
    scaler = StandardScaler()

    X_orig_norm = scaler.fit_transform(X_orig)
    X_synth_norm = scaler.transform(X_synth)

    pca = PCA(n_components=n_components)
    tsne = TSNE(n_components=n_components)

    pca_real = pd.DataFrame(
        pca.fit_transform(X_orig_norm),
        columns=[f"PCA{i + 1}" for i in range(n_components)],
    )
    pca_synth = pd.DataFrame(
        pca.transform(X_synth_norm),
        columns=[f"PCA{i + 1}" for i in range(n_components)],
    )

    combined_data = np.vstack((pca_real, pca_synth))
    tsne_results = pd.DataFrame(
        tsne.fit_transform(combined_data),
        columns=[f"t-SNE{i + 1}" for i in range(n_components)],
    )

    tsne_real = tsne_results.iloc[: len(X_orig)]
    tsne_synth = tsne_results.iloc[len(X_orig) :]

    return pca_real, pca_synth, tsne_real, tsne_synth


def plot_dimensionality_reduction(
    pca_real,
    pca_synth,
    tsne_real,
    tsne_synth,
    title="Validating synthetic vs real data diversity and distributions",
):
    """
    Plots the PCA and t-SNE results for original vs. synthetic datasets.

    Parameters:
    - pca_real: DataFrame, PCA results for the original dataset.
    - pca_synth: DataFrame, PCA results for the synthetic dataset.
    - tsne_real: DataFrame, t-SNE results for the original dataset.
    - tsne_synth: DataFrame, t-SNE results for the synthetic dataset.
    - title: str, title for the overall figure.
    """
    sns.set_style("whitegrid")
    fig = plt.figure(constrained_layout=True, figsize=(30, 15))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    ax = fig.add_subplot(spec[0, 0])
    ax.scatter(
        pca_real.iloc[:, 0], pca_real.iloc[:, 1], c="black", alpha=0.2, label="Original"
    )
    ax.scatter(
        pca_synth.iloc[:, 0],
        pca_synth.iloc[:, 1],
        c="red",
        alpha=0.2,
        label="Synthetic",
    )
    ax.set_title("PCA Results", fontsize=20, color="red", pad=10)
    ax.legend()

    ax2 = fig.add_subplot(spec[0, 1])
    ax2.scatter(
        tsne_real.iloc[:, 0],
        tsne_real.iloc[:, 1],
        c="black",
        alpha=0.2,
        label="Original",
    )
    ax2.scatter(
        tsne_synth.iloc[:, 0],
        tsne_synth.iloc[:, 1],
        c="red",
        alpha=0.2,
        label="Synthetic",
    )
    ax2.set_title("t-SNE Results", fontsize=20, color="red", pad=10)
    ax2.legend()

    fig.suptitle(title, fontsize=24, color="grey")
    plt.show()


def kl_divergence(X_orig, X_synth, n_components=50, x_min=0, x_max=1):
    def kl_divergence_continuous(p, q, x_min, x_max):
        """
        Compute the KL divergence between two continuous probability distributions
        estimated using KDE.
        """
        kl_div = quad(lambda x: p(x) * np.log(p(x) / q(x)), x_min, x_max)[0]
        return kl_div

    kl_divergences = []

    scaler = StandardScaler()
    X_orig_norm = scaler.fit_transform(X_orig)
    X_synth_norm = scaler.transform(X_synth)

    pca = PCA(n_components=n_components)
    X_orig_reduced = pca.fit_transform(X_orig_norm)
    X_synth_reduced = pca.transform(X_synth_norm)

    for i in range(n_components):
        p_pdf = gaussian_kde(X_orig_reduced[:, i])
        q_pdf = gaussian_kde(X_synth_reduced[:, i])

        kl_div = kl_divergence_continuous(
            p_pdf.pdf, q_pdf.pdf, x_min=x_min, x_max=x_max
        )
        kl_divergences.append(kl_div)

    kl_median = (np.median(kl_divergences),)
    kl_iqr = (iqr(kl_divergences),)
    kl_skewness = (skew(kl_divergences),)
    kl_kurtosis = kurtosis(kl_divergences)

    return kl_median, kl_iqr, kl_skewness, kl_kurtosis


###################################
# Train Synthetic, Test Real (TSTR)
###################################


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_data(data):
    """Applies normalization to the dataset."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data)
    return normalized_data


def reshape_for_sequential_model(X_processed, n_features, seq_len=20):
    """Reshapes the processed dataset for use with sequential models."""
    total_sequences = X_processed.shape[0]
    n_samples = (total_sequences // seq_len) * seq_len
    new_total_size = n_samples * n_features
    X_reshaped = X_processed.reshape(-1)[:new_total_size].reshape(
        -1, seq_len, n_features
    )
    return X_reshaped


class RNNRegression(nn.Module):
    """GRU-based regression model for TSTR evaluation."""

    def __init__(self, input_size: int, units: int, n_outputs: int):
        super().__init__()
        self.gru1 = nn.GRU(
            input_size=input_size,
            hidden_size=units * 4,
            batch_first=True,
        )
        self.bn1 = nn.BatchNorm1d(units * 4)
        self.drop1 = nn.Dropout(0.3)

        self.gru2 = nn.GRU(
            input_size=units * 4,
            hidden_size=units * 4,
            batch_first=True,
        )
        self.bn2 = nn.BatchNorm1d(units * 4)
        self.drop2 = nn.Dropout(0.3)

        self.output_layer = nn.Linear(units * 4, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru1(x)
        out = self.bn1(out.transpose(1, 2)).transpose(1, 2)
        out = self.drop1(out)

        out, _ = self.gru2(out)
        out = out[:, -1, :]
        out = self.bn2(out)
        out = self.drop2(out)

        return self.output_layer(out)


def step_decay(epoch: int, initial_lr: float = 0.001) -> float:
    drop = 0.5
    epochs_drop = 10.0
    return initial_lr * (drop ** ((1 + epoch) // epochs_drop))


def train_and_evaluate_model(X_train, y_train, X_test, y_test, n_outputs, units=12):
    """Trains an RNN model and evaluates it on the test set."""
    device = _get_device()

    input_size = X_train.shape[2]
    model = RNNRegression(input_size, units, n_outputs).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.01
    )
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)

    val_split = int(0.8 * len(train_ds))
    val_x = torch.tensor(X_train[val_split:], dtype=torch.float32, device=device)
    val_y = torch.tensor(y_train[val_split:], dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(200):
        lr = step_decay(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = criterion(val_pred, val_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    model.eval()
    with torch.no_grad():
        x_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        predictions = model(x_test_t).cpu().numpy()

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
    }
    return metrics


def preprocess_train_evaluate(
    X_orig, X_hat_transf, X_benchmark, n_features, seq_len=20, units=12, n_outputs=1
):
    """
    Processes the original, synthetic, and benchmark datasets, then trains and evaluates models on them.
    Assumes the last value of each sequence is the target for forecasting.

    Returns:
    - A DataFrame with comparison metrics.
    """
    X_orig_processed = preprocess_data(X_orig)
    X_hat_transf_processed = preprocess_data(X_hat_transf)
    X_benchmark_processed = preprocess_data(X_benchmark)

    def reshape_and_split(data, seq_len=seq_len, n_features=n_features):
        n_samples = data.shape[0] // (seq_len + n_outputs)
        reshaped_data = data[: n_samples * (seq_len + n_outputs)].reshape(
            n_samples, seq_len + n_outputs, n_features
        )
        X = reshaped_data[:, :seq_len, :]
        y = reshaped_data[:, seq_len : seq_len + n_outputs, :].reshape(
            n_samples, -1
        )
        return X, y

    X_train_orig, y_train_orig = reshape_and_split(X_orig_processed)
    X_train_synth, y_train_synth = reshape_and_split(X_hat_transf_processed)
    X_train_benchmark, y_train_benchmark = reshape_and_split(X_benchmark_processed)

    n_events = X_train_orig.shape[0]
    train_idx, test_idx = np.split(np.arange(n_events), [int(0.75 * n_events)])

    results = {}
    datasets = {
        "Real": (
            X_train_orig[train_idx],
            y_train_orig[train_idx],
            X_train_orig[test_idx],
            y_train_orig[test_idx],
        ),
        "L-GTA": (
            X_train_synth[train_idx],
            y_train_synth[train_idx],
            X_train_orig[test_idx],
            y_train_orig[test_idx],
        ),
        "Benchmark": (
            X_train_benchmark[train_idx],
            y_train_benchmark[train_idx],
            X_train_orig[test_idx],
            y_train_orig[test_idx],
        ),
    }

    for name, (X_train, y_train, X_test, y_test) in datasets.items():
        metrics = train_and_evaluate_model(
            X_train, y_train, X_test, y_test, n_outputs=n_features, units=units
        )
        results[name] = metrics

    return pd.DataFrame(results)


###################################
# Residuals
###################################


def standardize_and_calculate_residuals(*datasets):
    scaler = StandardScaler()
    standardized_datasets = [scaler.fit_transform(dataset) for dataset in datasets]
    residuals = [standardized_datasets[0] - d for d in standardized_datasets[1:]]
    return residuals


def plot_residuals(residuals_lgta, residuals_benchmark, n_series=4):
    """
    Plots the residuals of L-GTA and benchmark transformations side by side
    for the first n time series.
    """
    fig, axs = plt.subplots(
        n_series, 2, figsize=(12, n_series * 3), sharex="col", sharey="row"
    )

    for i in range(n_series):
        axs[i, 0].plot(residuals_lgta[i], label="L-GTA Residuals", color="darkblue")
        axs[i, 1].plot(
            residuals_benchmark[i], label="Benchmark Residuals", color="darkred"
        )

        axs[i, 0].set_title(f"Series {i + 1} - L-GTA")
        axs[i, 1].set_title(f"Series {i + 1} - Benchmark")

        if i == 0:
            axs[i, 0].legend()
            axs[i, 1].legend()

    for ax in axs.flat:
        ax.label_outer()

    fig.tight_layout()
    plt.show()


def analyze_transformations(residuals_lgta, residuals_benchmark):
    plot_residuals(residuals_lgta, residuals_benchmark)
    flat_residuals_lgta = residuals_lgta.flatten()
    flat_residuals_benchmark = residuals_benchmark.flatten()

    mean_lgta, std_lgta = np.mean(flat_residuals_lgta), np.std(flat_residuals_lgta)
    mean_benchmark, std_benchmark = np.mean(flat_residuals_benchmark), np.std(
        flat_residuals_benchmark
    )

    plt.figure(figsize=(10, 6))
    sns.kdeplot(flat_residuals_lgta, label="LGTA", color="skyblue")
    sns.kdeplot(flat_residuals_benchmark, label="Direct", color="lightgreen")

    plt.axvline(mean_lgta, color="blue", linestyle="--", label="Mean - LGTA")
    plt.axvline(mean_benchmark, color="green", linestyle="--", label="Mean - Direct")

    plt.fill_betweenx(
        [0, plt.ylim()[1]],
        mean_lgta - std_lgta,
        mean_lgta + std_lgta,
        color="blue",
        alpha=0.2,
        label="Std Dev - LGTA",
    )
    plt.fill_betweenx(
        [0, plt.ylim()[1]],
        mean_benchmark - std_benchmark,
        mean_benchmark + std_benchmark,
        color="green",
        alpha=0.2,
        label="Std Dev - Direct",
    )

    x_min = min(mean_lgta - 3 * std_lgta, mean_benchmark - 3 * std_benchmark)
    x_max = max(mean_lgta + 3 * std_lgta, mean_benchmark + 3 * std_benchmark)
    plt.xlim(x_min, x_max)

    plt.title("KDE of Residuals Distributions")
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    variance_lgta = np.var(residuals_lgta, axis=1)
    variance_benchmark = np.var(residuals_benchmark, axis=1)
    print(
        f"Mean Variance - LGTA: {np.mean(variance_lgta)}, Benchmark: {np.mean(variance_benchmark)}"
    )


def compare_residuals_variances_pca(
    residuals1, residuals2, n_components=0.95, alpha=0.05
):
    """
    Compares the variances of the principal components of two sets of residuals.
    """
    scaler = StandardScaler()
    residuals_all = np.concatenate((residuals1, residuals2), axis=0)
    residuals_all_scaled = scaler.fit_transform(residuals_all)

    pca = PCA(n_components=n_components)
    pca.fit(residuals_all_scaled)
    residuals1_pca = pca.transform(scaler.transform(residuals1))
    residuals2_pca = pca.transform(scaler.transform(residuals2))

    plt.figure(figsize=(10, 6))
    variances1 = []
    variances2 = []
    p_values = []

    for i in range(residuals1_pca.shape[1]):
        var1 = np.var(residuals1_pca[:, i], ddof=1)
        var2 = np.var(residuals2_pca[:, i], ddof=1)
        variances1.append(var1)
        variances2.append(var2)

        F = var1 / var2 if var1 > var2 else var2 / var1
        df1, df2 = residuals1_pca.shape[0] - 1, residuals2_pca.shape[0] - 1
        p_value = 2 * min(f.cdf(F, df1, df2), 1 - f.cdf(F, df1, df2))
        p_values.append(p_value)

        if p_value < alpha:
            plt.scatter(i, var1, color="red")
            plt.scatter(i, var2, color="blue")
        else:
            plt.scatter(i, var1, color="pink")
            plt.scatter(i, var2, color="lightblue")

    plt.plot(variances1, "r-", label="Residuals 1 Variance")
    plt.plot(variances2, "b-", label="Residuals 2 Variance")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance")
    plt.title("Comparison of Variances Across Principal Components")
    plt.legend()
    plt.show()

    significant_diffs = sum(p < alpha for p in p_values)
    if significant_diffs > 0:
        print(
            f"Found significant differences in variances for {significant_diffs} out of {len(p_values)} principal components."
        )
    else:
        print(
            "No significant differences in variances were found across principal components."
        )


def plot_residuals_gradient(residuals_lgta, residuals_benchmark, params):
    """Plots the KDE of residuals for L-GTA and Benchmark for different magnitudes."""
    flat_residuals_lgta = [residuals_lgta[i].flatten() for i in range(len(params))]
    flat_residuals_benchmark = [
        residuals_benchmark[i].flatten() for i in range(len(params))
    ]

    lgta_colors = sns.light_palette("darkorange", n_colors=len(params))
    benchmark_colors = sns.light_palette("dodgerblue", n_colors=len(params))

    plt.figure(figsize=(10, 6))

    for i, param in enumerate(params):
        sns.kdeplot(
            flat_residuals_lgta[i],
            label=f"L-GTA -> {param}",
            color=lgta_colors[i],
            linewidth=2,
        )

    for i, param in enumerate(params):
        sns.kdeplot(
            flat_residuals_benchmark[i],
            label=f"Benchmark -> {param}",
            color=benchmark_colors[i],
            linewidth=2,
        )

    plt.xlim(-3, 3)
    plt.title(
        "KDE of Residuals for the Jittering Transformation for Different Magnitudes",
        fontsize=14,
    )
    plt.xlabel("Residuals", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
