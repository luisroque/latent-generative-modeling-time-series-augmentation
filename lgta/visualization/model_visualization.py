import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_loss(
    history: dict[str, list[float]],
    first_index: int,
    dataset_name: str,
    base_dir: str = ".",
    show: bool = False,
) -> None:
    """
    Plot total loss, reconstruction loss and kl_loss per epoch. Saves to file
    only; does not open interactive mode unless show=True.

    :param history: recorded loss dictionary with keys "loss", "reconstruction_loss", "kl_loss"
    :param first_index: first index of the loss arrays to plot to avoid hard to read plots
    :param dataset_name: name of the dataset to plot and store
    :param show: if True, call plt.show(); default False to only store locally
    """
    _, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(history["loss"][first_index:])
    ax.plot(history["reconstruction_loss"][first_index:])
    ax.plot(history["kl_loss"][first_index:])
    ax.set_title("model loss")
    ax.set_ylabel("loss")
    ax.set_xlabel("epoch")
    plt.legend(["total_loss", "reconstruction_loss", "kl_loss"], loc="upper left")
    base = base_dir.rstrip("/") if base_dir not in (".", "") else ""
    plots_dir = Path(f"{base}/assets/plots" if base else "assets/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        plots_dir / f"vae_loss_{dataset_name}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    if show:
        plt.show()
    else:
        plt.close()


def plot_generated_vs_original(
    dec_pred_hat: np.ndarray,
    X_train_raw: np.ndarray,
    dataset_name: str,
    transf_param: float,
    model_version: str,
    transformation: str,
    n_series: int = 8,
    directory: str = "./",
) -> None:
    """
    Plot generated series and the original series and store as pdf

    Args:
        dec_pred_hat: predictions
        X_train_raw: original series
        param_vae: the parameter used in the VAE sampling
        dataset_name: name of the generated dataset
        n_series: number of series to plot
        directory: local directory to store the file
    """
    n_series = min(n_series, dec_pred_hat.shape[1])
    if not n_series % 2 == 0:
        n_series -= 1
    n_series = max(n_series, 2)
    n_cols = min(n_series, 2)
    n_rows = max(1, n_series // n_cols)
    _, ax = plt.subplots(n_rows, n_cols, figsize=(18, 10))
    ax = ax.ravel()
    n_samples = X_train_raw.shape[0]
    for i in range(n_series):
        ax[i].plot(np.arange(n_samples), dec_pred_hat[:, i], label="new sample")
        ax[i].plot(np.arange(n_samples), X_train_raw[:, i], label="orig")
    plt.legend()
    plt.suptitle(
        f"VAE generated dataset vs original -> {dataset_name} using {transformation} with sigma={transf_param}",
        fontsize=14,
    )
    base = directory.rstrip("/") if directory not in (".", "", "./") else ""
    plots_dir = Path(f"{base}/assets/plots" if base else "assets/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        plots_dir
        / f"vae_{model_version}_generated_vs_original_{dataset_name}_{transformation}_{transf_param}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()
