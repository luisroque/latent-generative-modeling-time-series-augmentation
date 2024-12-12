import matplotlib.pyplot as plt
from keras.callbacks import History
import numpy as np


def plot_loss(history_dict):
    plt.figure(figsize=(12, 8))

    # Total Loss
    plt.subplot(2, 1, 1)
    plt.plot(history_dict["loss"], label="Training Loss")
    plt.plot(history_dict["val_loss"], label="Validation Loss")
    plt.title("Total Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Reconstruction Loss
    plt.subplot(2, 1, 2)
    plt.plot(history_dict["reconstruction_loss"], label="Training Reconstruction Loss")
    plt.plot(
        history_dict["val_reconstruction_loss"], label="Validation Reconstruction Loss"
    )
    plt.title("Reconstruction Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # KL Loss
    plt.figure(figsize=(12, 4))
    plt.plot(history_dict["kl_loss"], label="Training KL Loss")
    plt.plot(history_dict["val_kl_loss"], label="Validation KL Loss")
    plt.title("KL Divergence Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


def plot_generated_vs_original(
    dec_pred_hat: np.ndarray,
    X_train_raw: np.ndarray,
    dataset_name: str,
    transf_param: float,
    model_version: str,
    transformation: str,
    n_series: int = 8,
    directory: str = ".",
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
    # n_series needs to be even
    if not n_series % 2 == 0:
        n_series -= 1
    _, ax = plt.subplots(int(n_series // 2), int(n_series // 4), figsize=(18, 10))
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
    plt.savefig(
        f"{directory}/plots/vae_{model_version}_generated_vs_original_{dataset_name}_{transformation}_{transf_param}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()
