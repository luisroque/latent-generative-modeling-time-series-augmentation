"""
Utilities for building the internal groups dict from wide-format DataFrames.
Used by the synthetic dataset generator in PreprocessDatasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_dataset_generated_vs_original(
    dec_pred_hat: np.ndarray, X_train_raw: np.ndarray
) -> None:
    _, ax = plt.subplots(4, 2, figsize=(18, 10))
    ax = ax.ravel()
    for i in range(8):
        ax[i].plot(dec_pred_hat[:, i], label="new sample")
        ax[i].plot(X_train_raw[:, i], label="orig")
    plt.legend()
    plt.show()


class DataTransform:
    def __init__(self, groups: dict) -> None:
        self.g = groups
        self.mu_data = np.mean(self.g["train"]["data"], axis=0)
        self.std_data = np.std(self.g["train"]["data"], axis=0)

    def std_transf_train(self) -> dict:
        self.g["train"]["data"] = (
            self.g["train"]["data"] - self.mu_data
        ) / self.std_data
        return self.g

    def inv_transf_train(self) -> dict:
        self.g["train"]["data"] = (
            self.g["train"]["data"] * self.std_data
        ) + self.mu_data
        return self.g

    def inv_transf_train_general(self, pred: np.ndarray) -> np.ndarray:
        pred_samples = pred.shape[0]
        pred = (
            (pred.reshape(-1, self.g["train"]["s"]) * self.std_data) + self.mu_data
        ).reshape(pred_samples, self.g["train"]["n"], self.g["train"]["s"])
        return pred

    def inv_transf_predict_general(self, pred: np.ndarray) -> np.ndarray:
        pred_samples = pred.shape[0]
        pred = (
            (pred.reshape(-1, self.g["predict"]["s"]) * self.std_data) + self.mu_data
        ).reshape(pred_samples, self.g["predict"]["n"], self.g["predict"]["s"])
        return pred


def generate_groups_data_flat(
    y: pd.DataFrame,
    dates: list,
    groups_input: dict,
    seasonality: int,
    h: int,
) -> dict:
    """
    Build the flat groups dict from a wide-format DataFrame with MultiIndex columns.
    Used by the synthetic dataset generator.
    """
    groups: dict = {}

    for split in ["train", "predict"]:
        groups[split] = {}
        if split == "train":
            y_ = y.iloc[:-h, :]
        else:
            y_ = y
        groups[split]["x_values"] = list(np.arange(y_.shape[0]))
        groups[split]["groups_idx"] = {}
        groups[split]["groups_n"] = {}
        groups[split]["groups_names"] = {}
        groups[split]["n"] = y_.shape[0]
        groups[split]["s"] = y_.shape[1]

        if len(next(iter(groups_input.values()))) == 1:
            for g in groups_input:
                group_idx = pd.get_dummies(
                    [i[groups_input[g][0]] for i in y_]
                ).values.argmax(1)
                groups[split]["groups_idx"][g] = np.tile(
                    group_idx, (groups[split]["n"], 1)
                ).flatten("F")
                groups[split]["groups_n"][g] = np.unique(group_idx).shape[0]
                group_names = [i[groups_input[g][0]] for i in y_]
                groups[split]["groups_names"][g] = np.unique(group_names)
        else:
            for g in groups_input:
                group_idx = pd.get_dummies(
                    [i[groups_input[g][0] : groups_input[g][1]] for i in y_]
                ).values.argmax(1)
                groups[split]["groups_idx"][g] = np.tile(
                    group_idx, (groups[split]["n"], 1)
                ).flatten("F")
                groups[split]["groups_n"][g] = np.unique(group_idx).shape[0]
                group_names = [
                    i[groups_input[g][0] : groups_input[g][1]] for i in y_
                ]
                groups[split]["groups_names"][g] = np.unique(group_names)

        groups[split]["n_series_idx"] = np.tile(
            np.arange(groups[split]["s"]), (groups[split]["n"], 1)
        ).flatten("F")
        groups[split]["n_series"] = np.arange(groups[split]["s"])
        groups[split]["g_number"] = len(groups_input)
        groups[split]["data"] = y_.values.T.ravel()

    groups["predict"]["original_data"] = y.values.T.ravel()
    groups["seasonality"] = seasonality
    groups["h"] = h
    groups["dates"] = dates

    return groups


def generate_groups_data_matrix(groups: dict) -> dict:
    """Reshape flat group data arrays into matrix form."""
    for group in groups["train"]["groups_idx"].keys():
        groups["train"]["groups_idx"][group] = (
            groups["train"]["groups_idx"][group]
            .reshape(groups["train"]["s"], groups["train"]["n"])
            .T[0, :]
        )
        groups["predict"]["groups_idx"][group] = (
            groups["predict"]["groups_idx"][group]
            .reshape(groups["predict"]["s"], groups["predict"]["n"])
            .T[0, :]
        )

    groups["train"]["full_data"] = (
        groups["train"]["data"].reshape(groups["train"]["s"], groups["train"]["n"]).T
    )
    groups["train"]["data"] = (
        groups["train"]["data"].reshape(groups["train"]["s"], groups["train"]["n"]).T
    )

    groups["train"]["n_series_idx_full"] = (
        groups["train"]["n_series_idx"]
        .reshape(groups["train"]["s"], groups["train"]["n"])
        .T[0, :]
    )
    groups["train"]["n_series_idx"] = (
        groups["train"]["n_series_idx"]
        .reshape(groups["train"]["s"], groups["train"]["n"])
        .T[0, :]
    )

    groups["predict"]["n_series_idx"] = (
        groups["predict"]["n_series_idx"]
        .reshape(groups["predict"]["s"], groups["predict"]["n"])
        .T[0, :]
    )
    groups["predict"]["data_matrix"] = (
        groups["predict"]["data"]
        .reshape(groups["predict"]["s"], groups["predict"]["n"])
        .T
    )

    return groups
