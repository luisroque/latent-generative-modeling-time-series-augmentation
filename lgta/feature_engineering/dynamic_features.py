import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_dynamic_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Create a dataframe with the dynamic features computed.
    The features are dynamic since they change over time (over the number of samples n).

    :param df: dataframe with the raw data
    :param freq: frequency of the dataset
    :param trigonometric: whether to use trigonometric functions or simple extraction

    :return: new dataframe with the dynamic features converted to np.ndarray
    """
    train_df_input = pd.DataFrame()

    freq_to_period = {
        "D": ("day", "week", "month", "quarter", "year"),
        "W": ("week", "month", "quarter", "year"),
        "MS": ("month", "quarter", "year"),
        "QS": ("quarter", "year"),
        "Y": ("year",),
    }

    periods = freq_to_period.get(freq, ("year",))

    for period in periods:
        if period == "week":
            values = df.index.isocalendar().week.astype(int)
        else:
            values = getattr(df.index, period)

        max_val = values.max()
        min_val = values.min()

        if (max_val - min_val) > 0:
            train_df_input[f"{period}_cos"] = np.cos(
                2 * np.pi * (values - min_val) / (max_val - min_val)
            )
            train_df_input[f"{period}_sin"] = np.sin(
                2 * np.pi * (values - min_val) / (max_val - min_val)
            )

    return train_df_input.astype(np.float32)


def plot_dynamic_features(df_dynamic: pd.DataFrame) -> None:
    """
    Help function to plot the dynamic features

    :param df_dynamic: dataframe containing the dynamic features
    """
    _, ax = plt.subplots(1, 2, figsize=(20, 5))

    ax = ax.ravel()

    ax[0].plot(df_dynamic.index, df_dynamic.iloc[:, :2])
    ax[1].plot(df_dynamic.index, df_dynamic.iloc[:, 2:])
    plt.show()
