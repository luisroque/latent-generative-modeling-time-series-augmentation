"""
Preprocessing pipeline for loading and formatting time series datasets.
Supports TourismSmall (via datasetsforecast) and synthetic data generation.
Converts raw data into the internal 'groups' dict consumed by the rest of the codebase.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from datasetsforecast.hierarchical import HierarchicalData
from datasetsforecast.m3 import M3

from .utils import generate_groups_data_flat, generate_groups_data_matrix

M3_GROUP_BY_FREQ: dict[str, tuple[str, int, int]] = {
    "Y": ("Yearly", 1, 6),
    "Q": ("Quarterly", 4, 8),
    "M": ("Monthly", 12, 18),
    "D": ("Other", 1, 8),
}

HIERARCHICAL_DATASET_PARAMS: dict[str, tuple[int, int]] = {
    "Traffic": (7, 14),
    "Wiki2": (7, 14),
    "Labour": (12, 8),
}

DATA_DIR = "assets/data"


class PreprocessDatasets:
    """
    Loads and preprocesses datasets into the internal groups dict format.

    Supported datasets:
        - tourism: Quarterly Australian Tourism (TourismSmall)
        - traffic: SF Bay Area daily traffic (HierarchicalData)
        - wiki2: Daily Wikipedia views (HierarchicalData)
        - labour: Australian monthly labour (HierarchicalData)
        - synthetic: Randomly generated time series
        - m3: M3 competition; freq selects group (Y/Q/M/D).

    Attributes
    ----------
    dataset : str
        Dataset identifier (e.g. 'tourism', 'traffic', 'wiki2', 'labour', 'synthetic', 'm3').
    freq : str
        Frequency of the time series (e.g. 'Q', 'D').
    input_dir : str
        Root directory for caching downloaded data and assets.
    """

    def __init__(
        self,
        dataset: str,
        freq: str,
        input_dir: str = "./",
        test_size: int | None = None,
        num_base_series_time_points: int = 100,
        num_latent_dim: int = 3,
        num_variants: int = 20,
        noise_scale: float = 0.1,
        amplitude: float = 1.0,
    ) -> None:
        self.dataset = dataset
        self.freq = freq
        self.input_dir = input_dir
        self.test_size = test_size
        self.num_base_series_time_points = num_base_series_time_points
        self.num_latent_dim = num_latent_dim
        self.num_variants = num_variants
        self.noise_scale = noise_scale
        self.amplitude = amplitude
        self._create_directories()

    def _create_directories(self) -> None:
        Path(self.input_dir, DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.input_dir, DATA_DIR, "original_datasets").mkdir(
            parents=True, exist_ok=True
        )

    def apply_preprocess(self) -> dict:
        method = getattr(self, f"_{self.dataset}", None)
        if method is None:
            raise ValueError(
                f"Unknown dataset '{self.dataset}'. "
                f"Supported: 'tourism', 'traffic', 'wiki2', 'labour', 'synthetic', 'm3'."
            )
        return method()

    def _tourism(self) -> dict:
        data_dir = str(Path(self.input_dir) / DATA_DIR)
        Y_df, S_df, tags = HierarchicalData.load(
            directory=data_dir, group="TourismSmall"
        )

        bottom_ids = S_df.columns.tolist()
        Y_bottom = Y_df[Y_df["unique_id"].isin(bottom_ids)]
        pivot = Y_bottom.pivot(index="ds", columns="unique_id", values="y")
        pivot = pivot.sort_index()
        pivot = pivot[sorted(pivot.columns)]

        if self.test_size is not None:
            pivot = pivot.iloc[:, : self.test_size]
            bottom_ids = pivot.columns.tolist()

        groups_names: dict[str, np.ndarray] = {}
        groups_idx: dict[str, np.ndarray] = {}

        purposes = [uid.split("-")[1] for uid in bottom_ids]
        states = [uid.split("-")[0] for uid in bottom_ids]
        city_types = [uid.split("-")[2] for uid in bottom_ids]

        for label, values in [
            ("Purpose", purposes),
            ("State", states),
            ("CityType", city_types),
        ]:
            unique_vals = np.array(sorted(set(values)))
            val_to_idx = {v: i for i, v in enumerate(unique_vals)}
            groups_names[label] = unique_vals
            groups_idx[label] = np.array([val_to_idx[v] for v in values])

        seasonality = 4
        h = 4
        groups = self._build_groups_dict(
            pivot, groups_names, groups_idx, seasonality, h
        )
        return groups

    def _load_hierarchical_bottom(
        self, group_name: str, seasonality: int, h: int
    ) -> dict:
        data_dir = str(Path(self.input_dir) / DATA_DIR)
        Y_df, S_df, _ = HierarchicalData.load(
            directory=data_dir, group=group_name
        )
        bottom_ids = S_df.columns.tolist()
        Y_bottom = Y_df[Y_df["unique_id"].isin(bottom_ids)]
        pivot = Y_bottom.pivot(index="ds", columns="unique_id", values="y")
        pivot = pivot.sort_index()
        pivot = pivot[sorted(pivot.columns)]
        if self.test_size is not None:
            pivot = pivot.iloc[:, : self.test_size]
            bottom_ids = pivot.columns.tolist()
        groups_names: dict[str, np.ndarray] = {}
        groups_idx: dict[str, np.ndarray] = {}
        groups_names["Series"] = np.array(bottom_ids)
        groups_idx["Series"] = np.arange(len(bottom_ids))
        return self._build_groups_dict(
            pivot, groups_names, groups_idx, seasonality, h
        )

    def _traffic(self) -> dict:
        seasonality, h = HIERARCHICAL_DATASET_PARAMS["Traffic"]
        return self._load_hierarchical_bottom("Traffic", seasonality, h)

    def _wiki2(self) -> dict:
        seasonality, h = HIERARCHICAL_DATASET_PARAMS["Wiki2"]
        return self._load_hierarchical_bottom("Wiki2", seasonality, h)

    def _labour(self) -> dict:
        seasonality, h = HIERARCHICAL_DATASET_PARAMS["Labour"]
        return self._load_hierarchical_bottom("Labour", seasonality, h)

    def _m3(self) -> dict:
        if self.freq not in M3_GROUP_BY_FREQ:
            raise ValueError(
                f"M3 dataset does not support freq '{self.freq}'. "
                f"Use one of: {list(M3_GROUP_BY_FREQ.keys())}."
            )
        group_name, seasonality, h = M3_GROUP_BY_FREQ[self.freq]
        data_dir = str(Path(self.input_dir) / DATA_DIR)
        load_result = M3.load(directory=data_dir, group=group_name)
        # Use only the first return: target series (unique_id, ds, y). Second/third are
        # optional exogenous/static; concatenating them produced wrong pivot and NaNs.
        full_df = (
            load_result[0]
            .drop_duplicates(subset=["unique_id", "ds"], keep="last")
            .sort_values(["unique_id", "ds"])
        )
        pivot = full_df.pivot(index="ds", columns="unique_id", values="y")
        pivot = pivot.sort_index()
        pivot = pivot[sorted(pivot.columns)]
        valid_mask = ~pivot.isna()
        if self.test_size is not None:
            pivot = pivot.iloc[:, : self.test_size]
            valid_mask = valid_mask.iloc[:, : self.test_size]
        bottom_ids = pivot.columns.tolist()
        groups_names: dict[str, np.ndarray] = {}
        groups_idx: dict[str, np.ndarray] = {}
        series_labels = np.array(bottom_ids)
        groups_names["Series"] = series_labels
        groups_idx["Series"] = np.arange(len(bottom_ids))
        groups = self._build_groups_dict(
            pivot, groups_names, groups_idx, seasonality, h, valid_mask=valid_mask.values
        )
        return groups

    @staticmethod
    def _build_groups_dict(
        df_pivot: pd.DataFrame,
        groups_names: dict[str, np.ndarray],
        groups_idx: dict[str, np.ndarray],
        seasonality: int,
        h: int,
        valid_mask: np.ndarray | None = None,
    ) -> dict:
        """Build the internal groups dict from a wide-format DataFrame."""
        dates = list(df_pivot.index)
        n_total = df_pivot.shape[0]
        n_train = n_total - h
        s = df_pivot.shape[1]
        data = df_pivot.values

        groups: dict = {}
        for split, n, d in [
            ("train", n_train, data[:n_train]),
            ("predict", n_total, data),
        ]:
            groups[split] = {
                "n": n,
                "s": s,
                "data": d,
                "full_data": d if split == "train" else None,
                "data_matrix": d if split == "predict" else None,
                "groups_idx": {k: v.copy() for k, v in groups_idx.items()},
                "groups_n": {k: len(v) for k, v in groups_names.items()},
                "groups_names": {k: v.copy() for k, v in groups_names.items()},
                "n_series_idx": np.arange(s),
                "n_series": np.arange(s),
                "x_values": list(range(n)),
                "g_number": len(groups_names),
            }
            if valid_mask is not None:
                m = valid_mask[:n] if split == "train" else valid_mask
                groups[split]["valid_mask"] = m.astype(np.uint8)

        groups["predict"]["original_data"] = data.T.ravel()
        groups["seasonality"] = seasonality
        groups["h"] = h
        groups["dates"] = dates

        return groups

    def _synthetic(self) -> dict:
        if self.freq == "D":
            seasonality, h = 365, 30
        elif self.freq == "W":
            seasonality, h = 52, 12
        elif self.freq == "M":
            seasonality, h = 12, 2
        elif self.freq == "Q":
            seasonality, h = 4, 2
        else:
            raise ValueError(f"Unsupported frequency: {self.freq}")

        base_series = self._create_base_time_series(
            self.num_base_series_time_points,
            self.num_latent_dim,
            seasonality,
            self.amplitude,
        )
        variants = self._generate_variants(
            base_series, self.num_variants, self.noise_scale
        )

        start_date = pd.Timestamp("2023-01-01")
        end_date = start_date + timedelta(days=self.num_base_series_time_points - 1)
        dates = pd.date_range(start_date, end_date, freq=self.freq)

        df = pd.DataFrame(np.concatenate(variants, axis=1), index=dates)

        column_tuples = [
            ("group_1", f"group_element_{i // self.num_variants + 1}")
            for i in range(self.num_latent_dim * self.num_variants)
        ]
        df.columns = pd.MultiIndex.from_tuples(
            column_tuples, names=["Group", "Element"]
        )

        groups_input = {"group_1": [1]}
        groups = generate_groups_data_flat(
            y=df,
            dates=list(df.index),
            groups_input=groups_input,
            seasonality=seasonality,
            h=h,
        )
        groups = generate_groups_data_matrix(groups)
        groups["base_series"] = np.array(base_series)
        return groups

    @staticmethod
    def _generate_time_series(length: int, num_series: int) -> list[np.ndarray]:
        scaler = MinMaxScaler(feature_range=(0, 1))
        return [
            scaler.fit_transform(np.random.randn(length, 1))
            for _ in range(num_series)
        ]

    @classmethod
    def _create_base_time_series(
        cls,
        length: int,
        num_series: int,
        seasonality_period: int,
        amplitude: float,
    ) -> list[np.ndarray]:
        base_series = cls._generate_time_series(length, num_series)
        for series in base_series:
            t = np.arange(len(series))
            seasonal_component = np.sin(
                2 * np.pi * t * amplitude / seasonality_period
            )
            base_series += seasonal_component[:, np.newaxis]
        return base_series

    @staticmethod
    def _generate_variants(
        base_series: list[np.ndarray],
        num_variants: int,
        noise_scale: float,
    ) -> list[np.ndarray]:
        variants = []
        for series in base_series:
            for _ in range(num_variants):
                noise = np.random.normal(scale=noise_scale, size=series.shape)
                variants.append(series + noise)
        return variants
