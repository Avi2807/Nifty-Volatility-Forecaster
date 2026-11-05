"""Turn the feature tables into standardised 22-day windows for the Transformer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import ALL_FEATURES, MARKET_FEATURES, NEWS_FEATURES, RunPaths
from src.data.loaders import read_table
from src.data.splits import Scaler
from src.models.transformer import WINDOW_DAYS


@dataclass(frozen=True)
class Windows:
    x: np.ndarray  # (n, 22, n_features), standardised
    y: np.ndarray  # (n, 1), standardised log RV of the day after dates[i]
    dates: np.ndarray  # the feature date t, which is the last day of each window


@dataclass(frozen=True)
class Dataset:
    feature_names: tuple[str, ...]
    scaler: Scaler
    target_mean: float
    target_std: float
    splits: dict[str, Windows]

    def to_log_rv(self, standardised: np.ndarray) -> np.ndarray:
        """Undo the target scaling, returning a forecast in log RV units."""
        return standardised * self.target_std + self.target_mean


def build_dataset(
    dates: np.ndarray, x: np.ndarray, target: np.ndarray, labels: np.ndarray, feature_names: tuple[str, ...]
) -> Dataset:
    """Standardise with training rows only, then cut one window ending on each row.

    A window's split is that of its last day. Validation and test windows may look back
    into earlier splits: those days are already known by then, and no training target is
    involved. The target is scaled too, because a final Dense layer starting near zero
    would otherwise spend training crawling toward log RV values near -10. Forecasts are
    converted back before any scoring.
    """
    if not np.isfinite(x).all():
        raise ValueError("Features contain missing values. Has the sentiment step run?")
    is_train = labels == "train"
    scaler = Scaler.fit(x[is_train])
    target_mean, target_std = float(target[is_train].mean()), float(target[is_train].std())
    scaled_x = scaler.transform(x)
    scaled_y = (target - target_mean) / target_std

    all_windows = np.lib.stride_tricks.sliding_window_view(scaled_x, WINDOW_DAYS, axis=0).transpose(0, 2, 1)
    last_rows = np.arange(WINDOW_DAYS - 1, dates.size)
    splits = {}
    for split in np.unique(labels):
        chosen = labels[last_rows] == split
        rows = last_rows[chosen]
        splits[str(split)] = Windows(all_windows[chosen], scaled_y[rows, None], dates[rows])
    return Dataset(feature_names, scaler, target_mean, target_std, splits)


def load_dataset(paths: RunPaths, with_sentiment: bool) -> Dataset:
    """Load one model variant's windows.

    The no-sentiment variant also gets the 'later' split after June 2023. The news archive
    ends there, so the with-sentiment variant can't, and its news is never filled in.
    """
    names = ALL_FEATURES if with_sentiment else MARKET_FEATURES
    tables = [read_table(paths.processed / "daily_features.csv")]
    later_path = paths.processed / "daily_features_later.csv"
    if not with_sentiment and later_path.exists():
        tables.append(read_table(later_path))
    x = np.vstack([np.column_stack([t[name].astype(np.float64) for name in names]) for t in tables])
    dates = np.concatenate([t["date"] for t in tables])
    target = np.concatenate([t["target_log_rv"].astype(np.float64) for t in tables])
    labels = np.concatenate([t["split"].astype(str) for t in tables])
    if set(names) & set(NEWS_FEATURES) and (labels == "later").any():
        raise ValueError("The with-sentiment variant must not include dates after the news archive ends")
    return build_dataset(dates, x, target, labels, names)
