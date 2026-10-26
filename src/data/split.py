"""Chronological splits, train-only scaling, and look-back windows."""
import numpy as np
import pandas as pd

from src.config import TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END


def split_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    target_date = df["target_date"]
    return {
        "train": (target_date <= TRAIN_END).to_numpy(),
        "val": ((target_date >= VAL_START) & (target_date <= VAL_END)).to_numpy(),
        "test": ((target_date >= TEST_START) & (target_date <= TEST_END)).to_numpy(),
    }


class Scaler:
    """Standardizer whose statistics must be fitted on training rows only."""
    def fit(self, X: np.ndarray) -> "Scaler":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mean"):
            raise RuntimeError("Scaler must be fitted before transform")
        return (X - self.mean) / self.std


def make_windows(X: np.ndarray, y: np.ndarray, T: int):
    if T < 1 or len(X) < T:
        raise ValueError("window length must be positive and fit within X")
    end_idx = np.arange(T - 1, len(X))
    windows = np.stack([X[i - T + 1:i + 1] for i in end_idx])
    return windows, y[end_idx], end_idx
