"""Date-based splits and a scaler fitted on the training period only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import TRAIN_END, VAL_END


def split_labels(dates: np.ndarray) -> np.ndarray:
    """'train', 'val' or 'test' for each date. Rows are never shuffled across splits."""
    return np.where(dates <= TRAIN_END, "train", np.where(dates <= VAL_END, "val", "test"))


@dataclass(frozen=True)
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x_train: np.ndarray) -> "Scaler":
        """Fit on training rows only. A constant column (e.g. absent turnover) gets std 1."""
        std = x_train.std(axis=0)
        return cls(x_train.mean(axis=0), np.where(std > 0, std, 1.0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std
