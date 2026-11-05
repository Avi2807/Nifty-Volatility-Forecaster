"""MLP: the last 5 days of the same standardised features, flattened, to next-day log RV.

Usage: python -m src.models.mlp

It sees the same rows, scaling and target as the with-news Transformer, cut from the same 22-day
windows. The project has 18 features with news, so the input is 90 numbers rather than the brief's 85.
"""

from __future__ import annotations

import numpy as np

from src.config import REAL_PATHS, RunPaths
from src.models.dataset import Windows
from src.models.neural import NEURAL_RECIPE, train_neural
from src.models.train import Recipe
from src.nn.layers import Dense, Dropout, ReLU, Sequential

INPUT_DAYS = 5
DROPOUT = 0.1
HIDDEN = (64, 32)
RECIPE: Recipe = NEURAL_RECIPE  # the brief adds gradient clipping for the LSTM only


def last_days_flat(windows: Windows) -> Windows:
    """(n, 22, f) -> (n, 5 * f), oldest day first."""
    recent = windows.x[:, -INPUT_DAYS:, :]
    return Windows(recent.reshape(recent.shape[0], -1), windows.y, windows.dates)


def build_mlp(n_features: int, seed: int) -> Sequential:
    rng = np.random.default_rng(seed)
    return Sequential([
        Dense(INPUT_DAYS * n_features, HIDDEN[0], rng),
        ReLU(),
        Dropout(DROPOUT, rng),
        Dense(HIDDEN[0], HIDDEN[1], rng),
        ReLU(),
        Dense(HIDDEN[1], 1, rng),
    ])


def run_mlp(paths: RunPaths) -> dict:
    return train_neural(paths, "mlp", build_mlp, last_days_flat, RECIPE)


if __name__ == "__main__":
    run_mlp(REAL_PATHS)
