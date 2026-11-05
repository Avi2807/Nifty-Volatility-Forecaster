"""Small numerical helpers shared across the project."""

from __future__ import annotations

import numpy as np


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax that cannot overflow: subtracting the maximum keeps every exponent <= 0."""
    shifted = x - x.max(axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)
