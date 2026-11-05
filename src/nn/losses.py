"""Mean squared error loss."""

from __future__ import annotations

import numpy as np


class MSELoss:
    def __init__(self) -> None:
        self.residual = np.empty(0)

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        self.residual = prediction - target
        return float(np.mean(self.residual**2))

    def backward(self) -> np.ndarray:
        """dL/d(prediction) = 2 (prediction - target) / n."""
        return 2.0 * self.residual / self.residual.size
