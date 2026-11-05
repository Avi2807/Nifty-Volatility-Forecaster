"""The volatility Transformer: 22 trading days of features -> next-day log RV."""

from __future__ import annotations

import numpy as np

from src.nn.encoder import TransformerEncoderBlock
from src.nn.layers import Dense, LastDayPool, Layer, ParamPairs, PositionalEmbedding

WINDOW_DAYS = 22
D_MODEL = 32
N_HEADS = 2
DROPOUT = 0.1


class TransformerForecaster:
    """Dense(n_features -> 32) + positional embedding -> encoder block -> last day -> Dense(32 -> 1).

    The brief specified mean pooling. It gave a validation MSE of 0.73-0.76 against 0.41 for HAR,
    so last-day pooling replaced it, on validation evidence and before the test set was opened.
    Mean pooling gives the latest day only 1/22 of the weight on the direct path to the output.
    """

    def __init__(self, n_features: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.project = Dense(n_features, D_MODEL, rng)
        self.position = PositionalEmbedding(WINDOW_DAYS, D_MODEL, rng)
        self.block = TransformerEncoderBlock(D_MODEL, N_HEADS, DROPOUT, rng)
        self.pool = LastDayPool()
        self.head = Dense(D_MODEL, 1, rng)
        self.layers: list[Layer] = [self.project, self.position, self.block, self.pool, self.head]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """(batch, 22, n_features) -> (batch, 1)."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def params(self) -> ParamPairs:
        return [pair for layer in self.layers for pair in layer.params()]

    def n_parameters(self) -> int:
        return sum(value.size for value, _ in self.params())

    def set_training(self, training: bool) -> None:
        self.block.set_training(training)

    def attention_weights(self) -> np.ndarray:
        """Weights from the last forward pass, averaged over heads: (batch, 22, 22)."""
        return self.block.attention_weights()

    def get_weights(self) -> list[np.ndarray]:
        return [value.copy() for value, _ in self.params()]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """Copy values in place, so the references held by an optimiser stay valid."""
        for (value, _), saved in zip(self.params(), weights):
            value[...] = saved
