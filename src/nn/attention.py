"""Scaled dot-product attention and multi-head self-attention, with hand-written gradients."""

from __future__ import annotations

import numpy as np

from src.nn.layers import Dense, Layer, ParamPairs, Softmax


def _swap(x: np.ndarray) -> np.ndarray:
    """Transpose the last two axes."""
    return np.swapaxes(x, -1, -2)


class ScaledDotProductAttention:
    """out = softmax(Q K^T / sqrt(d_k)) V, over the last two axes (time, d_k).

    This takes three inputs, so forward(q, k, v) and backward returning (dq, dk, dv) replace
    the single-input layer contract. It has no parameters of its own.
    """

    def __init__(self) -> None:
        self.softmax = Softmax()
        self.scale = 1.0
        self.q = self.k = self.v = self.weights = np.empty(0)

    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.q, self.k, self.v = q, k, v
        self.scale = 1.0 / np.sqrt(q.shape[-1])
        self.weights = self.softmax.forward(q @ _swap(k) * self.scale)
        return self.weights @ v

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Walk the forward computation in reverse: out = A V, then A = softmax(S), S = Q K^T scale.
        dv = _swap(self.weights) @ dout
        dweights = dout @ _swap(self.v)
        dscores = self.softmax.backward(dweights) * self.scale
        dq = dscores @ self.k
        dk = _swap(dscores) @ self.q
        return dq, dk, dv


class MultiHeadAttention(Layer):
    """Self-attention with several heads: inputs and outputs are (batch, time, d).

    Projections follow the original Transformer and have no bias term.
    """

    def __init__(self, d: int, n_heads: int, rng: np.random.Generator) -> None:
        if d % n_heads:
            raise ValueError(f"d={d} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.to_q = Dense(d, d, rng, bias=False)
        self.to_k = Dense(d, d, rng, bias=False)
        self.to_v = Dense(d, d, rng, bias=False)
        self.to_out = Dense(d, d, rng, bias=False)
        self.attention = ScaledDotProductAttention()

    def _split(self, x: np.ndarray) -> np.ndarray:
        """(batch, time, d) -> (batch, heads, time, d_head)."""
        batch, time, _ = x.shape
        return x.reshape(batch, time, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def _merge(self, x: np.ndarray) -> np.ndarray:
        """(batch, heads, time, d_head) -> (batch, time, d)."""
        batch, _, time, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, time, self.n_heads * self.d_head)

    def forward(self, x: np.ndarray) -> np.ndarray:
        q = self._split(self.to_q.forward(x))
        k = self._split(self.to_k.forward(x))
        v = self._split(self.to_v.forward(x))
        return self.to_out.forward(self._merge(self.attention.forward(q, k, v)))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dq, dk, dv = self.attention.backward(self._split(self.to_out.backward(dout)))
        # x feeds all three projections, so its gradient is the sum of the three paths.
        return (
            self.to_q.backward(self._merge(dq))
            + self.to_k.backward(self._merge(dk))
            + self.to_v.backward(self._merge(dv))
        )

    def params(self) -> ParamPairs:
        return self.to_q.params() + self.to_k.params() + self.to_v.params() + self.to_out.params()

    def attention_weights(self) -> np.ndarray:
        """Weights from the last forward pass, averaged over heads: (batch, time, time)."""
        return self.attention.weights.mean(axis=1)
