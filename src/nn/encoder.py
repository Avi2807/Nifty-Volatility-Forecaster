"""Pre-norm Transformer encoder block."""

from __future__ import annotations

import numpy as np

from src.nn.attention import MultiHeadAttention
from src.nn.layers import GELU, Dense, Dropout, Layer, LayerNorm, ParamPairs


class TransformerEncoderBlock(Layer):
    """x1 = x + dropout(MHA(LN(x))), then out = x1 + dropout(FFN(LN(x1))), FFN being d -> 2d -> d.

    Dropout sits on the residual branches, after attention has been computed, so the stored
    attention weights are never randomly zeroed.
    """

    def __init__(self, d: int, n_heads: int, dropout: float, rng: np.random.Generator) -> None:
        self.norm_attention = LayerNorm(d)
        self.attention = MultiHeadAttention(d, n_heads, rng)
        self.dropout_attention = Dropout(dropout, rng)
        self.norm_ffn = LayerNorm(d)
        self.ffn_in = Dense(d, 2 * d, rng)
        self.activation = GELU()
        self.ffn_out = Dense(2 * d, d, rng)
        self.dropout_ffn = Dropout(dropout, rng)

    def set_training(self, training: bool) -> None:
        self.dropout_attention.training = training
        self.dropout_ffn.training = training

    def forward(self, x: np.ndarray) -> np.ndarray:
        attended = self.attention.forward(self.norm_attention.forward(x))
        after_attention = x + self.dropout_attention.forward(attended)
        hidden = self.activation.forward(self.ffn_in.forward(self.norm_ffn.forward(after_attention)))
        return after_attention + self.dropout_ffn.forward(self.ffn_out.forward(hidden))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Each residual add sends the same gradient down both branches, and the two
        # incoming gradients add up at the point where the branches split.
        hidden = self.ffn_out.backward(self.dropout_ffn.backward(dout))
        ffn_branch = self.norm_ffn.backward(self.ffn_in.backward(self.activation.backward(hidden)))
        d_after_attention = dout + ffn_branch
        attended = self.attention.backward(self.dropout_attention.backward(d_after_attention))
        return d_after_attention + self.norm_attention.backward(attended)

    def params(self) -> ParamPairs:
        return (
            self.norm_attention.params()
            + self.attention.params()
            + self.norm_ffn.params()
            + self.ffn_in.params()
            + self.ffn_out.params()
        )

    def attention_weights(self) -> np.ndarray:
        """Weights from the last forward pass, averaged over heads: (batch, time, time)."""
        return self.attention.attention_weights()
