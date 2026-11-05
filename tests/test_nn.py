"""Behaviour of the NumPy core that gradient checks alone can't show."""

from __future__ import annotations

import numpy as np
import pytest

from src.nn.attention import MultiHeadAttention, ScaledDotProductAttention
from src.nn.encoder import TransformerEncoderBlock
from src.nn.layers import GELU, Dense, Dropout, LastDayPool, LayerNorm, MeanPool, PositionalEmbedding
from src.nn.losses import MSELoss
from src.nn.optim import AdamW, clip_grad_norm, warmup_lr


def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def n_parameters(layer) -> int:
    return sum(value.size for value, _ in layer.params())


def test_gelu_matches_known_values():
    out = GELU().forward(np.array([-10.0, 0.0, 10.0]))
    np.testing.assert_allclose(out, [0.0, 0.0, 10.0], atol=1e-6)


def test_dropout_is_identity_when_not_training():
    layer = Dropout(0.5, rng())
    layer.training = False
    x = np.ones((3, 4))
    assert layer.forward(x) is x
    np.testing.assert_array_equal(layer.backward(x), x)


def test_dropout_keeps_the_mean_and_zeroes_about_the_right_share():
    out = Dropout(0.1, rng()).forward(np.ones((1000, 1000)))
    assert (out == 0.0).mean() == pytest.approx(0.1, abs=0.005)
    assert out.mean() == pytest.approx(1.0, abs=0.005)


def test_layernorm_output_has_zero_mean_and_unit_variance_per_position():
    out = LayerNorm(16).forward(rng().normal(3.0, 5.0, size=(2, 4, 16)))
    np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(out.std(axis=-1), 1.0, atol=1e-3)


def test_positional_embedding_gives_different_positions_different_vectors():
    x = np.zeros((1, 3, 4))
    out = PositionalEmbedding(3, 4, rng()).forward(x)
    assert not np.allclose(out[0, 0], out[0, 1])


def test_mean_pool_shape_and_value():
    out = MeanPool().forward(np.arange(12.0).reshape(1, 4, 3))
    np.testing.assert_allclose(out, [[4.5, 5.5, 6.5]])


def test_last_day_pool_keeps_only_the_final_step():
    pool = LastDayPool()
    x = np.arange(24.0).reshape(1, 4, 6)
    np.testing.assert_array_equal(pool.forward(x), x[:, -1, :])
    gradient = pool.backward(np.ones((1, 6)))
    assert gradient.shape == x.shape
    np.testing.assert_array_equal(gradient[:, :-1, :], 0.0)
    np.testing.assert_array_equal(gradient[:, -1, :], 1.0)


def test_attention_weights_are_a_distribution_over_days():
    attention = ScaledDotProductAttention()
    generator = rng()
    q, k, v = (generator.normal(size=(2, 5, 8)) for _ in range(3))
    attention.forward(q, k, v)
    assert (attention.weights >= 0).all()
    np.testing.assert_allclose(attention.weights.sum(axis=-1), 1.0)


def test_attention_output_is_the_weighted_average_of_values():
    """If one query matches one key by far the most, its output is close to that key's value."""
    keys = np.eye(3)[None] * 100.0
    values = np.array([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]])
    out = ScaledDotProductAttention().forward(keys[:, [1]], keys, values)
    np.testing.assert_allclose(out, [[[0.0, 2.0, 0.0]]], atol=1e-8)


def test_multi_head_shapes_and_head_averaged_weights():
    layer = MultiHeadAttention(32, 2, rng())
    out = layer.forward(rng().normal(size=(3, 22, 32)))
    assert out.shape == (3, 22, 32)
    assert layer.attention.weights.shape == (3, 2, 22, 22)
    weights = layer.attention_weights()
    assert weights.shape == (3, 22, 22)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0)


def test_multi_head_rejects_a_size_that_does_not_divide():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(33, 2, rng())


def test_split_and_merge_are_inverses():
    layer = MultiHeadAttention(8, 2, rng())
    x = rng().normal(size=(2, 4, 8))
    np.testing.assert_array_equal(layer._merge(layer._split(x)), x)


def test_encoder_block_has_the_expected_parameter_count():
    # MHA 4 x 32 x 32 = 4096, two LayerNorms 2 x 64 = 128, FFN (32x64+64) + (64x32+32) = 4192.
    assert n_parameters(TransformerEncoderBlock(32, 2, 0.1, rng())) == 8416


def test_encoder_block_is_deterministic_when_not_training():
    block = TransformerEncoderBlock(8, 2, 0.5, rng())
    block.set_training(False)
    x = rng().normal(size=(2, 4, 8))
    np.testing.assert_array_equal(block.forward(x), block.forward(x))


def test_stored_attention_is_not_changed_by_dropout():
    """Dropout sits on residual branches, so training mode must not alter the weights."""
    x = rng().normal(size=(1, 4, 8))
    block = TransformerEncoderBlock(8, 2, 0.5, rng())
    block.forward(x)
    in_training = block.attention_weights().copy()
    block.set_training(False)
    block.forward(x)
    np.testing.assert_array_equal(in_training, block.attention_weights())


def test_gradients_are_written_in_place_so_the_optimiser_sees_them():
    layer = Dense(3, 2, rng())
    grad_array = layer.params()[0][1]
    layer.forward(np.ones((4, 3)))
    layer.backward(np.ones((4, 2)))
    assert grad_array is layer.dW and np.abs(grad_array).sum() > 0


def test_mse_value_and_gradient_on_a_known_case():
    loss = MSELoss()
    assert loss.forward(np.array([1.0, 3.0]), np.array([0.0, 1.0])) == pytest.approx(2.5)
    np.testing.assert_allclose(loss.backward(), [1.0, 2.0])


def test_clip_scales_to_the_limit_and_reports_the_original_norm():
    grads = [(np.zeros(1), np.array([3.0])), (np.zeros(1), np.array([4.0]))]
    assert clip_grad_norm(grads, 1.0) == pytest.approx(5.0)
    np.testing.assert_allclose([grads[0][1][0], grads[1][1][0]], [0.6, 0.8])


def test_clip_leaves_small_gradients_alone():
    grads = [(np.zeros(1), np.array([0.3]))]
    assert clip_grad_norm(grads, 1.0) == pytest.approx(0.3)
    np.testing.assert_allclose(grads[0][1], [0.3])


def test_warmup_rises_linearly_then_holds():
    rates = [warmup_lr(5e-4, epoch, 5) for epoch in range(7)]
    np.testing.assert_allclose(rates, [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 5e-4, 5e-4])


def test_a_small_transformer_can_overfit_a_few_samples():
    """All the parts wired together with AdamW must be able to memorise a small data set."""
    generator = rng()
    x = generator.normal(size=(16, 6, 3))
    y = x.sum(axis=(1, 2), keepdims=False)[:, None] / 4.0
    project, position = Dense(3, 8, generator), PositionalEmbedding(6, 8, generator)
    block, pool, head = TransformerEncoderBlock(8, 2, 0.0, generator), MeanPool(), Dense(8, 1, generator)
    layers = [project, position, block, pool, head]
    params = [pair for layer in layers for pair in layer.params()]
    optimiser = AdamW(params, lr=3e-3, weight_decay=0.0)
    loss = MSELoss()
    first = last = 0.0
    for step in range(600):
        out = x
        for layer in layers:
            out = layer.forward(out)
        last = loss.forward(out, y)
        first = last if step == 0 else first
        grad = loss.backward()
        for layer in reversed(layers):
            grad = layer.backward(grad)
        optimiser.step()
    assert last < first / 100.0
