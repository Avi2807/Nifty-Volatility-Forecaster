"""Central-difference gradient checks for every layer and a full encoder block.

Usage: python -m src.nn.gradcheck

Each check builds L = sum(output * random weights), which exercises every output, and
compares the hand-written gradient with (L(a + h) - L(a - h)) / 2h for every entry of every
parameter and input, in float64. The relative error is ||analytic - numeric|| / (||analytic||
+ ||numeric||). An all-zero gradient fails, because it could not show anything.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.models.lstm import LSTM, LSTMCell
from src.nn.attention import MultiHeadAttention, ScaledDotProductAttention
from src.nn.encoder import TransformerEncoderBlock
from src.nn.layers import (
    GELU,
    Dense,
    Dropout,
    LastDayPool,
    Layer,
    LayerNorm,
    MeanPool,
    ParamPairs,
    PositionalEmbedding,
    ReLU,
    Softmax,
)
from src.nn.losses import MSELoss

STEP = 1e-6
TOLERANCE = 1e-6


@dataclass(frozen=True)
class CheckResult:
    case: str
    array: str
    shape: tuple[int, ...]
    analytic_norm: float
    relative_error: float

    @property
    def passed(self) -> bool:
        return self.analytic_norm > 0.0 and self.relative_error < TOLERANCE


def _numeric_gradient(loss: Callable[[], float], array: np.ndarray) -> np.ndarray:
    gradient = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + STEP
        above = loss()
        array[index] = original - STEP
        below = loss()
        array[index] = original
        gradient[index] = (above - below) / (2.0 * STEP)
    return gradient


def check(
    case: str,
    run: Callable[[], np.ndarray],
    backward: Callable[[np.ndarray], dict[str, np.ndarray]],
    inputs: dict[str, np.ndarray],
    params: ParamPairs,
    seed: int = 0,
) -> list[CheckResult]:
    """Compare analytic and numeric gradients of sum(run() * weights) for each array."""
    weights = np.random.default_rng(seed).normal(size=np.shape(run()))
    input_grads = backward(weights)
    targets = [(name, value, input_grads[name].copy()) for name, value in inputs.items()]
    targets += [(f"param[{i}]", value, grad.copy()) for i, (value, grad) in enumerate(params)]

    def loss() -> float:
        return float(np.sum(run() * weights))

    results = []
    for name, value, analytic in targets:
        numeric = _numeric_gradient(loss, value)
        denominator = np.linalg.norm(analytic) + np.linalg.norm(numeric)
        error = float(np.linalg.norm(analytic - numeric) / denominator) if denominator > 0 else 0.0
        results.append(CheckResult(case, name, value.shape, float(np.linalg.norm(analytic)), error))
    return results


def _check_layer(case: str, layer: Layer, x: np.ndarray, before: Callable[[], None] | None = None) -> list[CheckResult]:
    def run() -> np.ndarray:
        if before is not None:
            before()  # e.g. restore a dropout seed, so that every forward uses the same mask
        return layer.forward(x)

    return check(case, run, lambda weights: {"input": layer.backward(weights)}, {"input": x}, layer.params())


def _check_attention(rng: np.random.Generator) -> list[CheckResult]:
    attention = ScaledDotProductAttention()
    q, k, v = (rng.normal(size=(2, 4, 8)) for _ in range(3))

    def backward(weights: np.ndarray) -> dict[str, np.ndarray]:
        dq, dk, dv = attention.backward(weights)
        return {"Q": dq, "K": dk, "V": dv}

    return check("ScaledDotProductAttention", lambda: attention.forward(q, k, v), backward, {"Q": q, "K": k, "V": v}, [])


def _check_loss(rng: np.random.Generator) -> list[CheckResult]:
    loss = MSELoss()
    prediction, target = rng.normal(size=(6, 1)), rng.normal(size=(6, 1))
    return check(
        "MSELoss",
        lambda: np.array(loss.forward(prediction, target)),
        lambda weights: {"prediction": weights * loss.backward()},
        {"prediction": prediction},
        [],
    )


def _check_block(case: str, d: int, n_time: int, rng: np.random.Generator) -> list[CheckResult]:
    block = TransformerEncoderBlock(d, 2, 0.1, rng)
    for pair in (block.norm_attention, block.norm_ffn):
        pair.gamma[...] = rng.normal(1.0, 0.3, d)
        pair.beta[...] = rng.normal(0.0, 0.3, d)

    def fix_dropout_masks() -> None:
        block.dropout_attention.rng = np.random.default_rng(11)
        block.dropout_ffn.rng = np.random.default_rng(12)

    return _check_layer(case, block, rng.normal(size=(2, n_time, d)), fix_dropout_masks)


def _check_lstm_cell(rng: np.random.Generator) -> list[CheckResult]:
    cell = LSTMCell(2, 3, rng)
    x, h, c = rng.normal(size=(2, 2)), rng.normal(size=(2, 3)), rng.normal(size=(2, 3))

    def backward(weights: np.ndarray) -> dict[str, np.ndarray]:
        dx, dh, dc = cell.backward(weights[0], weights[1])
        return {"x": dx, "h": dh, "c": dc}

    return check("LSTMCell (2 in, 3 hidden)", lambda: np.stack(cell.forward(x, h, c)), backward, {"x": x, "h": h, "c": c}, cell.params())


def run_all_checks() -> list[CheckResult]:
    rng = np.random.default_rng(2024)
    results: list[CheckResult] = []

    results += _check_layer("Dense", Dense(5, 3, rng), rng.normal(size=(2, 4, 5)))
    results += _check_layer("Dense (no bias)", Dense(5, 3, rng, bias=False), rng.normal(size=(2, 4, 5)))
    results += _check_layer("GELU", GELU(), rng.normal(0.0, 1.5, size=(3, 4)))

    dropout = Dropout(0.3, rng)

    def fix_mask() -> None:
        dropout.rng = np.random.default_rng(5)

    results += _check_layer("Dropout (training)", dropout, rng.normal(size=(3, 4)), fix_mask)

    norm = LayerNorm(6)
    norm.gamma[...] = rng.normal(1.0, 0.3, 6)
    norm.beta[...] = rng.normal(0.0, 0.3, 6)
    results += _check_layer("LayerNorm", norm, rng.normal(size=(2, 3, 6)))

    results += _check_layer("Softmax", Softmax(), rng.normal(size=(2, 3, 5)))
    results += _check_attention(rng)
    results += _check_layer("MultiHeadAttention", MultiHeadAttention(8, 2, rng), rng.normal(size=(2, 4, 8)))
    results += _check_layer("PositionalEmbedding", PositionalEmbedding(6, 4, rng), rng.normal(size=(2, 4, 4)))
    results += _check_layer("MeanPool", MeanPool(), rng.normal(size=(2, 4, 3)))
    results += _check_layer("LastDayPool", LastDayPool(), rng.normal(size=(2, 4, 3)))
    results += _check_loss(rng)
    results += _check_block("EncoderBlock (T=4, d=8)", 8, 4, rng)
    results += _check_block("EncoderBlock (T=22, d=32)", 32, 22, rng)
    # New checks go last, because each one draws from rng and would change the numbers of any after it.
    results += _check_layer("ReLU", ReLU(), rng.normal(0.0, 1.5, size=(3, 4)))
    results += _check_lstm_cell(rng)
    results += _check_layer("LSTM (2 in, 3 hid, T=4)", LSTM(2, 3, rng), rng.normal(size=(2, 4, 2)))
    return results


def main() -> int:
    results = run_all_checks()
    print(f"{'case':28s} {'array':11s} {'shape':14s} {'|analytic|':>11s} {'rel. error':>11s}  result")
    for r in results:
        print(
            f"{r.case:28s} {r.array:11s} {str(r.shape):14s} {r.analytic_norm:11.4e} {r.relative_error:11.3e}"
            f"  {'pass' if r.passed else 'FAIL'}"
        )
    worst = max(r.relative_error for r in results)
    n_passed = sum(r.passed for r in results)
    print(f"\n{n_passed}/{len(results)} passed, worst relative error {worst:.3e}, tolerance {TOLERANCE:.0e}")
    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
