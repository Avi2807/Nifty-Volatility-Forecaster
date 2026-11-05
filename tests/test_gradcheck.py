"""Every hand-written gradient must match central differences. Run with -s to see the table."""

from __future__ import annotations

import pytest

from src.nn.gradcheck import TOLERANCE, CheckResult, run_all_checks

RESULTS = run_all_checks()


def test_every_required_component_was_checked():
    cases = {result.case for result in RESULTS}
    required = {
        "Dense", "GELU", "Dropout (training)", "LayerNorm", "Softmax", "ScaledDotProductAttention",
        "MultiHeadAttention", "PositionalEmbedding", "MeanPool", "LastDayPool", "MSELoss", "EncoderBlock (T=22, d=32)",
        "ReLU", "LSTMCell (2 in, 3 hidden)", "LSTM (2 in, 3 hid, T=4)",
    }
    assert required <= cases


def test_lstm_checks_cover_every_array_in_float64():
    arrays = {(r.case, r.array, r.shape) for r in RESULTS if r.case.startswith("LSTM")}
    assert arrays == {
        ("LSTMCell (2 in, 3 hidden)", "x", (2, 2)),
        ("LSTMCell (2 in, 3 hidden)", "h", (2, 3)),
        ("LSTMCell (2 in, 3 hidden)", "c", (2, 3)),
        ("LSTMCell (2 in, 3 hidden)", "param[0]", (5, 12)),
        ("LSTMCell (2 in, 3 hidden)", "param[1]", (12,)),
        ("LSTM (2 in, 3 hid, T=4)", "input", (2, 4, 2)),
        ("LSTM (2 in, 3 hid, T=4)", "param[0]", (5, 12)),
        ("LSTM (2 in, 3 hid, T=4)", "param[1]", (12,)),
    }


@pytest.mark.parametrize("result", RESULTS, ids=lambda r: f"{r.case}-{r.array}")
def test_relative_error_is_below_tolerance(result: CheckResult):
    print(f"\n{result.case:28s} {result.array:11s} {str(result.shape):14s} rel. error {result.relative_error:.3e}", end="")
    assert result.analytic_norm > 0.0, "an all-zero gradient cannot verify anything"
    assert result.relative_error < TOLERANCE
