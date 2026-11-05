"""The LSTM: its shape, its gates, its window, and whether it can learn."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import ALL_FEATURES
from src.models.dataset import Windows
from src.models.lstm import LSTM, RECIPE, LSTMCell, build_lstm
from src.models.train import forecast_only, train_model
from src.models.transformer import WINDOW_DAYS, TransformerForecaster
from src.nn.gradcheck import TOLERANCE, check
from src.nn.layers import Dropout


def test_the_cell_has_one_stacked_weight_and_an_open_forget_gate():
    cell = LSTMCell(2, 3, np.random.default_rng(0))
    assert cell.W.shape == (2 + 3, 4 * 3)
    np.testing.assert_array_equal(cell.b[:3], np.ones(3))
    np.testing.assert_array_equal(cell.b[3:], np.zeros(9))


def test_one_step_matches_the_equations_by_hand():
    cell = LSTMCell(1, 1, np.random.default_rng(0))
    cell.W[...] = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    cell.b[...] = [1.0, 0.0, 0.0, 0.0]
    x, h, c = np.array([[0.5]]), np.array([[-0.2]]), np.array([[0.3]])
    new_h, new_c = cell.forward(x, h, c)

    def sigmoid(value: float) -> float:
        return 1.0 / (1.0 + np.exp(-value))

    forget = sigmoid(0.05 - 0.1 + 1.0)
    in_gate = sigmoid(0.1 - 0.12)
    candidate = np.tanh(0.15 - 0.14)
    out_gate = sigmoid(0.2 - 0.16)
    expected_c = forget * 0.3 + in_gate * candidate
    assert new_c.item() == pytest.approx(expected_c)
    assert new_h.item() == pytest.approx(out_gate * np.tanh(expected_c))


def test_parameter_count_stays_within_twice_the_transformer():
    n_features = len(ALL_FEATURES)
    lstm = build_lstm(n_features, 1).n_parameters()
    transformer = TransformerForecaster(n_features, 1).n_parameters()
    assert lstm == (18 + 32) * 4 * 32 + 4 * 32 + 32 + 1 == 6561
    assert transformer / 2 <= lstm <= transformer * 2


def test_forecast_depends_on_every_day_of_the_window():
    model = build_lstm(4, 3)
    model.set_training(False)
    x = np.random.default_rng(0).normal(size=(1, WINDOW_DAYS, 4))
    base = model.forward(x).copy()
    for day in range(WINDOW_DAYS):
        changed = x.copy()
        changed[0, day] += 1.0
        assert model.forward(changed).item() != base.item(), day


def test_an_early_step_only_changes_outputs_from_that_step_on():
    layer = LSTM(2, 3, np.random.default_rng(0))
    x = np.random.default_rng(1).normal(size=(1, 6, 2))
    base = layer.forward(x).copy()
    changed = x.copy()
    changed[0, 3] += 1.0
    after = layer.forward(changed)
    np.testing.assert_array_equal(base[0, :3], after[0, :3])
    assert not np.array_equal(base[0, 3:], after[0, 3:])


def test_whole_model_gradients_match_central_differences():
    model = build_lstm(3, 2)
    x = np.random.default_rng(0).normal(size=(2, 5, 3))
    dropout = next(layer for layer in model.layers if isinstance(layer, Dropout))

    def run() -> np.ndarray:
        dropout.rng = np.random.default_rng(1)
        return model.forward(x)

    results = check("lstm model", run, lambda w: {"input": model.backward(w)}, {"input": x}, model.params())
    assert all(r.analytic_norm > 0 and r.relative_error < TOLERANCE for r in results)


def test_repeated_backward_doesnt_pile_up_gradients():
    layer = LSTM(2, 3, np.random.default_rng(0))
    x = np.random.default_rng(1).normal(size=(2, 4, 2))
    dout = np.random.default_rng(2).normal(size=layer.forward(x).shape)
    layer.backward(dout)
    first = layer.cell.dW.copy()
    layer.backward(dout)
    np.testing.assert_array_equal(layer.cell.dW, first)


def test_it_can_memorise_a_small_noisy_set():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 6, 4))
    y = rng.normal(size=(32, 1))
    train = Windows(x, y, np.empty(32, dtype="datetime64[D]"))
    model = build_lstm(4, 1)
    model.layers[2].rate = 0.0  # a test of capacity, so dropout is off
    train_model(model, train, train, 1, max_epochs=1500, patience=1500, recipe=RECIPE)
    assert float(np.mean((forecast_only(model, x) - y) ** 2)) < 0.02
