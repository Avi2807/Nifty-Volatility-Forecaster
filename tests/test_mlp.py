"""The MLP: its shape, its gradients, its inputs, and whether it can learn at all."""

from __future__ import annotations

import numpy as np

from src.config import ALL_FEATURES
from src.models.dataset import Windows
from src.models.mlp import INPUT_DAYS, build_mlp, last_days_flat
from src.models.neural import NEURAL_RECIPE
from src.models.train import forecast_only, train_model
from src.models.transformer import WINDOW_DAYS
from src.nn.gradcheck import TOLERANCE, check
from src.nn.layers import Dropout


def test_shape_and_parameter_count_for_the_project_feature_set():
    model = build_mlp(len(ALL_FEATURES), 1)
    assert len(ALL_FEATURES) == 18
    assert model.n_parameters() == (90 * 64 + 64) + (64 * 32 + 32) + (32 + 1) == 7937
    assert model.forward(np.zeros((3, 90))).shape == (3, 1)


def test_the_layout_of_the_brief_with_85_inputs():
    """17 features gives the brief's 85 inputs, so only the feature count differs from it."""
    assert build_mlp(17, 1).layers[0].W.shape == (85, 64)


def test_input_is_the_last_five_days_oldest_first():
    x = np.arange(2 * WINDOW_DAYS * 3, dtype=float).reshape(2, WINDOW_DAYS, 3)
    flat = last_days_flat(Windows(x, np.zeros((2, 1)), np.empty(2, dtype="datetime64[D]")))
    assert flat.x.shape == (2, INPUT_DAYS * 3)
    np.testing.assert_array_equal(flat.x[1, :3], x[1, WINDOW_DAYS - 5])
    np.testing.assert_array_equal(flat.x[1, -3:], x[1, WINDOW_DAYS - 1])


def test_whole_model_gradients_match_central_differences():
    model = build_mlp(4, 3)
    x = np.random.default_rng(0).normal(size=(3, 5 * 4))
    dropout = next(layer for layer in model.layers if isinstance(layer, Dropout))

    def run() -> np.ndarray:
        dropout.rng = np.random.default_rng(1)
        return model.forward(x)

    results = check("mlp", run, lambda w: {"input": model.backward(w)}, {"input": x}, model.params())
    assert all(r.analytic_norm > 0 and r.relative_error < TOLERANCE for r in results)


def test_it_overfits_64_training_rows_to_near_zero_loss():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 90))
    y = rng.normal(size=(64, 1))  # pure noise, so only memorising can bring the loss down
    train = Windows(x, y, np.empty(64, dtype="datetime64[D]"))
    model = build_mlp(18, 1)
    model.layers[2].rate = 0.0  # dropout works against memorising, so this test of capacity turns it off
    run = train_model(model, train, train, 1, max_epochs=800, patience=800, recipe=NEURAL_RECIPE)
    final_loss = float(np.mean((forecast_only(model, x) - y) ** 2))
    print(f"\nMLP on 64 rows of noise: MSE {final_loss:.2e} after {run.epochs_run} epochs (starts near {float(y.var()):.2f})")
    assert final_loss < 1e-3


def test_dropout_is_off_when_forecasting():
    model = build_mlp(18, 1)
    x = np.random.default_rng(0).normal(size=(4, 90))
    np.testing.assert_array_equal(forecast_only(model, x), forecast_only(model, x))
