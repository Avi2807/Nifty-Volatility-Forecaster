"""The model, its windows and its training loop."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import MARKET_FEATURES
from src.eval.references import har_fit, har_forecast, naive_forecast, vix_implied_forecast
from src.models.dataset import build_dataset
from src.models.train import load_checkpoint, predict, save_checkpoint, train_model
from src.models.transformer import WINDOW_DAYS, TransformerForecaster
from src.nn.gradcheck import TOLERANCE, check

N_ROWS = 160
N_FEATURES = 6
NAMES = tuple(f"f{i}" for i in range(N_FEATURES))


def make_rows(seed: int = 0):
    """Rows where the target depends on recent inputs, split into train, val and test by date."""
    rng = np.random.default_rng(seed)
    dates = np.datetime64("2019-01-01") + np.arange(N_ROWS)
    x = rng.normal(size=(N_ROWS, N_FEATURES))
    target = -10.0 + 0.8 * x[:, 0] + 0.1 * rng.normal(size=N_ROWS)
    labels = np.array(["train"] * 100 + ["val"] * 30 + ["test"] * 30)
    return dates, x, target, labels


def dataset(seed: int = 0):
    return build_dataset(*make_rows(seed), NAMES)


@pytest.mark.parametrize("n_features, expected", [(18, 9761), (15, 9665)])
def test_parameter_counts(n_features: int, expected: int):
    assert TransformerForecaster(n_features, 1).n_parameters() == expected


def test_model_gradients_match_central_differences():
    model = TransformerForecaster(4, 3)
    x = np.random.default_rng(0).normal(size=(2, WINDOW_DAYS, 4))

    def run() -> np.ndarray:
        model.block.dropout_attention.rng = np.random.default_rng(1)
        model.block.dropout_ffn.rng = np.random.default_rng(2)
        return model.forward(x)

    results = check("model", run, lambda w: {"input": model.backward(w)}, {"input": x}, model.params())
    assert all(r.analytic_norm > 0 and r.relative_error < TOLERANCE for r in results)


def test_forecast_still_depends_on_every_earlier_day_through_attention():
    """Last-day pooling must not discard the rest of the 22-day window."""
    model = TransformerForecaster(4, 3)
    model.set_training(False)
    x = np.random.default_rng(0).normal(size=(1, WINDOW_DAYS, 4))
    base = model.forward(x).copy()
    for day in range(WINDOW_DAYS):
        changed = x.copy()
        changed[0, day] += 1.0
        assert not np.allclose(model.forward(changed), base), f"day {day} has no effect"


def test_each_window_ends_on_its_own_date_and_holds_no_later_row():
    dates, x, target, labels = make_rows()
    data = build_dataset(dates, x, target, labels, NAMES)
    val = data.splits["val"]
    scaled = data.scaler.transform(x)
    for i, date in enumerate(val.dates):
        row = int(np.flatnonzero(dates == date)[0])
        assert val.x.shape[1:] == (WINDOW_DAYS, N_FEATURES)
        np.testing.assert_array_equal(val.x[i], scaled[row - WINDOW_DAYS + 1 : row + 1])


def test_window_targets_line_up_and_convert_back_to_log_rv():
    dates, x, target, labels = make_rows()
    data = build_dataset(dates, x, target, labels, NAMES)
    test = data.splits["test"]
    positions = np.searchsorted(dates, test.dates)
    np.testing.assert_allclose(data.to_log_rv(test.y)[:, 0], target[positions])


def test_windows_take_their_split_from_their_last_day():
    data = dataset()
    assert data.splits["train"].y.shape[0] == 100 - (WINDOW_DAYS - 1)
    assert data.splits["val"].y.shape[0] == 30  # may look back into the training rows
    assert data.splits["test"].dates.min() > data.splits["val"].dates.max() > data.splits["train"].dates.max()


def test_scaling_uses_training_rows_only():
    """This closes the gap left by the stage 1 scaler test: here the wiring itself is tested."""
    dates, x, target, labels = make_rows()
    changed_x, changed_target = x.copy(), target.copy()
    later = labels != "train"
    changed_x[later] *= 1000.0
    changed_target[later] *= 1000.0
    first = build_dataset(dates, x, target, labels, NAMES)
    second = build_dataset(dates, changed_x, changed_target, labels, NAMES)
    np.testing.assert_array_equal(first.scaler.mean, second.scaler.mean)
    np.testing.assert_array_equal(first.scaler.std, second.scaler.std)
    assert first.target_mean == second.target_mean and first.target_std == second.target_std
    np.testing.assert_array_equal(first.splits["train"].x, second.splits["train"].x)
    np.testing.assert_allclose(first.scaler.transform(x[labels == "train"]).mean(axis=0), 0.0, atol=1e-12)


def test_missing_features_are_refused():
    dates, x, target, labels = make_rows()
    x[5, 0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        build_dataset(dates, x, target, labels, NAMES)


def test_training_never_sees_the_test_rows():
    """Corrupting every test row must leave the trained weights exactly the same."""
    dates, x, target, labels = make_rows()
    changed_x, changed_target = x.copy(), target.copy()
    changed_x[labels == "test"] = 1e6
    changed_target[labels == "test"] = 1e6
    weights = []
    for features, y in ((x, target), (changed_x, changed_target)):
        data = build_dataset(dates, features, y, labels, NAMES)
        model = TransformerForecaster(N_FEATURES, 1)
        train_model(model, data.splits["train"], data.splits["val"], seed=1, max_epochs=2)
        weights.append(model.get_weights())
    for first, second in zip(*weights):
        np.testing.assert_array_equal(first, second)


def test_same_seed_reproduces_and_a_different_seed_differs():
    data = dataset()
    runs = []
    for seed in (1, 1, 2):
        model = TransformerForecaster(N_FEATURES, seed)
        train_model(model, data.splits["train"], data.splits["val"], seed=seed, max_epochs=2)
        runs.append(predict(model, data.splits["val"].x)[0])
    np.testing.assert_array_equal(runs[0], runs[1])
    assert not np.array_equal(runs[0], runs[2])


def test_early_stopping_restores_the_best_epoch():
    data = dataset()
    model = TransformerForecaster(N_FEATURES, 1)
    val = data.splits["val"]
    run = train_model(model, data.splits["train"], val, seed=1, max_epochs=40, patience=3)
    assert run.val_curve[run.best_epoch - 1] == min(run.val_curve)
    assert run.epochs_run <= 40
    assert float(np.mean((predict(model, val.x)[0] - val.y) ** 2)) == pytest.approx(run.best_val_mse)


def test_training_generalises_and_can_find_the_most_recent_day():
    """With a real-sized set, the target depends only on the last day of each 22-day window."""
    rng = np.random.default_rng(4)
    n_train, n_val = 1200, 250
    x = rng.normal(size=(n_train + n_val, N_FEATURES))
    target = -10.0 + 0.8 * x[:, 0] + 0.1 * rng.normal(size=n_train + n_val)
    dates = np.datetime64("2015-01-01") + np.arange(n_train + n_val)
    labels = np.array(["train"] * n_train + ["val"] * n_val)
    data = build_dataset(dates, x, target, labels, NAMES)
    model = TransformerForecaster(N_FEATURES, 1)
    run = train_model(model, data.splits["train"], data.splits["val"], seed=1, max_epochs=40)
    assert run.best_val_mse < 0.5 * float(np.mean(data.splits["val"].y ** 2))


def test_model_can_overfit_a_small_training_set():
    data = dataset()
    train = data.splits["train"]
    model = TransformerForecaster(N_FEATURES, 1)
    model.block.dropout_attention.rate = model.block.dropout_ffn.rate = 0.0
    run = train_model(model, train, train, seed=1, max_epochs=200, patience=200)
    assert run.best_val_mse < 0.1 * float(np.mean(train.y**2))


def test_attention_output_shape_and_prediction_is_deterministic():
    data = dataset()
    model = TransformerForecaster(N_FEATURES, 1)
    x = data.splits["test"].x
    first, attention = predict(model, x)
    assert attention.shape == (x.shape[0], WINDOW_DAYS, WINDOW_DAYS)
    np.testing.assert_allclose(attention.sum(axis=-1), 1.0)
    np.testing.assert_array_equal(first, predict(model, x)[0])


def test_checkpoint_round_trip(tmp_path):
    data = dataset()
    model = TransformerForecaster(N_FEATURES, 5)
    path = tmp_path / "model.npz"
    save_checkpoint(path, model, data, 5)
    restored, meta = load_checkpoint(path)
    x = data.splits["val"].x
    np.testing.assert_array_equal(predict(model, x)[0], predict(restored, x)[0])
    assert meta["feature_names"] == NAMES
    assert meta["target_mean"] == data.target_mean
    np.testing.assert_array_equal(meta["scaler_mean"], data.scaler.mean)


def test_later_rows_are_not_allowed_into_the_sentiment_variant(tmp_path):
    from src.config import RunPaths
    from src.data.loaders import write_table
    from src.models.dataset import load_dataset
    from src.config import ALL_FEATURES

    paths = RunPaths(tmp_path, tmp_path, tmp_path, True)
    n = 40
    dates = np.datetime64("2019-12-01") + np.arange(n)
    rows = {"date": dates, **{name: np.ones(n) + i for i, name in enumerate(ALL_FEATURES)},
            "target_log_rv": np.linspace(-11, -9, n), "split": np.array(["train"] * 30 + ["val"] * 10)}
    write_table(tmp_path / "daily_features.csv", rows)
    later = {**rows, "date": dates + 100, "split": np.array(["later"] * n)}
    later.update({name: np.full(n, np.nan) for name in ALL_FEATURES if name not in MARKET_FEATURES})
    write_table(tmp_path / "daily_features_later.csv", later)

    with_news = load_dataset(paths, with_sentiment=True)
    without_news = load_dataset(paths, with_sentiment=False)
    assert "later" not in with_news.splits
    assert "later" in without_news.splits and without_news.splits["later"].y.shape[0] == n
    assert len(without_news.feature_names) == len(MARKET_FEATURES)


def test_naive_is_todays_log_rv():
    np.testing.assert_array_equal(naive_forecast(np.array([-10.0, -9.0])), [-10.0, -9.0])


def test_har_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(200, 3))
    truth = np.array([-1.0, 0.5, 0.3, 0.1])
    y = truth[0] + inputs @ truth[1:]
    coefficients = har_fit(inputs, y)
    np.testing.assert_allclose(coefficients, truth)
    np.testing.assert_allclose(har_forecast(inputs, coefficients), y)


def test_vix_implied_forecast_on_a_known_value():
    # VIX 16 is 16% a year: daily variance 0.16^2 / 252.
    np.testing.assert_allclose(vix_implied_forecast(np.log(np.array([16.0]))), np.log(0.16**2 / 252))
