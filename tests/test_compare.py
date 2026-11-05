"""The comparison on fake data: it adds to the results, and must not change any that exist."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.data.loaders import append_columns, read_table, write_table
from src.eval.compare import LADDER, ROLLING_WINDOW, build_rows, run_comparison
from src.eval.metrics import diebold_mariano, kupiec_lr, qlike_losses, squared_errors, var99, var_breaches
from src.models.garch import load_fit, load_returns, log_variance_forecasts, run_garch
from src.models.lstm import run_lstm
from src.models.mlp import run_mlp
from tests.test_api import paths as api_paths  # noqa: F401  (a fixture)
from tests.test_eval import run  # noqa: F401  (a fixture: the whole evaluation on fake data)

NEW = ("garch", "mlp", "lstm")


def _text_columns(path, names):
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    positions = [header.index(name) for name in names]
    return [[line.split(",")[p] for p in positions] for line in lines]


@pytest.fixture(scope="module")
def compared(api_paths):  # noqa: F811
    paths = api_paths
    forecasts_path, var_path = paths.artifacts / "forecasts.csv", paths.artifacts / "var_series.csv"
    before = {
        "forecast_header": forecasts_path.read_text(encoding="utf-8").splitlines()[0].split(","),
        "var_header": var_path.read_text(encoding="utf-8").splitlines()[0].split(","),
        "metrics": json.loads((paths.artifacts / "metrics.json").read_text(encoding="utf-8")),
        "card": json.loads((paths.artifacts / "model_card.json").read_text(encoding="utf-8")),
        "freeze": (paths.artifacts / "frozen_models.json").read_bytes(),
    }
    before["forecast_text"] = _text_columns(forecasts_path, before["forecast_header"])
    before["var_text"] = _text_columns(var_path, before["var_header"])
    run_garch(paths)
    run_mlp(paths)
    run_lstm(paths)
    comparison = run_comparison(paths)
    return paths, comparison, before


def _metrics(paths):
    return json.loads((paths.artifacts / "metrics.json").read_text(encoding="utf-8"))


def test_every_existing_csv_cell_keeps_its_exact_text(compared):
    paths, _, before = compared
    assert _text_columns(paths.artifacts / "forecasts.csv", before["forecast_header"]) == before["forecast_text"]
    assert _text_columns(paths.artifacts / "var_series.csv", before["var_header"]) == before["var_text"]


def test_forecasts_csv_gains_the_columns_the_brief_asks_for(compared):
    paths, _, before = compared
    columns = list(read_table(paths.artifacts / "forecasts.csv"))
    assert columns[: len(before["forecast_header"])] == before["forecast_header"]
    for name in NEW:
        assert {name, f"var99_{name}", f"breach_{name}"} <= set(columns)


def test_existing_metrics_only_gain_entries(compared):
    paths, _, before = compared
    metrics = _metrics(paths)
    expected = copy.deepcopy(before["metrics"])
    for name in NEW:
        assert name in metrics["test"]["models"]
        expected["test"]["models"][name] = metrics["test"]["models"][name]
        expected["validation_mse_log_rv"][name] = metrics["validation_mse_log_rv"][name]
    expected["comparison"] = metrics["comparison"]
    assert metrics == expected


def test_the_scoreboards_seed_count_still_covers_only_the_transformer(compared):
    """The Scoreboard counts every entry here as a single-seed Transformer."""
    paths, _, _ = compared
    assert set(_metrics(paths)["test"]["seed_mse_log_rv"]) == {"transformer", "transformer_nosent"}


def test_the_later_check_is_untouched(compared):
    paths, _, before = compared
    assert _metrics(paths)["later_check"] == before["metrics"]["later_check"]
    assert not {"garch", "mlp", "lstm"} & set(read_table(paths.artifacts / "forecasts_later.csv"))


def test_the_transformer_freeze_record_is_untouched(compared):
    paths, _, before = compared
    assert (paths.artifacts / "frozen_models.json").read_bytes() == before["freeze"]


def test_model_card_only_gains_counts_and_gradient_checks(compared):
    paths, _, before = compared
    card = json.loads((paths.artifacts / "model_card.json").read_text(encoding="utf-8"))
    for key in ("architecture", "feature_train_stats", "training", "sentiment_classifier"):
        assert card[key] == before["card"][key]
    assert card["parameter_count"]["transformer"] == before["card"]["parameter_count"]["transformer"]
    assert {"garch", "mlp", "lstm"} <= set(card["parameter_count"])
    assert card["gradcheck"]["passed"] == card["gradcheck"]["checks"]


def test_scores_recompute_from_the_saved_columns(compared):
    paths, _, _ = compared
    table = read_table(paths.artifacts / "forecasts.csv")
    test = table["split"].astype(str) == "test"
    metrics = _metrics(paths)["test"]["models"]
    actual, returns = table["actual_log_rv"][test].astype(float), table["return_oc"][test].astype(float)
    for name in NEW:
        forecast = table[name][test].astype(float)
        assert metrics[name]["mse_log_rv"] == pytest.approx(squared_errors(actual, forecast).mean())
        assert metrics[name]["qlike"] == pytest.approx(qlike_losses(actual, forecast).mean())
        breaches = int(var_breaches(forecast, returns).sum())
        assert metrics[name]["var99"]["breaches"] == breaches == int(table[f"breach_{name}"][test].sum())
        assert metrics[name]["var99"]["kupiec_lr"] == pytest.approx(kupiec_lr(breaches, int(test.sum())))
        np.testing.assert_allclose(table[f"var99_{name}"][test].astype(float), var99(forecast))


def test_garch_forecasts_are_the_recursion_for_the_next_session(compared):
    paths, _, _ = compared
    table = read_table(paths.artifacts / "forecasts.csv")
    dates, returns = load_returns(paths)
    np.testing.assert_allclose(table["garch"].astype(float), log_variance_forecasts(load_fit(paths), dates, returns, table["date"]))


def test_changing_test_data_cant_change_the_fitted_comparison_models(compared, tmp_path):
    """Their checkpoints are read back as saved. Only a tampered checkpoint is refused."""
    paths, _, _ = compared
    saved = paths.artifacts / "comparison_models" / "garch.json"
    original = saved.read_bytes()
    try:
        saved.write_bytes(original + b" ")
        with pytest.raises(RuntimeError, match="has changed since it was frozen"):
            run_comparison(paths)
    finally:
        saved.write_bytes(original)


def test_a_changed_saved_forecast_stops_the_comparison_before_it_writes(compared):
    paths, _, _ = compared
    path = paths.artifacts / "forecasts.csv"
    metrics_before = (paths.artifacts / "metrics.json").read_bytes()
    original = path.read_bytes()
    try:
        table = read_table(path)
        table["transformer"] = table["transformer"].astype(float) + 0.01
        write_table(path, table)
        with pytest.raises(RuntimeError, match="differ from what the frozen model gives"):
            run_comparison(paths)
        assert (paths.artifacts / "metrics.json").read_bytes() == metrics_before
    finally:
        path.write_bytes(original)


def test_rerunning_gives_the_same_results(compared):
    paths, _, _ = compared
    before = _metrics(paths)
    forecasts = (paths.artifacts / "forecasts.csv").read_bytes()
    text = (paths.artifacts / "comparison.md").read_text(encoding="utf-8")
    run_comparison(paths)
    after = _metrics(paths)
    for model in after["comparison"]["models"].values():
        model.pop("training_seconds")
    for model in before["comparison"]["models"].values():
        model.pop("training_seconds")
    assert after == before
    assert (paths.artifacts / "forecasts.csv").read_bytes() == forecasts
    assert (paths.artifacts / "comparison.md").read_text(encoding="utf-8").split("## All the scores")[0] == text.split("## All the scores")[0]


def test_the_ladder_runs_from_simplest_to_most_complex_and_covers_every_model(compared):
    _, comparison, _ = compared
    assert tuple(comparison["ladder"]) == LADDER == (
        "naive", "vix_implied", "har", "garch", "mlp", "lstm", "transformer_nosent", "transformer",
    )
    assert set(comparison["models"]) == set(LADDER)


def test_seed_spread_is_reported_for_neural_models_only(compared):
    _, comparison, _ = compared
    for name in ("mlp", "lstm", "transformer", "transformer_nosent"):
        seeds = comparison["models"][name]["seed_scores"]
        assert len(seeds["mse_log_rv"]) == 3
        assert seeds["mse_mean"] == pytest.approx(np.mean(seeds["mse_log_rv"]))
        assert seeds["qlike_spread"] == pytest.approx(np.std(seeds["qlike"], ddof=1))
    for name in ("naive", "vix_implied", "har", "garch"):
        assert comparison["models"][name]["seed_scores"] is None


def test_parameter_counts_and_training_times(compared):
    _, comparison, _ = compared
    models = comparison["models"]
    assert models["naive"]["n_parameters"] == 0 and models["naive"]["training_seconds"] is None
    assert models["har"]["n_parameters"] == 4
    assert models["garch"]["n_parameters"] == 3
    assert all(models[name]["training_seconds"] > 0 for name in ("garch", "mlp", "lstm", "transformer"))


def test_dm_covers_the_transformer_against_every_other_model_on_both_losses(compared):
    paths, comparison, _ = compared
    pairs = {(t["model_a"], t["model_b"], t["loss"]) for t in comparison["diebold_mariano"]}
    for other in LADDER:
        if other != "transformer":
            assert ("transformer", other, "squared_error_log_rv") in pairs
            assert ("transformer", other, "qlike") in pairs
    table = read_table(paths.artifacts / "forecasts.csv")
    test = table["split"].astype(str) == "test"
    actual = table["actual_log_rv"][test].astype(float)
    entry = next(t for t in comparison["diebold_mariano"] if t["model_b"] == "lstm" and t["model_a"] == "transformer" and t["loss"] == "qlike")
    expected = diebold_mariano(qlike_losses(actual, table["transformer"][test].astype(float)), qlike_losses(actual, table["lstm"][test].astype(float)), 5)
    assert entry["statistic"] == pytest.approx(expected["statistic"])
    assert entry["p_value"] == pytest.approx(expected["p_value"])
    assert entry["lower_loss"] == ("transformer" if expected["mean_loss_difference"] < 0 else "lstm")


def test_existing_dm_results_are_reproduced_by_the_new_table(compared):
    paths, comparison, _ = compared
    new = {(t["model_a"], t["model_b"], t["loss"]): t for t in comparison["diebold_mariano"]}
    for old in _metrics(paths)["test"]["diebold_mariano"]:
        assert new[(old["model_a"], old["model_b"], old["loss"])]["statistic"] == pytest.approx(old["statistic"], rel=1e-9)


def test_regimes_split_the_test_days_on_information_known_in_advance(compared):
    paths, comparison, _ = compared
    regimes = comparison["regimes"]
    assert [regime["name"] for regime in regimes] == ["calm", "middle", "volatile"]
    assert sum(regime["n_days"] for regime in regimes) == comparison["n_days"]
    assert regimes[0]["annualised_vol_pct_range"][1] <= regimes[2]["annualised_vol_pct_range"][0]


def test_verdict_names_every_model_and_says_plainly_if_it_loses(compared):
    paths, comparison, _ = compared
    text = (paths.artifacts / "comparison.md").read_text(encoding="utf-8")
    for label in ("Naive", "VIX-implied", "HAR-RV", "GARCH(1,1)", "MLP", "LSTM", "Transformer, no news"):
        assert label in text
    metrics = _metrics(paths)["test"]["models"]
    lost = [
        name for name in LADDER if name != "transformer"
        for key in ("mse_log_rv", "qlike") if metrics[name][key] < metrics["transformer"][key]
    ]
    assert ("It loses to a simpler model" in text) == bool(lost)
    assert "beats every simpler model" in text or lost
    assert "One seed at a time, the mean MSE" in text and "One seed at a time, the mean QLIKE" in text


def test_figures_exist(compared):
    paths, _, _ = compared
    for name in ("rolling_qlike", "model_ladder"):
        assert (paths.artifacts / "figures" / f"{name}.png").stat().st_size > 0
    assert ROLLING_WINDOW == 63


def test_build_rows_matches_the_saved_new_columns(compared):
    paths, _, _ = compared
    rows = build_rows(paths)
    table = read_table(paths.artifacts / "forecasts.csv")
    for name in NEW:
        np.testing.assert_allclose(rows.forecasts[name], table[name].astype(float), rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", NEW)
def test_the_api_serves_the_new_models_with_no_other_changes(compared, name):
    paths, _, _ = compared
    with TestClient(create_app(paths)) as client:
        summary = client.get("/api/summary").json()
        assert name in summary["metrics"]["test"]["models"]
        assert "comparison" in summary["metrics"]
        body = client.get("/api/forecasts", params={"models": name}).json()
        assert body["models"] == [name]
        assert name in body["rows"][0]
        assert f"var99_{name}" in body["rows"][0]
        var = client.get("/api/var", params={"model": name}).json()
        assert var["model"] == name
        assert var["kupiec"]["breaches"] == summary["metrics"]["test"]["models"][name]["var99"]["breaches"]
        assert sum(point["breach"] for point in var["series"]) == var["kupiec"]["breaches"]
        assert name in client.get("/api/health").json()["models"]


def test_append_columns_replaces_on_a_rerun_and_rejects_a_wrong_length(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("date,a\n2021-01-01,0.1234567890123456789\n2021-01-02,2\n", encoding="utf-8")
    append_columns(path, {"b": np.array([1.5, 2.5])})
    assert path.read_text(encoding="utf-8").splitlines() == ["date,a,b", "2021-01-01,0.1234567890123456789,1.5", "2021-01-02,2,2.5"]
    append_columns(path, {"b": np.array([3, 4])})
    assert path.read_text(encoding="utf-8").splitlines() == ["date,a,b", "2021-01-01,0.1234567890123456789,3", "2021-01-02,2,4"]
    with pytest.raises(ValueError, match="has 1 values"):
        append_columns(path, {"c": np.array([1.0])})
