"""Scores and tests against worked values, then the whole evaluation on a small fake data set."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.config import ALL_FEATURES, MARKET_FEATURES, NEWS_FEATURES, RunPaths
from src.data.loaders import read_table, write_table
from src.eval.evaluate import build_evaluation_rows, run_evaluation, verify_frozen
from src.eval.export import annualised_vol_pct, daily_move_pct, run_export
from src.eval.figures import write_figures
from src.eval.metrics import (
    diebold_mariano,
    kupiec_lr,
    newey_west_variance,
    qlike_losses,
    score_model,
    squared_errors,
    var99,
    var_breaches,
    var_diagnostics,
)
from src.models.dataset import load_dataset
from src.models.train import SEEDS, VARIANTS, save_checkpoint, write_freeze_record
from src.models.transformer import TransformerForecaster


def test_qlike_is_zero_for_a_perfect_forecast():
    assert qlike_losses(np.array([-9.0]), np.array([-9.0]))[0] == pytest.approx(0.0)


def test_qlike_matches_its_formula_and_punishes_under_forecasting_more():
    rv, h = math.exp(-9.0), math.exp(-10.0)
    expected = rv / h - math.log(rv / h) - 1.0
    assert qlike_losses(np.array([-9.0]), np.array([-10.0]))[0] == pytest.approx(expected)
    assert qlike_losses(np.array([-9.0]), np.array([-10.0]))[0] > qlike_losses(np.array([-9.0]), np.array([-8.0]))[0]


def test_squared_errors():
    np.testing.assert_allclose(squared_errors(np.array([1.0, 2.0]), np.array([3.0, 2.0])), [4.0, 0.0])


def test_newey_west_matches_a_hand_calculation():
    # Centred [-1.5, -0.5, 0.5, 1.5]: gamma0 = 1.25, gamma1 = 0.3125, Bartlett weight 1/2.
    assert newey_west_variance(np.array([1.0, 2.0, 3.0, 4.0]), 1) == pytest.approx(1.25 + 2 * 0.5 * 0.3125)


def test_newey_west_with_no_lags_is_the_plain_variance():
    series = np.random.default_rng(0).normal(size=50)
    assert newey_west_variance(series, 0) == pytest.approx(series.var())


def test_newey_west_includes_every_lag_up_to_the_limit():
    series = np.random.default_rng(0).normal(size=200).cumsum()
    centred = series - series.mean()
    by_definition = float(centred @ centred) / series.size + sum(
        2.0 * (1.0 - lag / 6.0) * float(centred[lag:] @ centred[:-lag]) / series.size for lag in range(1, 6)
    )
    assert newey_west_variance(series, 5) == pytest.approx(by_definition)


def test_diebold_mariano_statistic_and_p_value_on_a_worked_case():
    a, b = np.array([3.0, 1.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0, 1.0])
    # Difference [2, 0, 1, 1]: mean 1, centred [1, -1, 0, 0], gamma0 = 0.5, gamma1 = -0.25, weight 1/2.
    variance = 0.5 + 2.0 * 0.5 * -0.25
    result = diebold_mariano(a, b, lags=1)
    assert result["statistic"] == pytest.approx(1.0 / math.sqrt(variance / 4.0))
    assert result["p_value"] == pytest.approx(math.erfc(result["statistic"] / math.sqrt(2.0)))


def test_diebold_mariano_p_value_at_the_known_critical_value():
    """A statistic of 1.96 must give a two-sided p of about 0.05."""
    generator = np.random.default_rng(3)
    noise = generator.normal(size=2000)
    noise -= noise.mean()
    noise /= noise.std()
    shift = 1.959964 / math.sqrt(noise.size)
    result = diebold_mariano(noise + shift, np.zeros(noise.size), lags=0)
    assert result["statistic"] == pytest.approx(1.959964)
    assert result["p_value"] == pytest.approx(0.05, abs=1e-5)


def test_diebold_mariano_sign_shows_which_model_won():
    generator = np.random.default_rng(1)
    better, worse = generator.normal(1.0, 0.5, 300), generator.normal(2.0, 0.5, 300)
    assert diebold_mariano(better, worse)["statistic"] < -3.0
    assert diebold_mariano(worse, better)["statistic"] > 3.0
    assert diebold_mariano(better, worse)["p_value"] < 0.01


def test_diebold_mariano_of_identical_losses_finds_nothing():
    series = np.arange(10.0)
    assert diebold_mariano(series, series) == {"mean_loss_difference": 0.0, "statistic": 0.0, "p_value": 1.0}


def test_kupiec_matches_worked_values():
    assert kupiec_lr(5, 250) == pytest.approx(1.9568, abs=1e-3)
    assert kupiec_lr(0, 250) == pytest.approx(-2.0 * 250 * math.log(0.99))
    assert kupiec_lr(10, 1000) == pytest.approx(0.0)


def test_rejection_uses_the_briefs_threshold_of_3_84():
    """13 breaches in 615 days gives LR of about 5.8, which rejects at 3.84 but not at the 1% value 6.63."""
    forecast = np.full(615, math.log(0.01 ** 2))
    returns = np.zeros(615)
    returns[:13] = -0.05
    var = score_model(forecast, forecast, returns)["var99"]
    assert 3.84 < var["kupiec_lr"] < 6.63
    assert var["kupiec_rejected"] is True


def test_kupiec_rejects_too_many_and_too_few_breaches():
    assert kupiec_lr(30, 615) > 3.84
    assert kupiec_lr(0, 615) > 3.84
    assert kupiec_lr(6, 615) < 3.84


def test_var_uses_the_forecast_standard_deviation():
    np.testing.assert_allclose(var99(np.array([math.log(0.01 ** 2)])), [0.0233])


def test_only_a_loss_beyond_the_var_is_a_breach():
    forecast = np.full(4, math.log(0.01 ** 2))
    breaches = var_breaches(forecast, np.array([-0.03, -0.02, 0.03, -0.0232]))
    assert breaches.tolist() == [True, False, False, False]


def test_var_diagnostics_on_a_worked_case():
    actual = np.full(4, math.log(0.01 ** 2))
    returns = np.array([-0.03, 0.02, 0.0, 0.0])  # mean squared return 3.25e-4 against an RV of 1e-4
    checks = var_diagnostics(actual, returns)
    assert checks["mean_squared_return_over_mean_rv"] == pytest.approx(3.25)
    assert checks["breaches_if_rv_were_known"] == 1


def test_score_model_flags_kupiec_rejection():
    forecast = np.full(200, math.log(0.01 ** 2))
    returns = np.zeros(200)
    returns[:20] = -0.05
    score = score_model(forecast, forecast, returns)
    assert score["var99"]["breaches"] == 20
    assert score["var99"]["kupiec_rejected"] is True
    assert score["mse_log_rv"] == 0.0


N_MAIN, N_LATER = 150, 40


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> tuple[RunPaths, object]:
    """A small labelled fake run: untrained checkpoints are enough to test the plumbing."""
    root = tmp_path_factory.mktemp("eval")
    paths = RunPaths(root / "raw", root / "processed", root / "artifacts", True)
    paths.processed.mkdir()
    (paths.artifacts / "models").mkdir(parents=True)

    generator = np.random.default_rng(7)
    n = N_MAIN + N_LATER + 1
    dates = np.busday_offset("2021-01-01", np.arange(n), roll="forward").astype("datetime64[D]")
    log_rv = -10.0 + generator.normal(0.0, 0.5, n)
    opens = 100.0 + generator.normal(0.0, 1.0, n)
    closes = opens * np.exp(generator.normal(0.0, 0.01, n))
    write_table(paths.processed / "daily_bars.csv", {
        "date": dates, "open": opens, "high": closes + 1, "low": closes - 1, "close": closes,
        "rv": np.exp(log_rv), "n_bars": np.full(n, 75),
    })

    labels = np.array(["train"] * 80 + ["val"] * 30 + ["test"] * 40 + ["later"] * N_LATER)
    features = {name: generator.normal(size=n - 1) for name in ALL_FEATURES}
    features["log_rv"] = log_rv[:-1]
    features["log_vix"] = generator.normal(2.8, 0.1, n - 1)
    is_main = np.arange(n - 1) < N_MAIN
    for mask, filename in ((is_main, "daily_features.csv"), (~is_main, "daily_features_later.csv")):
        columns: dict[str, np.ndarray] = {"date": dates[:-1][mask]}
        for name in ALL_FEATURES:
            values = features[name][mask].copy()
            if filename.endswith("later.csv") and name in NEWS_FEATURES:
                values[:] = np.nan
            columns[name] = values
        columns["target_log_rv"] = log_rv[1:][mask]
        columns["split"] = labels[mask]
        write_table(paths.processed / filename, columns)

    write_table(paths.processed / "headline_extremes.csv", {
        "date": np.repeat(dates[140], 2), "side": np.array(["negative", "positive"]), "rank": np.array([1, 1]),
        "score": np.array([-0.5, 0.5]), "headline": np.array(["Bad news", "Good news"]),
    })
    (paths.processed / "sentiment_metrics.json").write_text(
        json.dumps({"holdout": {"accuracy": 0.8, "macro_f1": 0.7}, "vocabulary_size": 10}), encoding="utf-8"
    )

    variants = {}
    for name, with_sentiment in VARIANTS.items():
        data = load_dataset(paths, with_sentiment)
        for seed in SEEDS:
            model = TransformerForecaster(len(data.feature_names), seed)
            save_checkpoint(paths.artifacts / "models" / f"{name}_seed{seed}.npz", model, data, seed)
        runs = [{"seed": s, "best_epoch": 1, "epochs_run": 2, "val_curve": [1.0, 1.1], "seconds": 1.0} for s in SEEDS]
        variants[name] = {
            "features": list(data.feature_names), "n_parameters": model.n_parameters(),
            "seed_runs": runs, "ensemble_val_mse_log_rv": 0.5,
        }
    (paths.artifacts / "training_log.json").write_text(json.dumps({
        "variants": variants, "training_seconds": 1.0, "validation_references_mse_log_rv": {"har": 0.4, "naive": 0.4},
    }), encoding="utf-8")
    write_freeze_record(paths)

    evaluation = run_evaluation(paths)
    write_figures(paths, evaluation)
    run_export(paths, evaluation)
    return paths, evaluation


def test_forecast_rows_are_for_the_next_session_with_its_own_return(run):
    paths, _ = run
    bars = read_table(paths.processed / "daily_bars.csv")
    table = read_table(paths.artifacts / "forecasts.csv")
    position = np.searchsorted(bars["date"], table["date"])
    target_dates = np.array(table["target_date"], dtype="datetime64[D]")
    np.testing.assert_array_equal(target_dates, bars["date"][position + 1])
    np.testing.assert_allclose(table["actual_log_rv"], np.log(bars["rv"][position + 1]))
    np.testing.assert_allclose(table["return_oc"], np.log(bars["close"][position + 1] / bars["open"][position + 1]))


def test_a_misaligned_target_is_refused(run):
    paths, _ = run
    bars_file = paths.processed / "daily_bars.csv"
    original = bars_file.read_bytes()
    try:
        bars = read_table(bars_file)
        bars["rv"] = np.roll(bars["rv"], 1)
        write_table(bars_file, bars)
        with pytest.raises(RuntimeError, match="misaligned"):
            build_evaluation_rows(paths)
    finally:
        bars_file.write_bytes(original)


def test_naive_uses_today_and_the_vix_forecast_uses_the_brief_formula(run):
    paths, _ = run
    table = read_table(paths.artifacts / "forecasts.csv")
    features = read_table(paths.processed / "daily_features.csv")
    position = np.searchsorted(features["date"], table["date"])
    np.testing.assert_allclose(table["naive"], features["log_rv"][position])
    vix = np.exp(features["log_vix"][position])
    np.testing.assert_allclose(table["vix_implied"], np.log((vix / 100.0) ** 2 / 252.0))


def test_har_is_fitted_on_training_rows_only(run):
    paths, evaluation = run
    features = read_table(paths.processed / "daily_features.csv")
    train = features["split"].astype(str) == "train"
    design = np.column_stack([np.ones(train.sum()), features["log_rv"][train], features["log_rv_w"][train], features["log_rv_m"][train]])
    beta = np.linalg.lstsq(design, features["target_log_rv"][train], rcond=None)[0]
    np.testing.assert_allclose(list(evaluation.metrics["har_coefficients"].values()), beta)


def test_breach_column_agrees_with_return_and_var(run):
    paths, _ = run
    table = read_table(paths.artifacts / "forecasts.csv")
    np.testing.assert_allclose(table["var99_transformer"], 2.33 * np.sqrt(np.exp(table["transformer"])))
    np.testing.assert_array_equal(table["breach_transformer"], (-table["return_oc"] > table["var99_transformer"]).astype(int))


def test_metrics_are_reproducible_from_the_published_csv(run):
    paths, _ = run
    table = read_table(paths.artifacts / "forecasts.csv")
    test = table["split"].astype(str) == "test"
    metrics = json.loads((paths.artifacts / "metrics.json").read_text(encoding="utf-8"))
    for name, score in metrics["test"]["models"].items():
        forecast, actual = table[name][test], table["actual_log_rv"][test]
        assert score["mse_log_rv"] == pytest.approx(np.mean((forecast - actual) ** 2))
        ratio = np.exp(actual) / np.exp(forecast)
        assert score["qlike"] == pytest.approx(np.mean(ratio - np.log(ratio) - 1.0))
    assert metrics["test"]["n_days"] == int(test.sum())


def test_every_brief_comparison_is_tested_on_both_losses(run):
    _, evaluation = run
    pairs = {(t["model_a"], t["model_b"], t["loss"]) for t in evaluation.metrics["test"]["diebold_mariano"]}
    for b in ("har", "naive", "transformer_nosent"):
        assert ("transformer", b, "squared_error_log_rv") in pairs
        assert ("transformer", b, "qlike") in pairs


def test_forecasts_csv_has_the_columns_the_brief_lists_in_order(run):
    paths, _ = run
    columns = list(read_table(paths.artifacts / "forecasts.csv"))
    assert columns[:11] == [
        "date", "actual_log_rv", "transformer", "transformer_nosent", "har", "naive", "vix_implied",
        "return_oc", "var99_transformer", "breach_transformer", "split",
    ]


def test_the_later_check_is_separate_and_excludes_the_news_model(run):
    paths, _ = run
    main = read_table(paths.artifacts / "forecasts.csv")
    later = read_table(paths.artifacts / "forecasts_later.csv")
    assert "later" not in set(main["split"].astype(str))
    assert "transformer" not in later
    assert later["date"].size == N_LATER
    assert np.isfinite(later["transformer_nosent"]).all()
    metrics = json.loads((paths.artifacts / "metrics.json").read_text(encoding="utf-8"))
    assert "transformer" not in metrics["later_check"]["models"]
    assert set(metrics["later_check"]["seed_mse_log_rv"]) == {"transformer_nosent"}


def test_no_news_variant_gets_no_news_inputs(run):
    paths, _ = run
    assert load_dataset(paths, False).feature_names == MARKET_FEATURES


def test_changing_test_outcomes_cannot_change_any_forecast(run):
    """Forecasts use only the inputs, and HAR fits on training rows only."""
    paths, evaluation = run
    features_file, bars_file = paths.processed / "daily_features.csv", paths.processed / "daily_bars.csv"
    originals = {path: path.read_bytes() for path in (features_file, bars_file)}
    try:
        table = read_table(features_file)
        is_test = table["split"].astype(str) == "test"
        table["target_log_rv"][is_test] += 3.0
        write_table(features_file, table)
        bars = read_table(bars_file)
        changed_rows = np.searchsorted(bars["date"], table["date"][is_test]) + 1
        bars["rv"][changed_rows] *= math.exp(3.0)
        write_table(bars_file, bars)
        changed, _ = build_evaluation_rows(paths)
    finally:
        for path, content in originals.items():
            path.write_bytes(content)
    main = evaluation.labels != "later"
    assert not np.allclose(changed.actual[main], evaluation.actual[main])
    for name in ("transformer", "transformer_nosent", "har", "naive", "vix_implied"):
        np.testing.assert_allclose(changed.forecasts[name][main], evaluation.forecasts[name][main])


def test_a_changed_checkpoint_is_refused(run):
    paths, _ = run
    checkpoint = paths.artifacts / "models" / f"transformer_seed{SEEDS[0]}.npz"
    original = checkpoint.read_bytes()
    try:
        checkpoint.write_bytes(original + b"x")
        with pytest.raises(RuntimeError, match="changed since training"):
            verify_frozen(paths)
    finally:
        checkpoint.write_bytes(original)


def test_final_forecast_is_the_mean_over_seeds(run):
    _, evaluation = run
    for name in ("transformer", "transformer_nosent"):
        np.testing.assert_allclose(evaluation.forecasts[name], evaluation.seed_forecasts[name].mean(axis=0))
    assert evaluation.seed_forecasts["transformer"].shape[0] == len(SEEDS)


def test_scored_period_is_the_test_split_only(run):
    paths, evaluation = run
    table = read_table(paths.artifacts / "forecasts.csv")
    test_dates = table["date"][table["split"].astype(str) == "test"]
    assert evaluation.metrics["test"]["forecast_dates"] == [str(test_dates[0]), str(test_dates[-1])]


def test_var_file_covers_every_model_on_test_dates(run):
    paths, evaluation = run
    table = read_table(paths.artifacts / "var_series.csv")
    test = evaluation.labels == "test"
    for name in ("transformer", "transformer_nosent", "har", "naive", "vix_implied"):
        np.testing.assert_allclose(table[f"var99_{name}"], 2.33 * np.sqrt(np.exp(evaluation.forecasts[name][test])))
        assert int(table[f"breach_{name}"].sum()) == evaluation.metrics["test"]["models"][name]["var99"]["breaches"]


def test_one_day_file_per_test_date_with_a_consistent_window(run):
    paths, evaluation = run
    test_dates = evaluation.dates[evaluation.labels == "test"]
    assert len(list((paths.artifacts / "day_details").glob("*.json"))) == test_dates.size
    for date in test_dates:
        detail = json.loads((paths.artifacts / "day_details" / f"{date}.json").read_text(encoding="utf-8"))
        assert detail["input_dates"][-1] == str(date) and len(detail["input_dates"]) == 22
        assert detail["input_dates"] == sorted(detail["input_dates"])
        matrix = np.array(detail["attention"]["matrix"])
        assert matrix.shape == (22, 22)
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-3)
        np.testing.assert_allclose(sum(detail["attention"]["mean_over_input_days"]), 1.0, atol=1e-3)
        np.testing.assert_allclose(detail["attention"]["last_day_row"], matrix[-1])
        np.testing.assert_allclose(detail["attention"]["mean_over_input_days"], matrix.mean(axis=0), atol=1e-4)
        assert set(detail["features"]) == set(ALL_FEATURES)


def test_day_file_converts_the_forecast_correctly(run):
    paths, evaluation = run
    row = np.flatnonzero(evaluation.labels == "test")[0]
    forecast = evaluation.forecasts["transformer"][row]
    detail = json.loads((paths.artifacts / "day_details" / f"{evaluation.dates[row]}.json").read_text(encoding="utf-8"))
    assert detail["forecast"]["log_rv"] == pytest.approx(forecast)
    assert detail["forecast"]["annualised_vol_pct"] == pytest.approx(math.sqrt(252.0 * math.exp(forecast)) * 100.0)
    assert detail["forecast"]["expected_daily_move_pct"] == pytest.approx(math.sqrt(math.exp(forecast)) * 100.0)
    assert annualised_vol_pct(-10.0) == pytest.approx(daily_move_pct(-10.0) * math.sqrt(252.0))
    assert detail["actual"]["return_oc_pct"] == pytest.approx(evaluation.return_oc[row] * 100.0)
    assert detail["forecast"]["var99_pct"] == pytest.approx(2.33 * math.sqrt(math.exp(forecast)) * 100.0)
    assert detail["actual"]["log_rv"] == pytest.approx(evaluation.actual[row])
    assert detail["target_date"] == str(evaluation.target_dates[row])


def test_headlines_reach_their_own_day_and_only_that_day(run):
    paths, evaluation = run
    has_news = read_table(paths.processed / "headline_extremes.csv")["date"][0]
    detail = json.loads((paths.artifacts / "day_details" / f"{has_news}.json").read_text(encoding="utf-8"))
    assert detail["headlines"]["negative"] == [{"headline": "Bad news", "score": -0.5}]
    assert detail["headlines"]["positive"] == [{"headline": "Good news", "score": 0.5}]
    other = evaluation.dates[(evaluation.labels == "test") & (evaluation.dates != has_news)][0]
    detail = json.loads((paths.artifacts / "day_details" / f"{other}.json").read_text(encoding="utf-8"))
    assert detail["headlines"] == {"negative": [], "positive": []}


def test_model_card_holds_what_the_brief_asks_for(run):
    paths, _ = run
    card = json.loads((paths.artifacts / "model_card.json").read_text(encoding="utf-8"))
    assert card["parameter_count"]["transformer"] == TransformerForecaster(len(ALL_FEATURES), 1).n_parameters()
    assert card["gradcheck"]["passed"] == card["gradcheck"]["checks"] > 0
    assert card["gradcheck"]["worst_relative_error"] < 1e-6
    assert card["sentiment_classifier"]["holdout_accuracy"] == 0.8
    assert "Last-day pooling" in card["architecture"]["layers"]


def test_model_card_feature_averages_use_training_rows_only(run):
    paths, _ = run
    card = json.loads((paths.artifacts / "model_card.json").read_text(encoding="utf-8"))
    features = read_table(paths.processed / "daily_features.csv")
    train = features["split"].astype(str) == "train"
    assert set(card["feature_train_stats"]) == set(ALL_FEATURES)
    for name in ALL_FEATURES:
        assert card["feature_train_stats"][name]["mean"] == pytest.approx(features[name][train].mean())
        assert card["feature_train_stats"][name]["std"] == pytest.approx(features[name][train].std())


def test_every_figure_was_written(run):
    paths, _ = run
    names = {path.stem for path in (paths.artifacts / "figures").glob("*.png")}
    assert names == {
        "forecast_vs_actual", "cumulative_loss_difference", "model_errors", "var_breaches",
        "attention_heatmap", "training_curves",
    }
