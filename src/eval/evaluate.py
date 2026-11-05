"""Score the frozen models. This is the only module that makes forecasts for the test set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from src.config import RunPaths
from src.data.loaders import read_table, write_table
from src.eval.metrics import (
    DM_LAGS,
    KUPIEC_CRITICAL,
    diebold_mariano,
    qlike_losses,
    score_model,
    squared_errors,
    var99,
    var_breaches,
    var_diagnostics,
)
from src.eval.references import har_fit, har_forecast, naive_forecast, vix_implied_forecast
from src.models.dataset import load_dataset
from src.models.train import SEEDS, load_checkpoint, predict

MODELS = ("transformer", "transformer_nosent", "har", "naive", "vix_implied")
LATER_MODELS = ("transformer_nosent", "har", "naive", "vix_implied")  # the news archive ends in June 2023
DM_PAIRS = (("transformer", "har"), ("transformer", "naive"), ("transformer", "transformer_nosent"))
LATER_DM_PAIRS = (("transformer_nosent", "har"), ("transformer_nosent", "naive"))


@dataclass(frozen=True)
class Evaluation:
    """One row per forecast. dates[i] is the day it was made and target_dates[i] the day it is for."""

    dates: np.ndarray
    target_dates: np.ndarray
    labels: np.ndarray
    actual: np.ndarray  # log RV of the target date
    return_oc: np.ndarray  # ln(close / open) of the target date
    forecasts: dict[str, np.ndarray]  # log RV; transformer is NaN for 'later' rows
    seed_forecasts: dict[str, np.ndarray]  # model -> (n_seeds, n_rows)
    test_attention: np.ndarray  # (n_test, 22, 22), averaged over heads and seeds
    metrics: dict


def verify_frozen(paths: RunPaths) -> None:
    """Refuse to score if any checkpoint differs from the one recorded at the end of training."""
    record = json.loads((paths.artifacts / "frozen_models.json").read_text(encoding="utf-8"))
    for name, digest in record.items():
        if hashlib.sha256((paths.artifacts / "models" / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Checkpoint {name} has changed since training was frozen")


def _variant_forecasts(
    paths: RunPaths, name: str, with_sentiment: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dates, split labels, forecasts (n_seeds, n) and test attention averaged over seeds."""
    data = load_dataset(paths, with_sentiment)
    windows = list(data.splits.items())
    dates = np.concatenate([w.dates for _, w in windows])
    labels = np.concatenate([np.full(w.dates.size, split) for split, w in windows])
    x = np.concatenate([w.x for _, w in windows])
    order = np.argsort(dates)
    dates, labels, x = dates[order], labels[order], x[order]

    forecasts, attention = [], []
    for seed in SEEDS:
        model, meta = load_checkpoint(paths.artifacts / "models" / f"{name}_seed{seed}.npz")
        if meta["feature_names"] != data.feature_names or not np.allclose(meta["scaler_mean"], data.scaler.mean):
            raise RuntimeError(f"{name} seed {seed} was trained on different features or scaling")
        standardised, weights = predict(model, x)
        forecasts.append(data.to_log_rv(standardised)[:, 0])
        attention.append(weights[labels == "test"])
    return dates, labels, np.array(forecasts), np.mean(attention, axis=0)


def _feature_rows(paths: RunPaths) -> dict[str, np.ndarray]:
    tables = [read_table(paths.processed / "daily_features.csv")]
    later = paths.processed / "daily_features_later.csv"
    if later.exists():
        tables.append(read_table(later))
    columns = ("date", "log_rv", "log_rv_w", "log_rv_m", "log_vix", "target_log_rv", "split")
    return {column: np.concatenate([table[column] for table in tables]) for column in columns}


def _target_day(paths: RunPaths, dates: np.ndarray, target_log_rv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The next session's date and open-to-close return, after checking it is the target's day."""
    bars = read_table(paths.processed / "daily_bars.csv")
    next_row = np.searchsorted(bars["date"], dates) + 1
    log_rv = np.log(bars["rv"][next_row].astype(np.float64))
    if not np.allclose(log_rv, target_log_rv, rtol=0.0, atol=1e-9):
        raise RuntimeError("Targets don't match the next session's realized variance: rows are misaligned")
    opens, closes = bars["open"][next_row].astype(np.float64), bars["close"][next_row].astype(np.float64)
    return bars["date"][next_row], np.log(closes / opens)


def build_evaluation_rows(paths: RunPaths) -> tuple[Evaluation, np.ndarray]:
    """Forecasts from every model, aligned by date. Also returns the HAR coefficients."""
    dates, labels, nosent, _ = _variant_forecasts(paths, "transformer_nosent", False)
    with_dates, _, full, attention = _variant_forecasts(paths, "transformer", True)
    in_main_window = labels != "later"
    if not np.array_equal(with_dates, dates[in_main_window]):
        raise RuntimeError("The two Transformer variants cover different dates")

    rows = _feature_rows(paths)
    position = np.searchsorted(rows["date"], dates)
    har_inputs = np.column_stack([rows[c].astype(np.float64) for c in ("log_rv", "log_rv_w", "log_rv_m")])
    target = rows["target_log_rv"].astype(np.float64)
    is_train = rows["split"].astype(str) == "train"
    coefficients = har_fit(har_inputs[is_train], target[is_train])

    target_dates, return_oc = _target_day(paths, dates, target[position])
    full_forecast = np.full((len(SEEDS), dates.size), np.nan)
    full_forecast[:, in_main_window] = full
    forecasts = {
        "transformer": full_forecast.mean(axis=0),
        "transformer_nosent": nosent.mean(axis=0),
        "har": har_forecast(har_inputs[position], coefficients),
        "naive": naive_forecast(har_inputs[position, 0]),
        "vix_implied": vix_implied_forecast(rows["log_vix"].astype(np.float64)[position]),
    }
    seed_forecasts = {"transformer": full_forecast, "transformer_nosent": nosent}
    evaluation = Evaluation(
        dates, target_dates, labels.astype(str), target[position], return_oc, forecasts, seed_forecasts, attention, {}
    )
    return evaluation, coefficients


def _score_period(evaluation: Evaluation, split: str, models: tuple[str, ...], pairs: tuple[tuple[str, str], ...]) -> dict:
    chosen = evaluation.labels == split
    actual, return_oc = evaluation.actual[chosen], evaluation.return_oc[chosen]
    tests = []
    for a, b in pairs:
        forecast_a, forecast_b = evaluation.forecasts[a][chosen], evaluation.forecasts[b][chosen]
        for loss_name, loss in (("squared_error_log_rv", squared_errors), ("qlike", qlike_losses)):
            result = diebold_mariano(loss(actual, forecast_a), loss(actual, forecast_b), DM_LAGS)
            tests.append({"model_a": a, "model_b": b, "loss": loss_name, **result})
    return {
        "forecast_dates": [str(evaluation.dates[chosen][0]), str(evaluation.dates[chosen][-1])],
        "target_dates": [str(evaluation.target_dates[chosen][0]), str(evaluation.target_dates[chosen][-1])],
        "n_days": int(chosen.sum()),
        "models": {name: score_model(actual, evaluation.forecasts[name][chosen], return_oc) for name in models},
        "seed_mse_log_rv": {
            name: [float(squared_errors(actual, row[chosen]).mean()) for row in rows]
            for name, rows in evaluation.seed_forecasts.items()
            if name in models
        },
        "diebold_mariano": tests,
        "var_diagnostics": var_diagnostics(actual, return_oc),
    }


def write_json(path: Path, content: dict) -> None:
    """allow_nan=False: a NaN would be invalid JSON and break the frontend, so fail here instead."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")


def _write_tables(paths: RunPaths, evaluation: Evaluation) -> None:
    """forecasts.csv holds the June 2023 window. The later check stays in its own file."""
    forecasts, labels = evaluation.forecasts, evaluation.labels
    main = labels != "later"
    write_table(paths.artifacts / "forecasts.csv", {
        "date": evaluation.dates[main],
        "actual_log_rv": evaluation.actual[main],
        **{name: forecasts[name][main] for name in MODELS},
        "return_oc": evaluation.return_oc[main],
        "var99_transformer": var99(forecasts["transformer"][main]),
        "breach_transformer": var_breaches(forecasts["transformer"][main], evaluation.return_oc[main]).astype(int),
        "split": labels[main],
        "target_date": evaluation.target_dates[main],
    })

    test = labels == "test"
    var_columns: dict[str, np.ndarray] = {
        "date": evaluation.dates[test],
        "target_date": evaluation.target_dates[test],
        "return_oc": evaluation.return_oc[test],
    }
    for name in MODELS:
        var_columns[f"var99_{name}"] = var99(forecasts[name][test])
        var_columns[f"breach_{name}"] = var_breaches(forecasts[name][test], evaluation.return_oc[test]).astype(int)
    write_table(paths.artifacts / "var_series.csv", var_columns)

    later = labels == "later"
    if later.any():
        write_table(paths.artifacts / "forecasts_later.csv", {
            "date": evaluation.dates[later],
            "actual_log_rv": evaluation.actual[later],
            **{name: forecasts[name][later] for name in LATER_MODELS},
            "return_oc": evaluation.return_oc[later],
            "split": labels[later],
            "target_date": evaluation.target_dates[later],
        })


def _print_period(title: str, period: dict) -> None:
    print(f"\n{title}: {period['n_days']} forecasts for {period['target_dates'][0]} to {period['target_dates'][1]}")
    print(f"  {'model':20s} {'MSE':>8s} {'QLIKE':>8s} {'bias':>8s} {'breaches':>9s} {'expected':>9s} {'Kupiec LR':>10s}")
    for name, score in period["models"].items():
        var = score["var99"]
        print(
            f"  {name:20s} {score['mse_log_rv']:8.4f} {score['qlike']:8.4f} {score['mean_error_log_rv']:8.3f} "
            f"{var['breaches']:9d} {var['expected_breaches']:9.1f} {var['kupiec_lr']:9.2f}"
            f"{'*' if var['kupiec_rejected'] else ' '}"
        )
    checks = period["var_diagnostics"]
    print(f"  Mean squared open-to-close return / mean RV: {checks['mean_squared_return_over_mean_rv']:.3f}. "
          f"Breaches if RV were known exactly: {checks['breaches_if_rv_were_known']}")
    print("  Diebold-Mariano (negative statistic: the first model had the lower loss):")
    for test in period["diebold_mariano"]:
        print(
            f"    {test['model_a']} vs {test['model_b']}, {test['loss']}: "
            f"statistic {test['statistic']:.2f}, p = {test['p_value']:.4f}"
        )


def run_evaluation(paths: RunPaths) -> Evaluation:
    """Check the freeze, forecast, score and write metrics.json and the forecast tables."""
    print(f"\n=== Evaluation ({paths.data_source} data) ===")
    verify_frozen(paths)
    evaluation, coefficients = build_evaluation_rows(paths)
    training_log = json.loads((paths.artifacts / "training_log.json").read_text(encoding="utf-8"))
    metrics = {
        "data_source": paths.data_source,
        "test": _score_period(evaluation, "test", MODELS, DM_PAIRS),
        "validation_mse_log_rv": {
            **{name: variant["ensemble_val_mse_log_rv"] for name, variant in training_log["variants"].items()},
            **training_log["validation_references_mse_log_rv"],
        },
        "har_coefficients": dict(zip(("intercept", "log_rv", "log_rv_w", "log_rv_m"), coefficients.tolist())),
        "dm_lags": DM_LAGS,
        "kupiec_critical_value": KUPIEC_CRITICAL,
    }
    if (evaluation.labels == "later").any():
        metrics["later_check"] = {
            **_score_period(evaluation, "later", LATER_MODELS, LATER_DM_PAIRS),
            "excluded": "The with-sentiment Transformer: the news archive ends on 2023-06-30 and news is never filled in.",
        }
    write_json(paths.artifacts / "metrics.json", metrics)
    _write_tables(paths, evaluation)
    _print_period("Test period", metrics["test"])
    if "later_check" in metrics:
        _print_period("Later check", metrics["later_check"])
    return replace(evaluation, metrics=metrics)
