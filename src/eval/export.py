"""Precomputed files for the frontend: one JSON per test date, plus the model card."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from src.config import ALL_FEATURES, RunPaths
from src.data.loaders import read_table
from src.eval.evaluate import MODELS, Evaluation, write_json
from src.eval.metrics import var99, var_breaches
from src.eval.references import TRADING_DAYS
from src.models.train import BATCH_SIZE, CLIP_NORM, LEARNING_RATE, MAX_EPOCHS, PATIENCE, SEEDS, WARMUP_EPOCHS, WEIGHT_DECAY
from src.models.transformer import D_MODEL, DROPOUT, N_HEADS, WINDOW_DAYS
from src.nn.gradcheck import TOLERANCE, run_all_checks

ATTENTION_DECIMALS = 5


def annualised_vol_pct(log_rv: float) -> float:
    return float(np.sqrt(TRADING_DAYS * np.exp(log_rv)) * 100.0)


def daily_move_pct(log_rv: float) -> float:
    """One standard deviation of the open-to-close return, in percent."""
    return float(np.sqrt(np.exp(log_rv)) * 100.0)


def _headlines_by_day(paths: RunPaths) -> dict:
    table = read_table(paths.processed / "headline_extremes.csv")
    days: dict = defaultdict(lambda: {"negative": [], "positive": []})
    for date, side, score, text in zip(table["date"], table["side"], table["score"], table["headline"]):
        days[date][str(side)].append({"headline": str(text), "score": round(float(score), 4)})
    return days


def write_day_details(paths: RunPaths, evaluation: Evaluation) -> int:
    """day_details/<date>.json, where <date> is the day t the forecast was made."""
    features = read_table(paths.processed / "daily_features.csv")
    all_dates = features["date"]
    headlines = _headlines_by_day(paths)
    test_rows = np.flatnonzero(evaluation.labels == "test")
    forecast = evaluation.forecasts["transformer"]

    for row, attention in zip(test_rows, evaluation.test_attention):
        date = evaluation.dates[row]
        position = int(np.searchsorted(all_dates, date))
        day_headlines = headlines.get(date, {"negative": [], "positive": []})
        write_json(paths.artifacts / "day_details" / f"{date}.json", {
            "date": str(date),
            "target_date": str(evaluation.target_dates[row]),
            "data_source": paths.data_source,
            "features": {name: float(features[name][position]) for name in ALL_FEATURES},
            "input_dates": [str(d) for d in all_dates[position - WINDOW_DAYS + 1 : position + 1]],
            "attention": {
                "matrix": np.round(attention, ATTENTION_DECIMALS).tolist(),
                "mean_over_input_days": np.round(attention.mean(axis=0), ATTENTION_DECIMALS).tolist(),
                "last_day_row": np.round(attention[-1], ATTENTION_DECIMALS).tolist(),
                "note": "Row = the day paying attention, column = the day attended to, oldest first. "
                        "Only the last row feeds the forecast directly, because the model pools on the last day.",
            },
            "headlines": day_headlines,
            "forecast": {
                "log_rv": float(forecast[row]),
                "annualised_vol_pct": annualised_vol_pct(forecast[row]),
                "expected_daily_move_pct": daily_move_pct(forecast[row]),
                "var99_pct": float(var99(forecast[row]) * 100.0),
                "all_models_log_rv": {name: float(evaluation.forecasts[name][row]) for name in MODELS},
            },
            "actual": {
                "log_rv": float(evaluation.actual[row]),
                "annualised_vol_pct": annualised_vol_pct(evaluation.actual[row]),
                "daily_move_pct": daily_move_pct(evaluation.actual[row]),
                "return_oc_pct": float(evaluation.return_oc[row] * 100.0),
                "var_breached": bool(var_breaches(forecast[row], evaluation.return_oc[row])),
            },
        })
    return test_rows.size


def feature_train_stats(paths: RunPaths) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of each feature over training rows only, for "today vs usual"."""
    features = read_table(paths.processed / "daily_features.csv")
    train = features["split"].astype(str) == "train"
    return {
        name: {"mean": float(features[name][train].mean()), "std": float(features[name][train].std())}
        for name in ALL_FEATURES
    }


def gradcheck_summary() -> dict:
    checks = run_all_checks()
    return {
        "checks": len(checks),
        "passed": sum(c.passed for c in checks),
        "worst_relative_error": max(c.relative_error for c in checks),
        "tolerance": TOLERANCE,
        "results": [
            {"case": c.case, "array": c.array, "shape": list(c.shape), "relative_error": c.relative_error, "passed": c.passed}
            for c in checks
        ],
    }


def write_model_card(paths: RunPaths) -> dict:
    training = json.loads((paths.artifacts / "training_log.json").read_text(encoding="utf-8"))
    sentiment = json.loads((paths.processed / "sentiment_metrics.json").read_text(encoding="utf-8"))
    variants = training["variants"]
    card = {
        "data_source": paths.data_source,
        "architecture": {
            "layers": [
                "Dense(n_features -> 32)",
                "Learned positional embedding (22 x 32)",
                "Pre-norm Transformer encoder block (d=32, 2 heads, FFN 64, dropout 0.1)",
                "Last-day pooling",
                "Dense(32 -> 1)",
            ],
            "window_days": WINDOW_DAYS,
            "d_model": D_MODEL,
            "heads": N_HEADS,
            "dropout": DROPOUT,
            "features": {name: variant["features"] for name, variant in variants.items()},
            "departures_from_brief": [
                "18 and 15 input features rather than 17 and 14, because the weekday one-hot has 5 columns.",
                "Last-day pooling rather than mean pooling, chosen on validation data before the test set was opened.",
                "Weekly expiry flag: Thursdays to 2025-08-28 and Tuesdays after, including pre-2019 Thursdays as a project choice.",
            ],
        },
        "feature_train_stats": feature_train_stats(paths),
        "parameter_count": {name: variant["n_parameters"] for name, variant in variants.items()},
        "training": {
            "seconds_all_models": training["training_seconds"],
            "seeds": list(SEEDS),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_epochs": WARMUP_EPOCHS,
            "clip_norm": CLIP_NORM,
            "patience": PATIENCE,
            "max_epochs": MAX_EPOCHS,
            "best_epochs": {name: [run["best_epoch"] for run in variant["seed_runs"]] for name, variant in variants.items()},
        },
        "gradcheck": gradcheck_summary(),
        "sentiment_classifier": {
            "holdout_accuracy": sentiment["holdout"]["accuracy"],
            "holdout_macro_f1": sentiment["holdout"]["macro_f1"],
            "vocabulary_size": sentiment["vocabulary_size"],
            "note": "Measured on Financial PhraseBank. Accuracy on Indian news headlines was not measured.",
        },
    }
    write_json(paths.artifacts / "model_card.json", card)
    return card


def run_export(paths: RunPaths, evaluation: Evaluation) -> None:
    n_days = write_day_details(paths, evaluation)
    card = write_model_card(paths)
    print(
        f"\nExported {n_days} day files and the model card "
        f"(gradcheck {card['gradcheck']['passed']}/{card['gradcheck']['checks']}) to {paths.artifacts}"
    )
