"""The forward record: every forecast from the live code, kept and scored apart from the study.

`python -m src.live` adds to it. A row is "live" only if it was saved before its session opened. A row made
any later uses only data up to the evening before, but its outcome could have been known by then, so it is
"backfilled" and is never counted as live evidence. A saved forecast is never rewritten, and a model added
after a row was made stays null on that row.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, time
from pathlib import Path

import numpy as np

from src.config import MARKET_FEATURES, RunPaths
from src.data.bars import DailyBars
from src.data.features import FeatureTable
from src.data.loaders import load_index_opens, load_index_closes, read_table
from src.data.splits import Scaler
from src.eval.challengers import CHALLENGERS, NEURAL
from src.eval.metrics import VAR_MULTIPLIER, VAR_RATE, qlike_losses, squared_errors
from src.eval.references import har_fit, har_forecast, vix_implied_forecast
from src.models.comparison import model_folder, verify_comparison_frozen
from src.models.dataset import Windows
from src.models.garch import PERCENT, load_fit, load_returns, variance_path
from src.models.train import SEEDS, forecast_only, load_checkpoint, predict
from src.models.transformer import WINDOW_DAYS

RECORD_FILE = "forward_record.json"
SESSION_OPEN = time(9, 15)
RECORD_MODELS = ("transformer_nosent", "har", "naive", "vix_implied", *CHALLENGERS)
VAR_LINES = ("rule", "calibrated", "garch")
KINDS = ("live", "backfilled")


def transformer_seed_forecasts(paths: RunPaths, windows: np.ndarray) -> np.ndarray:
    """(n_seeds, n) log RV forecasts from the frozen no-news Transformer, for raw (n, 22, 15) windows."""
    forecasts = []
    for seed in SEEDS:
        model, meta = load_checkpoint(paths.artifacts / "models" / f"transformer_nosent_seed{seed}.npz")
        if meta["feature_names"] != MARKET_FEATURES:
            raise RuntimeError(f"Checkpoint seed {seed} was trained on different features")
        scaled = Scaler(meta["scaler_mean"], meta["scaler_std"]).transform(windows)
        forecasts.append(predict(model, scaled)[0][:, 0] * meta["target_std"] + meta["target_mean"])
    return np.array(forecasts)


def har_coefficients(paths: RunPaths) -> np.ndarray:
    study = read_table(paths.processed / "daily_features.csv")
    inputs = np.column_stack([study[name].astype(np.float64) for name in ("log_rv", "log_rv_w", "log_rv_m")])
    is_train = study["split"].astype(str) == "train"
    return har_fit(inputs[is_train], study["target_log_rv"].astype(np.float64)[is_train])


def _neural_forecast(paths: RunPaths, name: str, windows: np.ndarray) -> np.ndarray:
    """The mean of seeds, in log RV. Each checkpoint carries the training-period scaling it needs."""
    build, prepare, _ = NEURAL[name]
    forecasts = []
    for seed in SEEDS:
        model, meta = load_checkpoint(model_folder(paths) / f"{name}_seed{seed}.npz", build)
        if meta["feature_names"] != MARKET_FEATURES:
            raise RuntimeError(f"{name} seed {seed} was trained on different features")
        scaled = Scaler(meta["scaler_mean"], meta["scaler_std"]).transform(windows)
        x = prepare(Windows(scaled, np.empty((scaled.shape[0], 1)), np.empty(scaled.shape[0], dtype="datetime64[D]"))).x
        forecasts.append(forecast_only(model, x)[:, 0] * meta["target_std"] + meta["target_mean"])
    return np.mean(forecasts, axis=0)


def garch_return_history(paths: RunPaths, daily: DailyBars) -> tuple[np.ndarray, np.ndarray]:
    """Open-to-close returns in percent up to the last session: the study's bars, NSE's daily figures, then the live bars.

    GARCH needs an unbroken run of returns. The minute file and Yahoo's bars don't meet, and NSE's figures cover the gap.
    """
    study_dates, study_returns = load_returns(paths)
    before = study_dates < daily.dates[0]
    parts = [(study_dates[before], study_returns[before])]
    opens, closes = load_index_opens(paths.raw), load_index_closes(paths.raw)
    if opens is None or closes is None:
        print("WARNING: NSE's daily index files are missing, so GARCH runs across the gap in its returns")
    else:
        common, open_at, close_at = np.intersect1d(opens.dates, closes.dates, return_indices=True)
        in_gap = (common > study_dates[before][-1]) & (common < daily.dates[0])
        parts.append((common[in_gap], np.log(closes.values[close_at][in_gap] / opens.values[open_at][in_gap]) * PERCENT))
    parts.append((daily.dates, np.log(daily.close / daily.open) * PERCENT))
    return np.concatenate([part[0] for part in parts]), np.concatenate([part[1] for part in parts])


def garch_forecasts(paths: RunPaths, daily: DailyBars, forecast_dates: np.ndarray) -> np.ndarray:
    """Log variance, in decimal units like log RV, for the session after each forecast date."""
    fit = load_fit(paths)
    dates, returns = garch_return_history(paths, daily)
    path = variance_path(returns, fit.omega, fit.alpha, fit.beta, fit.initial_variance)
    return np.log(path[np.searchsorted(dates, forecast_dates) + 1] / PERCENT**2)


def kind_of_forecast(target_date: str, made_at: datetime) -> str:
    """Live only if it was made before the target session opened, on the IST clock."""
    opens = datetime.combine(datetime.fromisoformat(target_date).date(), SESSION_OPEN, made_at.tzinfo)
    return "live" if made_at < opens else "backfilled"


def session_rows(paths: RunPaths, table: FeatureTable, daily: DailyBars, next_session: np.datetime64, multiplier: float, now: datetime) -> list[dict]:
    """A row for every session with a full window of features, using the data up to that session only."""
    verify_comparison_frozen(paths)
    columns = [table.columns.index(name) for name in MARKET_FEATURES]
    market = table.X[:, columns]
    windows = np.lib.stride_tricks.sliding_window_view(market, WINDOW_DAYS, axis=0).transpose(0, 2, 1)
    dates = table.dates[WINDOW_DAYS - 1 :]
    latest = market[WINDOW_DAYS - 1 :]
    position = np.searchsorted(daily.dates, dates)
    targets = np.append(daily.dates, next_session)[position + 1]

    transformer = transformer_seed_forecasts(paths, windows).mean(axis=0)
    har_inputs = latest[:, [MARKET_FEATURES.index(name) for name in ("log_rv", "log_rv_w", "log_rv_m")]]
    forecasts = {
        "transformer_nosent": transformer,
        "har": har_forecast(har_inputs, har_coefficients(paths)),
        "naive": har_inputs[:, 0],
        "vix_implied": vix_implied_forecast(latest[:, MARKET_FEATURES.index("log_vix")]),
        "garch": garch_forecasts(paths, daily, dates),
        **{name: _neural_forecast(paths, name, windows) for name in NEURAL},
    }
    made_at = now.isoformat(timespec="seconds")
    rows = []
    for i, (date, target) in enumerate(zip(dates, targets)):
        rows.append({
            "date": str(date),
            "target_date": str(target),
            "made_at": made_at,
            "kind": kind_of_forecast(str(target), now),
            "forecasts": {name: float(forecasts[name][i]) for name in RECORD_MODELS},
            "var99": {
                "rule": float(VAR_MULTIPLIER * np.sqrt(np.exp(transformer[i]))),
                "calibrated": float(multiplier * np.sqrt(np.exp(transformer[i]))),
                "garch": float(VAR_MULTIPLIER * np.sqrt(np.exp(forecasts["garch"][i]))),
            },
            "actual_log_rv": None,
            "return_oc": None,
        })
    return rows


def row_from_saved_forecast(saved: dict) -> dict:
    """The row for a forecast the live code wrote earlier. It holds only what was actually saved at the time."""
    made_at = datetime.fromisoformat(saved["made_at"])
    references = saved["references"]
    return {
        "date": saved["date"],
        "target_date": saved["target_date"],
        "made_at": saved["made_at"],
        "kind": kind_of_forecast(saved["target_date"], made_at),
        "forecasts": {
            **{name: None for name in CHALLENGERS},
            "transformer_nosent": saved["forecast"],
            "har": references["har"],
            "naive": references["naive"],
            "vix_implied": references["vix_implied"],
        },
        "var99": {"rule": saved["var99"], "calibrated": None, "garch": None},
        "actual_log_rv": None,
        "return_oc": None,
    }


def merge_rows(existing: list[dict], new: list[dict]) -> list[dict]:
    """Add rows for dates not yet on file. The first forecast made for a date is the one that stays."""
    known = {row["date"] for row in existing}
    return sorted([*existing, *(row for row in new if row["date"] not in known)], key=lambda row: row["date"])


def add_outcomes(rows: list[dict], daily: DailyBars) -> list[dict]:
    """Fill in what happened, for each row whose target session is now complete."""
    sessions = {str(day): i for i, day in enumerate(daily.dates)}
    filled = []
    for row in rows:
        i = sessions.get(row["target_date"])
        if row["actual_log_rv"] is None and i is not None:
            row = {**row, "actual_log_rv": float(np.log(daily.rv[i])), "return_oc": float(np.log(daily.close[i] / daily.open[i]))}
        filled.append(row)
    return filled


def score_rows(rows: list[dict]) -> dict:
    """Running scores for one kind of row. Each model and line counts only the days it has a figure for."""
    scored = [row for row in rows if row["actual_log_rv"] is not None]
    models = {}
    for name in RECORD_MODELS:
        chosen = [row for row in scored if row["forecasts"][name] is not None]
        actual = np.array([row["actual_log_rv"] for row in chosen], dtype=np.float64)
        forecast = np.array([row["forecasts"][name] for row in chosen], dtype=np.float64)
        models[name] = {
            "n_days": len(chosen),
            "mse_log_rv": float(squared_errors(actual, forecast).mean()) if chosen else None,
            "qlike": float(qlike_losses(actual, forecast).mean()) if chosen else None,
        }
    lines = {}
    for name in VAR_LINES:
        chosen = [row for row in scored if row["var99"][name] is not None]
        lines[name] = {
            "n_days": len(chosen),
            "breaches": sum(-row["return_oc"] > row["var99"][name] for row in chosen),
            "expected_breaches": len(chosen) * VAR_RATE,
        }
    return {"n_days": len(scored), "models": models, "var99": lines}


def write_json(path: Path, content: dict) -> None:
    """Write through a temporary file, so the API can never read a half-written file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def update_record(paths: RunPaths, new_rows: list[dict], daily: DailyBars, previous: dict | None, multiplier: float, now: datetime) -> dict:
    """Add this run's rows and any outcomes now known, then rescore. `previous` is the last saved live forecast."""
    path = paths.artifacts / RECORD_FILE
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"] if path.exists() else []
    if previous is not None:
        rows = merge_rows(rows, [row_from_saved_forecast(previous)])
    rows = add_outcomes(merge_rows(rows, new_rows), daily)
    record = {
        "data_source": paths.data_source,
        "updated_at": now.isoformat(timespec="seconds"),
        "calibrated_multiplier": multiplier,
        "rule_multiplier": VAR_MULTIPLIER,
        "rows": rows,
        "scores": {kind: score_rows([row for row in rows if row["kind"] == kind]) for kind in KINDS},
    }
    write_json(path, record)
    return record
