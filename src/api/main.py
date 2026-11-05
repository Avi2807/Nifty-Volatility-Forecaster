"""Read-only API over the precomputed artifacts. Nothing is trained or scored at request time.

Run with: uvicorn src.api.main:app --port 8000

Everything is loaded once at startup, so a missing or mismatched file stops the server with a
clear message instead of failing on a later request. The exceptions are /live and /forward, which read
their files per request so that a new forecast appears without a restart. In every row, `date` is the day t
the forecast was made and `target_date` is the session it is for.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from src.config import REAL_PATHS, RunPaths
from src.data.loaders import read_table
from src.eval.metrics import VAR_MULTIPLIER, var99, var_breaches

FRONTEND_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
Row = dict[str, object]


@dataclass(frozen=True)
class Store:
    """The artifacts, ready to serve. Dates are ISO strings, which sort like the dates they name."""

    data_source: str
    metrics: dict
    model_card: dict
    models: tuple[str, ...]
    forecast_rows: list[Row]
    forecast_dates: list[str]
    var_series: dict[str, list[Row]]
    sentiment_rows: list[Row]
    sentiment_dates: list[str]
    later_rows: list[Row]
    later_dates: list[str]
    test_dates: list[str]
    day_folder: Path


def _rows(table: dict[str, np.ndarray]) -> list[Row]:
    """Turn NumPy columns into JSON-ready rows. A missing value becomes null, since NaN isn't valid JSON."""
    columns: dict[str, list] = {}
    for name, values in table.items():
        if values.dtype.kind == "M":
            columns[name] = [str(value) for value in values]
        else:
            columns[name] = [None if isinstance(v, float) and math.isnan(v) else v for v in values.tolist()]
    return [dict(zip(columns, row)) for row in zip(*columns.values())]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _var_series(rows: list[Row], models: tuple[str, ...]) -> dict[str, list[Row]]:
    return {
        model: [
            {
                "date": row["date"],
                "target_date": row["target_date"],
                "return_oc": row["return_oc"],
                "loss": -row["return_oc"],
                "var99": row[f"var99_{model}"],
                "breach": bool(row[f"breach_{model}"]),
            }
            for row in rows
        ]
        for model in models
    }


def _with_index_close(rows: list[Row], path: Path) -> list[Row]:
    """Add NSE's published close on `date`. It's null on a day with no figure, and in a run without the file."""
    closes: dict[str, float] = {}
    if path.exists():
        table = read_table(path)
        closes = {str(day): float(close) for day, close in zip(table["date"], table["close"])}
    return [{**row, "nifty_close": closes.get(str(row["date"]))} for row in rows]


def _later_rows(path: Path, closes: Path) -> list[Row]:
    """The later check's rows, with the no-news model's VaR by the same rule as its score."""
    table = read_table(path)
    forecast = table["transformer_nosent"].astype(np.float64)
    return _with_index_close(
        _rows({
            **table,
            "var99_transformer_nosent": var99(forecast),
            "breach_transformer_nosent": var_breaches(forecast, table["return_oc"].astype(np.float64)).astype(int),
        }),
        closes,
    )


def load_store(paths: RunPaths) -> Store:
    required = [
        paths.artifacts / "metrics.json",
        paths.artifacts / "model_card.json",
        paths.artifacts / "forecasts.csv",
        paths.artifacts / "var_series.csv",
        paths.processed / "sentiment_daily.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing artifacts: {missing}. Run `python run_all.py` first.")

    metrics, model_card = _read_json(required[0]), _read_json(required[1])
    for name, content in (("metrics.json", metrics), ("model_card.json", model_card)):
        if content["data_source"] != paths.data_source:
            raise RuntimeError(f"{name} holds {content['data_source']} results, but this server serves {paths.data_source} data")

    forecast_rows = _with_index_close(_rows(read_table(paths.artifacts / "forecasts.csv")), paths.processed / "index_closes.csv")
    sentiment_rows = _rows(read_table(paths.processed / "sentiment_daily.csv"))
    test_dates = [str(row["date"]) for row in forecast_rows if row["split"] == "test"]
    day_folder = paths.artifacts / "day_details"
    missing_days = [day for day in test_dates if not (day_folder / f"{day}.json").exists()]
    if missing_days:
        raise RuntimeError(f"{len(missing_days)} test dates have no day file, starting with {missing_days[0]}. Run `python run_all.py`.")

    later_path = paths.artifacts / "forecasts_later.csv"
    later_rows = _later_rows(later_path, paths.processed / "index_closes.csv") if later_path.exists() else []

    models = tuple(metrics["test"]["models"])
    return Store(
        data_source=paths.data_source,
        metrics=metrics,
        model_card=model_card,
        models=models,
        forecast_rows=forecast_rows,
        forecast_dates=[str(row["date"]) for row in forecast_rows],
        var_series=_var_series(_rows(read_table(paths.artifacts / "var_series.csv")), models),
        sentiment_rows=sentiment_rows,
        sentiment_dates=[str(row["date"]) for row in sentiment_rows],
        later_rows=later_rows,
        later_dates=[str(row["date"]) for row in later_rows],
        test_dates=test_dates,
        day_folder=day_folder,
    )


def _in_range(rows: list[Row], dates: list[str], start: Date | None, end: Date | None) -> list[Row]:
    """Rows dated from start to end, both inclusive. Dates must be sorted."""
    if start is not None and end is not None and start > end:
        raise HTTPException(422, f"start ({start}) is after end ({end})")
    low = bisect_left(dates, start.isoformat()) if start is not None else 0
    high = bisect_right(dates, end.isoformat()) if end is not None else len(dates)
    return rows[low:high]


def _chosen_models(models: str | None, available: tuple[str, ...]) -> list[str]:
    """Parse a comma-separated models parameter. Omitted or empty means every model."""
    wanted = [name.strip() for name in models.split(",") if name.strip()] if models else list(available)
    unknown = [name for name in wanted if name not in available]
    if unknown:
        raise HTTPException(422, f"Unknown models {unknown}. Choose from {list(available)}")
    return wanted


def _not_a_test_date(store: Store, day: str) -> str:
    first, last = store.test_dates[0], store.test_dates[-1]
    coverage = f"Day files exist for the {len(store.test_dates)} trading days from {first} to {last}."
    try:
        Date.fromisoformat(day)
    except ValueError:
        return f"'{day}' is not a date in YYYY-MM-DD format. {coverage}"
    nearest = store.test_dates[min(bisect_left(store.test_dates, day), len(store.test_dates) - 1)]
    return f"{day} is not a test date. {coverage} The nearest test date on or after it is {nearest}."


router = APIRouter(prefix="/api")


def _store(request: Request) -> Store:
    return request.app.state.store


@router.get("/health")
def health(request: Request) -> dict:
    store = _store(request)
    return {"status": "ok", "data_source": store.data_source, "test_days": len(store.test_dates), "models": list(store.models)}


@router.get("/summary")
def summary(request: Request) -> dict:
    """The latest test-day forecast is the last one in the test period, not a live forecast."""
    store = _store(request)
    latest = _read_json(store.day_folder / f"{store.test_dates[-1]}.json")
    return {
        "data_source": store.data_source,
        "metrics": store.metrics,
        "model_card": store.model_card,
        "latest_test_forecast": {
            "date": latest["date"],
            "target_date": latest["target_date"],
            "forecast": latest["forecast"],
            "actual": latest["actual"],
        },
    }


@router.get("/forecasts")
def forecasts(
    request: Request,
    start: Date | None = Query(None, description="First forecast date, inclusive"),
    end: Date | None = Query(None, description="Last forecast date, inclusive"),
    models: str | None = Query(None, description="Comma-separated model names. Omit for all"),
) -> dict:
    """Rows from forecasts.csv plus `nifty_close`, for every split. Train rows are in-sample, so check `split`."""
    store = _store(request)
    wanted = _chosen_models(models, store.models)
    dropped = set(store.models) - set(wanted)
    rows = _in_range(store.forecast_rows, store.forecast_dates, start, end)
    return {
        "data_source": store.data_source,
        "models": wanted,
        "count": len(rows),
        "rows": [{key: value for key, value in row.items() if key not in dropped} for row in rows],
    }


@router.get("/later")
def later_check(
    request: Request,
    start: Date | None = Query(None, description="First forecast date, inclusive"),
    end: Date | None = Query(None, description="Last forecast date, inclusive"),
    models: str | None = Query(None, description="Comma-separated model names. Omit for all"),
) -> dict:
    """Window B rows and scores, kept apart from the main results. The news model isn't in them."""
    store = _store(request)
    if not store.later_rows:
        raise HTTPException(404, "This run has no later check. It needs data after the study window.")
    summary = store.metrics["later_check"]
    available = tuple(summary["models"])
    wanted = _chosen_models(models, available)
    dropped = set(available) - set(wanted)
    rows = _in_range(store.later_rows, store.later_dates, start, end)
    return {
        "data_source": store.data_source,
        "models": wanted,
        "count": len(rows),
        "summary": summary,
        "rows": [{key: value for key, value in row.items() if key not in dropped} for row in rows],
    }


@router.get("/live")
def live_forecast(request: Request) -> dict:
    """The latest next-session forecast, written by `python -m src.live`. It has no outcome yet and is in no score."""
    path = request.app.state.paths.artifacts / "live_forecast.json"
    if not path.exists():
        raise HTTPException(404, "No live forecast has been made yet. Run: python -m src.live")
    return _read_json(path)


@router.get("/forward")
def forward_record(request: Request) -> dict:
    """Every forecast since the study, scored apart from it, with the simpler models' later-check scores."""
    artifacts = request.app.state.paths.artifacts
    path = artifacts / "forward_record.json"
    if not path.exists():
        raise HTTPException(404, "The forward record has no rows yet. Run: python -m src.live")
    challengers = artifacts / "challengers.json"
    return {**_read_json(path), "challengers": _read_json(challengers) if challengers.exists() else None}


@router.get("/day/{day}")
def day_details(request: Request, day: str) -> dict:
    # The file name comes from the known list of test dates, never directly from the request.
    store = _store(request)
    position = bisect_left(store.test_dates, day)
    if position == len(store.test_dates) or store.test_dates[position] != day:
        raise HTTPException(404, _not_a_test_date(store, day))
    return _read_json(store.day_folder / f"{store.test_dates[position]}.json")


@router.get("/var")
def value_at_risk(request: Request, model: str = Query("transformer", description="Model name")) -> dict:
    """The VaR series for the test period, with its breaches and the Kupiec test."""
    store = _store(request)
    if model not in store.models:
        raise HTTPException(422, f"Unknown model '{model}'. Choose from {list(store.models)}")
    series = store.var_series[model]
    var = store.metrics["test"]["models"][model]["var99"]
    return {
        "data_source": store.data_source,
        "model": model,
        "multiplier": VAR_MULTIPLIER,
        "series": series,
        "breach_dates": [row["target_date"] for row in series if row["breach"]],
        "kupiec": {
            "breaches": var["breaches"],
            "expected_breaches": var["expected_breaches"],
            "breach_rate": var["breach_rate"],
            "likelihood_ratio": var["kupiec_lr"],
            "critical_value": store.metrics["kupiec_critical_value"],
            "rejected": var["kupiec_rejected"],
        },
        "diagnostics": store.metrics["test"]["var_diagnostics"],
    }


@router.get("/sentiment")
def sentiment(
    request: Request,
    start: Date | None = Query(None, description="First trading day, inclusive"),
    end: Date | None = Query(None, description="Last trading day, inclusive"),
) -> dict:
    """Daily news sentiment. Trading days with no headlines have no row. The archive ends on 2023-06-30."""
    store = _store(request)
    rows = _in_range(store.sentiment_rows, store.sentiment_dates, start, end)
    return {"data_source": store.data_source, "count": len(rows), "rows": rows}


def create_app(paths: RunPaths) -> FastAPI:
    """Build the app for one run's artifacts. Real and synthetic results are never served together."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.store = load_store(paths)
        yield

    app = FastAPI(title="Nifty Volatility Forecaster", lifespan=lifespan)
    app.state.paths = paths
    app.add_middleware(CORSMiddleware, allow_origins=FRONTEND_ORIGINS, allow_methods=["GET"], allow_headers=["*"])
    app.include_router(router)
    return app


app = create_app(REAL_PATHS)
