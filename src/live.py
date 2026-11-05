"""Forecast the next session with the frozen no-news model and today's market data.

Usage: python -m src.live

Run it each evening, once NSE has published that day's closing file. Nothing is retrained: the
checkpoints are verified against their recorded checksums, then used as they are. The news model
can't do this, because its archive ends in June 2023.

The result goes to artifacts/live_forecast.json and is kept out of every score, because it has no
outcome yet. Downloads are kept in data/live/, which the main pipeline never reads, so fetched
data can't reach the study tables.

Sources
- 5-minute Nifty bars: Yahoo's unofficial chart endpoint, the last 60 days or so.
- India VIX and turnover: NSE's raw files in data/raw, then its daily closing file for later days.
- Trading holidays: NSE. A special weekend session isn't in that list and won't be predicted.
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

from src.config import MARKET_FEATURES, REAL_PATHS, ROOT, SESSION_END_MINUTE, RunPaths
from src.data.bars import daily_from_minute
from src.data.expiry import weekly_expiry_dates
from src.data.features import build_feature_table, weekday_index
from src.data.loaders import DatedSeries, MinuteData, clean_series, load_turnover, load_vix, read_table
from src.data.splits import Scaler
from src.eval.evaluate import verify_frozen
from src.eval.metrics import var99
from src.eval.references import har_fit, har_forecast, vix_implied_forecast
from src.models.transformer import WINDOW_DAYS
from src.models.train import SEEDS, load_checkpoint, predict

IST = timezone(timedelta(hours=5, minutes=30))
IST_OFFSET_SECONDS = 19800
LIVE_DIR = ROOT / "data" / "live"
BARS_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?interval=5m&range=60d"
CLOSING_FILE_URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{day}.csv"
HOLIDAYS_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}
# Yahoo's last bar was within 0.07% of NSE's close on all 59 days checked.
MAX_CLOSE_DIFFERENCE = 0.0025
LOOKAHEAD_DAYS = 14  # far enough to see the next weekly expiry past any holiday
BAND_QUANTILES = (0.1, 0.9)


class LiveDataError(Exception):
    """A required input is missing or disagrees with another source. No forecast is written."""


@dataclass(frozen=True)
class DayClose:
    """One day's Nifty 50 and India VIX figures from NSE's closing file."""

    date: np.datetime64
    nifty_close: float
    turnover_crore: float
    india_vix: float


def _fetch(url: str) -> bytes:
    with urlopen(Request(url, headers=HEADERS), timeout=30) as response:
        return response.read()


def _day_text(day: np.datetime64, layout: str) -> str:
    return day.astype("datetime64[D]").item().strftime(layout)


def parse_bars(payload: bytes) -> MinuteData:
    """Yahoo's chart JSON as bars on the IST wall clock, each stamped with its start, like the minute file."""
    result = json.loads(payload)["chart"]["result"][0]
    if result["meta"]["gmtoffset"] != IST_OFFSET_SECONDS:
        raise LiveDataError(f"Yahoo's bars are at UTC offset {result['meta']['gmtoffset']}s, expected IST")
    stamps = (np.array(result["timestamp"], dtype=np.int64) + IST_OFFSET_SECONDS).astype("datetime64[s]")
    quote = result["indicators"]["quote"][0]
    # A null price becomes NaN, and daily_from_minute drops that bar.
    prices = [np.array(quote[name], dtype=np.float64) for name in ("open", "high", "low", "close")]
    return MinuteData(stamps.astype("datetime64[m]"), *prices)


def complete_sessions_only(minute: MinuteData, now: datetime) -> MinuteData:
    """Drop today's bars until the session has closed, so a part-day never counts as a full one."""
    if now.hour * 60 + now.minute >= SESSION_END_MINUTE:
        return minute
    keep = minute.timestamps.astype("datetime64[D]") < np.datetime64(now.date())
    return MinuteData(minute.timestamps[keep], minute.open[keep], minute.high[keep], minute.low[keep], minute.close[keep])


def parse_closing_file(text: str, day: np.datetime64) -> DayClose:
    """Pick the Nifty 50 and India VIX rows out of NSE's ind_close_all file, checking the date."""
    try:
        rows = {row["Index Name"].strip().lower(): row for row in csv.DictReader(io.StringIO(text.lstrip("﻿")))}
        nifty, vix = rows["nifty 50"], rows["india vix"]
        expected = _day_text(day, "%d-%m-%Y")
        if nifty["Index Date"].strip() != expected or vix["Index Date"].strip() != expected:
            raise LiveDataError(f"NSE's closing file for {day} is dated {nifty['Index Date']}")
        return DayClose(
            day,
            float(nifty["Closing Index Value"]),
            float(nifty["Turnover (Rs. Cr.)"]),
            float(vix["Closing Index Value"]),
        )
    except (KeyError, ValueError) as error:
        raise LiveDataError(f"NSE's closing file for {day} has an unexpected layout: {error}") from error


def parse_holidays(payload: bytes) -> np.ndarray:
    """Equity-market holidays from NSE's list, which only covers the current calendar year."""
    rows = json.loads(payload)["CM"]
    return np.array(
        sorted(datetime.strptime(row["tradingDate"], "%d-%b-%Y").date().isoformat() for row in rows),
        dtype="datetime64[D]",
    )


def upcoming_sessions(after: np.datetime64, holidays: np.ndarray) -> np.ndarray:
    """Weekdays that aren't holidays, in the days after the last completed session."""
    days = np.arange(after + 1, after + 1 + LOOKAHEAD_DAYS)
    return days[(weekday_index(days) < 5) & ~np.isin(days, holidays)]


def _closing_data(days: np.ndarray, live_dir: Path) -> list[DayClose]:
    """NSE's closing file for each day. A published file doesn't change, so each is downloaded once."""
    closes = []
    for day in days:
        path = live_dir / f"nse_close_{day}.csv"
        if not path.exists():
            try:
                path.write_bytes(_fetch(CLOSING_FILE_URL.format(day=_day_text(day, "%d%m%Y"))))
            except URLError as error:
                raise LiveDataError(
                    f"NSE's closing file for {day} isn't available ({error}). It's usually published in the "
                    "evening, so try again later. The previous forecast has been left in place."
                ) from error
        closes.append(parse_closing_file(path.read_text(encoding="utf-8-sig"), day))
    return closes


def extend_series(series: DatedSeries, days: np.ndarray, values: list[float]) -> DatedSeries:
    return clean_series(np.concatenate([series.dates, days]), np.concatenate([series.values, values]))


def build_forecast(
    paths: RunPaths,
    minute: MinuteData,
    vix: DatedSeries,
    turnover: DatedSeries,
    expiry_dates: np.ndarray,
    next_session: np.datetime64,
) -> dict:
    """The next-session forecast and its reference forecasts, by the same steps as the later check."""
    verify_frozen(paths)
    daily, _ = daily_from_minute(minute)
    table = build_feature_table(daily, vix, turnover, expiry_dates, None, daily.dates[0], daily.dates[-1], next_session)
    if table.dates.size < WINDOW_DAYS or table.dates[-1] != daily.dates[-1]:
        raise LiveDataError(f"Only {table.dates.size} complete feature rows up to {daily.dates[-1]}, and {WINDOW_DAYS} are needed")
    columns = [table.columns.index(name) for name in MARKET_FEATURES]
    window = table.X[-WINDOW_DAYS:][:, columns]

    seed_forecasts = []
    for seed in SEEDS:
        model, meta = load_checkpoint(paths.artifacts / "models" / f"transformer_nosent_seed{seed}.npz")
        if meta["feature_names"] != MARKET_FEATURES:
            raise RuntimeError(f"Checkpoint seed {seed} was trained on different features")
        scaled = Scaler(meta["scaler_mean"], meta["scaler_std"]).transform(window)
        standardised, _ = predict(model, scaled[None])
        seed_forecasts.append(float(standardised[0, 0]) * meta["target_std"] + meta["target_mean"])

    latest = dict(zip(MARKET_FEATURES, (float(value) for value in window[-1])))
    study = read_table(paths.processed / "daily_features.csv")
    har_inputs = np.column_stack([study[name].astype(np.float64) for name in ("log_rv", "log_rv_w", "log_rv_m")])
    is_train = study["split"].astype(str) == "train"
    coefficients = har_fit(har_inputs[is_train], study["target_log_rv"].astype(np.float64)[is_train])
    har = har_forecast(np.array([[latest["log_rv"], latest["log_rv_w"], latest["log_rv_m"]]]), coefficients)
    return {
        "date": str(daily.dates[-1]),
        "target_date": str(next_session),
        "forecast": float(np.mean(seed_forecasts)),
        "seed_forecasts": seed_forecasts,
        "references": {
            "har": float(har[0]),
            "naive": latest["log_rv"],
            "vix_implied": float(vix_implied_forecast(np.array([latest["log_vix"]]))[0]),
        },
        "features": latest,
        "first_session": str(daily.dates[0]),
        "sessions_used": int(daily.dates.size),
    }


def track_record(paths: RunPaths) -> dict:
    """How this model did on the later check, and the error band that implies for a new forecast."""
    later = read_table(paths.artifacts / "forecasts_later.csv")
    errors = later["actual_log_rv"].astype(np.float64) - later["transformer_nosent"].astype(np.float64)
    low, high = np.quantile(errors, BAND_QUANTILES)
    metrics = json.loads((paths.artifacts / "metrics.json").read_text(encoding="utf-8"))["later_check"]
    model = metrics["models"]["transformer_nosent"]
    return {
        "n_days": metrics["n_days"],
        "first_date": metrics["forecast_dates"][0],
        "last_date": metrics["forecast_dates"][1],
        "var99_breaches": model["var99"]["breaches"],
        "var99_expected_breaches": model["var99"]["expected_breaches"],
        "var99_rejected": model["var99"]["kupiec_rejected"],
        "error_band_share": round(BAND_QUANTILES[1] - BAND_QUANTILES[0], 2),
        "error_band_low": float(low),
        "error_band_high": float(high),
    }


def write_json(path: Path, content: dict) -> None:
    """Write through a temporary file, so the API can never read a half-written forecast."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(paths: RunPaths = REAL_PATHS, now: datetime | None = None) -> dict:
    now = now or datetime.now(IST)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    payload = _fetch(BARS_URL)
    minute = complete_sessions_only(parse_bars(payload), now)
    daily, _ = daily_from_minute(minute)
    last_session = daily.dates[-1]
    (LIVE_DIR / f"yahoo_bars_{last_session}.json").write_bytes(payload)

    holiday_file = LIVE_DIR / "nse_holidays.json"
    try:
        holiday_file.write_bytes(_fetch(HOLIDAYS_URL))
    except URLError as error:
        if not holiday_file.exists():
            raise LiveDataError(f"Can't fetch NSE's holiday list: {error}") from error
        print(f"WARNING: couldn't refresh NSE's holiday list, so the saved copy is in use ({error})")
    holidays = parse_holidays(holiday_file.read_bytes())
    future = upcoming_sessions(last_session, holidays)
    expiry_dates = weekly_expiry_dates(np.union1d(daily.dates, future))

    vix, turnover = load_vix(paths.raw), load_turnover(paths.raw)
    if turnover is None:
        raise LiveDataError("NSE's turnover files are missing from data/raw")
    # The last session's official figures always come from its closing file, as do days the raw files don't reach.
    newest_on_file = min(vix.dates[-1], turnover.dates[-1])
    days = np.union1d(daily.dates[daily.dates > newest_on_file], last_session)
    closes = _closing_data(days, LIVE_DIR)
    vix = extend_series(vix, days, [close.india_vix for close in closes])
    turnover = extend_series(turnover, days, [close.turnover_crore for close in closes])

    official = closes[-1]
    difference = abs(float(daily.close[-1]) / official.nifty_close - 1)
    if difference > MAX_CLOSE_DIFFERENCE:
        raise LiveDataError(
            f"Yahoo's last price for {last_session} ({daily.close[-1]:.2f}) is {difference:.2%} from NSE's close "
            f"({official.nifty_close:.2f}), so the bars can't be trusted today"
        )

    forecast = build_forecast(paths, minute, vix, turnover, expiry_dates, future[0])
    result = {
        "data_source": paths.data_source,
        "made_at": now.isoformat(timespec="seconds"),
        "model": "transformer_nosent",
        **forecast,
        "var99": float(var99(np.array([forecast["forecast"]]))[0]),
        "nifty_close": official.nifty_close,
        "india_vix": official.india_vix,
        "turnover_crore": official.turnover_crore,
        "track_record": track_record(paths),
        "sources": {
            "bars": BARS_URL,
            "closing_file": CLOSING_FILE_URL.format(day=_day_text(official.date, "%d%m%Y")),
            "holidays": HOLIDAYS_URL,
        },
    }
    write_json(paths.artifacts / "live_forecast.json", result)
    move = np.sqrt(np.exp(result["forecast"])) * 100
    print(f"Forecast for {result['target_date']}, made after the close on {result['date']}: +/-{move:.2f}% (session only)")
    print(f"Wrote {paths.artifacts / 'live_forecast.json'}")
    return result


if __name__ == "__main__":
    try:
        main()
    except LiveDataError as error:
        raise SystemExit(f"No forecast written. {error}") from error
