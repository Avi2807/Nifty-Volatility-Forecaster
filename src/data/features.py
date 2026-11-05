"""Daily feature table. Every row t uses data dated <= t, with one exception.

The exception is `expiry_next`, which reads the next trading date and the expiry
calendar. Both are published in advance, so this is not leakage.
Target: y(t) = log RV of the next retained trading day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import (
    DOW_FEATURES,
    MARKET_FEATURES,
    MONTHLY_WINDOW,
    NEWS_FEATURES,
    TURNOVER_WINDOW,
    WEEKLY_WINDOW,
    WINDOW_END,
    WINDOW_START,
    RunPaths,
)
from src.data.bars import DailyBars, daily_from_minute
from src.data.loaders import (
    DatedSeries,
    load_expiry_dates,
    load_minute,
    load_index_closes, load_turnover,
    load_vix,
    read_table,
    write_table,
)
from src.data.splits import split_labels


@dataclass(frozen=True)
class DailySentiment:
    """Aggregated news per trading day, as stored in sentiment_daily.csv."""

    dates: np.ndarray
    mean_sentiment: np.ndarray
    negative_share: np.ndarray
    log_count: np.ndarray


@dataclass(frozen=True)
class FeatureTable:
    dates: np.ndarray  # datetime64[D], the feature date t
    columns: tuple[str, ...]
    X: np.ndarray  # (rows, columns)
    target: np.ndarray  # log RV of day t+1
    nan_counts: dict[str, int]  # rows lost to each column, before dropping


def trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Mean of x[t-window+1..t]. NaN until the window is full or if it holds a NaN."""
    out = np.full(x.shape, np.nan)
    if x.size >= window:
        out[window - 1 :] = np.lib.stride_tricks.sliding_window_view(x, window).mean(axis=1)
    return out


def lagged_log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """ln(numerator[t] / denominator[t-1]), NaN on the first row."""
    out = np.full(numerator.shape, np.nan)
    out[1:] = np.log(numerator[1:] / denominator[:-1])
    return out


def align_to_calendar(calendar: np.ndarray, series: DatedSeries) -> np.ndarray:
    """Values on the trading calendar, forward-filling by at most one trading day.

    Each row looks up the newest observation dated <= t, so a future value can never be
    used. Observations older than the previous trading day leave NaN, and the row is
    dropped later. There are no backward fills.
    """
    newest = np.searchsorted(series.dates, calendar, side="right") - 1
    has_past = newest >= 0
    previous_day = np.concatenate([calendar[:1], calendar[:-1]])
    recent = has_past & (series.dates[np.maximum(newest, 0)] >= previous_day)
    out = np.full(calendar.shape, np.nan)
    out[recent] = series.values[newest[recent]]
    return out


def weekday_index(dates: np.ndarray) -> np.ndarray:
    """Monday = 0 ... Sunday = 6. 1970-01-01 (day 0 of datetime64) was a Thursday."""
    return (dates.astype("datetime64[D]").astype(np.int64) + 3) % 7


def roll_to_trading_day(event_dates: np.ndarray, trading_days: np.ndarray) -> np.ndarray:
    """Index of the first trading day >= each event date, or -1 if there is none.

    Weekend and holiday headlines move forward to the next session and are never moved
    backward, so day t only ever receives headlines dated <= t.
    """
    index = np.searchsorted(trading_days, event_dates, side="left")
    return np.where(index < trading_days.size, index, -1)


def build_feature_table(
    daily: DailyBars,
    vix: DatedSeries,
    turnover: DatedSeries | None,
    expiry_dates: np.ndarray,
    sentiment: DailySentiment | None = None,
    start: np.datetime64 = WINDOW_START,
    end: np.datetime64 = WINDOW_END,
    next_session: np.datetime64 | None = None,
) -> FeatureTable:
    """Compute all features on the full calendar, then trim to the study window.

    Computing before trimming means the last row of the window still gets its target
    from the first session after the window. A row is dropped if any market feature or
    the target is missing. News columns stay NaN (not dropped) if sentiment is None,
    which is the state of the table before the sentiment step has run.

    Give `next_session` for a live forecast: the last row then gets its expiry flag from that
    date and is kept with a NaN target, because that session hasn't happened yet.
    """
    dates = daily.dates
    log_rv = np.log(daily.rv)
    ret = lagged_log_ratio(daily.close, daily.close)

    if turnover is None:
        turnover_ratio = np.zeros(dates.shape)
    else:
        turnover_on_days = align_to_calendar(dates, turnover)
        turnover_ratio = np.log(turnover_on_days / trailing_mean(turnover_on_days, TURNOVER_WINDOW))

    # Only the last calendar row has no next day; that row has no target and is dropped.
    expiry_next = np.full(dates.shape, np.nan)
    expiry_next[:-1] = np.isin(dates[1:], expiry_dates).astype(np.float64)
    if next_session is not None:
        expiry_next[-1] = float(np.isin(next_session, expiry_dates))

    target = np.full(dates.shape, np.nan)
    target[:-1] = log_rv[1:]

    values: dict[str, np.ndarray] = {
        "log_rv": log_rv,
        "log_rv_w": trailing_mean(log_rv, WEEKLY_WINDOW),
        "log_rv_m": trailing_mean(log_rv, MONTHLY_WINDOW),
        "parkinson": (np.log(daily.high) - np.log(daily.low)) ** 2 / (4.0 * np.log(2.0)),
        "ret": ret,
        "ret_neg": np.minimum(ret, 0.0),  # NaN stays NaN on the first row
        "gap": lagged_log_ratio(daily.open, daily.close),
        "log_vix": np.log(align_to_calendar(dates, vix)),
        "log_turnover_ratio": turnover_ratio,
        "expiry_next": expiry_next,
    }
    # Weekday of day t. DOW_FEATURES lists names in Monday-first order.
    weekdays = weekday_index(dates)
    for day_number, name in enumerate(DOW_FEATURES):
        values[name] = (weekdays == day_number).astype(np.float64)

    news = _news_columns(dates, sentiment)
    values.update(zip(NEWS_FEATURES, news))

    market = np.column_stack([values[name] for name in MARKET_FEATURES])
    in_window = (dates >= start) & (dates <= end)
    complete = np.isfinite(market).all(axis=1) & np.isfinite(target)
    if next_session is not None:
        complete[-1] = np.isfinite(market[-1]).all()
    keep = in_window & complete

    # Rows lost to each column, counted inside the window for the build report.
    nan_counts = {name: int(np.isnan(values[name][in_window]).sum()) for name in MARKET_FEATURES}
    nan_counts["target"] = int(np.isnan(target[in_window]).sum())

    columns = MARKET_FEATURES + NEWS_FEATURES
    X = np.column_stack([values[name] for name in columns])[keep]
    return FeatureTable(dates[keep], columns, X, target[keep], nan_counts)


def _news_columns(dates: np.ndarray, sentiment: DailySentiment | None) -> list[np.ndarray]:
    """News features on the calendar. A day with no headlines is neutral: all three are 0."""
    if sentiment is None:
        return [np.full(dates.shape, np.nan) for _ in NEWS_FEATURES]
    out = [np.zeros(dates.shape) for _ in NEWS_FEATURES]
    position = np.searchsorted(dates, sentiment.dates)
    in_range = position < dates.size
    matched = np.zeros(sentiment.dates.shape, dtype=bool)
    matched[in_range] = dates[position[in_range]] == sentiment.dates[in_range]
    fields = (sentiment.mean_sentiment, sentiment.negative_share, sentiment.log_count)
    for column, field in zip(out, fields):
        column[position[matched]] = field[matched]
    return out


def load_sentiment_csv(paths: RunPaths) -> DailySentiment | None:
    path = paths.processed / "sentiment_daily.csv"
    if not path.exists():
        return None
    table = read_table(path)
    return DailySentiment(
        table["date"],
        table["mean_sentiment"].astype(np.float64),
        table["negative_share"].astype(np.float64),
        table["log_count"].astype(np.float64),
    )


def write_index_closes(paths: RunPaths) -> None:
    """Write NSE's published closes for the app to show.

    daily_bars.csv has a close too, but it's the last intraday bar's, which isn't the published figure.
    """
    closes = load_index_closes(paths.raw)
    if closes is None:
        print("NOTE: no NSE Historical_PR files found, so the app won't show the Nifty close.")
        return
    paths.processed.mkdir(parents=True, exist_ok=True)
    write_table(paths.processed / "index_closes.csv", {"date": closes.dates, "close": closes.values})


def build_daily_tables(paths: RunPaths) -> DailyBars:
    """Write daily bars and the dropped-day log. The kept sessions are the trading calendar."""
    print(f"\n=== Daily bars ({paths.data_source} data) ===")
    daily, dropped = daily_from_minute(load_minute(paths.raw))
    paths.processed.mkdir(parents=True, exist_ok=True)
    (paths.processed / "DATA_SOURCE.txt").write_text(paths.data_source + "\n", encoding="utf-8")
    write_table(
        paths.processed / "daily_bars.csv",
        {"date": daily.dates, "open": daily.open, "high": daily.high, "low": daily.low,
         "close": daily.close, "rv": daily.rv, "n_bars": daily.n_bars},
    )
    write_table(
        paths.processed / "dropped_days.csv",
        {"date": dropped.dates, "n_bars": dropped.n_bars, "reason": dropped.reasons},
    )
    write_index_closes(paths)
    print(f"Sessions kept: {daily.dates.size} ({daily.dates[0]} to {daily.dates[-1]}), dropped: {dropped.dates.size}")
    return daily


def build_and_write(paths: RunPaths, daily: DailyBars) -> FeatureTable:
    """Build daily_features.csv from the daily bars, VIX, turnover, expiry and news."""
    print(f"\n=== Features ({paths.data_source} data) ===")
    vix = load_vix(paths.raw)
    turnover = load_turnover(paths.raw)
    expiry_dates, expiry_is_placeholder = load_expiry_dates()
    sentiment = load_sentiment_csv(paths)

    if turnover is None:
        print("WARNING: no turnover file found, so log_turnover_ratio is set to 0 on every row.")
    if expiry_is_placeholder:
        print("WARNING: config/expiry_dates.csv is the placeholder calendar (last Thursdays). TODO: replace it.")
    if sentiment is None:
        print("NOTE: sentiment_daily.csv does not exist yet, so the news columns are left empty.")

    table = build_feature_table(daily, vix, turnover, expiry_dates, sentiment)
    labels = split_labels(table.dates)

    write_table(
        paths.processed / "daily_features.csv",
        {"date": table.dates, **{name: table.X[:, i] for i, name in enumerate(table.columns)},
         "target_log_rv": table.target, "split": labels},
    )

    # Window B: rows after the news archive ends, for the frozen no-sentiment model only.
    # News columns stay empty here, because no real news exists to put in them.
    later = build_feature_table(daily, vix, turnover, expiry_dates, None, WINDOW_END + 1, daily.dates[-1])
    write_table(
        paths.processed / "daily_features_later.csv",
        {"date": later.dates, **{name: later.X[:, i] for i, name in enumerate(later.columns)},
         "target_log_rv": later.target, "split": np.full(later.dates.shape, "later")},
    )
    span = f"{later.dates[0]} to {later.dates[-1]}" if later.dates.size else "empty"
    print(f"Later rows (no-sentiment model only): {later.dates.size}, {span}")

    print(f"Feature rows: {table.dates.size}, {table.dates[0]} to {table.dates[-1]}")
    for split in ("train", "val", "test"):
        members = table.dates[labels == split]
        span = f"{members[0]} to {members[-1]}" if members.size else "empty"
        print(f"  {split:5s} {members.size:5d} rows  {span}")
    lost = {name: count for name, count in table.nan_counts.items() if count}
    print(f"Missing values in the window before dropping rows: {lost or 'none'}")
    log_rv = table.X[:, table.columns.index("log_rv")]
    print(f"corr(log RV_t, log RV_t+1) = {np.corrcoef(log_rv, table.target)[0, 1]:.3f}")
    return table
