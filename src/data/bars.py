"""Turn 1-minute prices into daily OHLC and realized variance, using NumPy only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import (
    BAR_MINUTES,
    BARS_PER_DAY,
    MIN_BARS_PER_DAY,
    SESSION_END_MINUTE,
    SESSION_START_MINUTE,
)
from src.data.loaders import DataFileError, MinuteData


@dataclass(frozen=True)
class DailyBars:
    """One entry per retained trading day. These dates are the trading calendar."""

    dates: np.ndarray  # datetime64[D]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    rv: np.ndarray  # sum of squared 5-minute log returns, within the day only
    n_bars: np.ndarray


@dataclass(frozen=True)
class DroppedDays:
    dates: np.ndarray
    n_bars: np.ndarray
    reasons: np.ndarray


def _minute_of_day(timestamps: np.ndarray) -> np.ndarray:
    days = timestamps.astype("datetime64[D]")
    return (timestamps - days).astype("timedelta64[m]").astype(np.int64)


def _check_session_clock(timestamps: np.ndarray) -> None:
    """Fail loudly if timestamps are not IST: in UTC the session would start at 03:45."""
    minutes = _minute_of_day(timestamps)
    day_starts = np.flatnonzero(np.insert(timestamps[1:].astype("datetime64[D]") != timestamps[:-1].astype("datetime64[D]"), 0, True))
    usual_open = int(np.bincount(minutes[day_starts]).argmax())
    if usual_open != SESSION_START_MINUTE:
        raise DataFileError(
            f"Minute file: most days start at {usual_open // 60:02d}:{usual_open % 60:02d}, expected 09:15 IST. "
            "If this is 03:45 the timestamps are in UTC and must be converted."
        )


def daily_from_minute(minute: MinuteData) -> tuple[DailyBars, DroppedDays]:
    """Resample to 5-minute closes, compute RV and daily OHLC, drop thin days."""
    # Sort by time and keep the last row of any duplicated timestamp.
    order = np.argsort(minute.timestamps, kind="stable")
    timestamps = minute.timestamps[order]
    prices = np.column_stack([minute.open, minute.high, minute.low, minute.close])[order]
    last_of_stamp = np.append(timestamps[1:] != timestamps[:-1], True)
    valid = last_of_stamp & np.all(np.isfinite(prices) & (prices > 0), axis=1)
    timestamps, prices = timestamps[valid], prices[valid]
    if timestamps.size == 0:
        raise DataFileError("Minute file: no rows with valid positive prices")
    _check_session_clock(timestamps)

    all_dates = np.unique(timestamps.astype("datetime64[D]"))
    minutes = _minute_of_day(timestamps)
    in_session = (minutes >= SESSION_START_MINUTE) & (minutes <= SESSION_END_MINUTE)
    timestamps, prices, minutes = timestamps[in_session], prices[in_session], minutes[in_session]
    open_, high, low, close = prices.T

    dates, day_index = np.unique(timestamps.astype("datetime64[D]"), return_inverse=True)
    # A 15:30 print is the closing value of the final slot, not a 76th bar.
    slots = np.minimum((minutes - SESSION_START_MINUTE) // BAR_MINUTES, BARS_PER_DAY - 1)

    # Data is time-sorted, so the last row of each (day, slot) key holds that slot's close.
    keys = day_index * BARS_PER_DAY + slots
    slot_last = np.append(keys[1:] != keys[:-1], True)
    bar_close = np.full((dates.size, BARS_PER_DAY), np.nan)
    bar_close[day_index[slot_last], slots[slot_last]] = close[slot_last]
    n_bars = np.isfinite(bar_close).sum(axis=1)

    # Differencing one day at a time is what keeps the overnight gap out of RV.
    rv = np.array([np.sum(np.diff(np.log(row[np.isfinite(row)])) ** 2) for row in bar_close])

    day_first = np.flatnonzero(np.insert(day_index[1:] != day_index[:-1], 0, True))
    day_last = np.append(day_first[1:] - 1, day_index.size - 1)
    daily_open = open_[day_first]
    daily_high = np.maximum.reduceat(high, day_first)
    daily_low = np.minimum.reduceat(low, day_first)
    daily_close = close[day_last]

    thin = n_bars < MIN_BARS_PER_DAY
    flat = ~thin & (rv <= 0)  # log RV would be undefined
    keep = ~(thin | flat)
    reasons = np.where(thin, f"fewer than {MIN_BARS_PER_DAY} of {BARS_PER_DAY} bars", "zero realized variance")
    # Days with rows but none in session, such as the Diwali Muhurat evening session.
    off_session = np.setdiff1d(all_dates, dates)
    dropped_dates = np.concatenate([dates[~keep], off_session])
    order = np.argsort(dropped_dates)
    return (
        DailyBars(dates[keep], daily_open[keep], daily_high[keep], daily_low[keep], daily_close[keep], rv[keep], n_bars[keep]),
        DroppedDays(
            dropped_dates[order],
            np.concatenate([n_bars[~keep], np.zeros(off_session.size, dtype=n_bars.dtype)])[order],
            np.concatenate([reasons[~keep], np.full(off_session.size, "no bars inside 09:15-15:30")])[order],
        ),
    )
