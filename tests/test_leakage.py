"""Leakage tests: nothing dated after t may influence the features of row t."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.config import BARS_PER_DAY, MARKET_FEATURES, MONTHLY_WINDOW, SESSION_START_MINUTE, TRAIN_END, VAL_END
from src.data.bars import DailyBars, daily_from_minute
from src.data.features import (
    DailySentiment,
    FeatureTable,
    align_to_calendar,
    build_feature_table,
    roll_to_trading_day,
    trailing_mean,
)
from src.data.loaders import DatedSeries, MinuteData
from src.data.splits import Scaler, split_labels
from src.data.synthetic import simulate_market

START, END = np.datetime64("2014-01-01"), np.datetime64("2030-01-01")
CUTOFF = np.datetime64("2015-06-30")


@pytest.fixture(scope="module")
def market():
    return simulate_market("2015-01-01", "2015-12-31", seed=7)


@pytest.fixture(scope="module")
def expiry(market):
    return market.days[::21]


def sentiment_for(days: np.ndarray, seed: int = 1) -> DailySentiment:
    rng = np.random.default_rng(seed)
    return DailySentiment(days, rng.normal(size=days.size), rng.random(days.size), rng.random(days.size))


def build(market, expiry, minute=None, vix=None, turnover=None, sentiment=None) -> FeatureTable:
    daily, _ = daily_from_minute(minute if minute is not None else market.minute)
    return build_feature_table(
        daily,
        vix if vix is not None else market.vix,
        turnover if turnover is not None else market.turnover,
        expiry,
        sentiment if sentiment is not None else sentiment_for(market.days),
        START,
        END,
    )


def corrupt_after(market, cutoff: np.datetime64):
    """Change every input dated after the cutoff by a large amount."""
    late = market.minute.timestamps.astype("datetime64[D]") > cutoff
    factor = np.where(late, 3.0, 1.0)
    noise = np.where(late, 1.0 + 0.01 * np.sin(np.arange(late.size)), 1.0)
    minute = MinuteData(
        market.minute.timestamps,
        market.minute.open * factor * noise,
        market.minute.high * factor * noise * np.where(late, 1.02, 1.0),
        market.minute.low * factor * noise * np.where(late, 0.98, 1.0),
        market.minute.close * factor * noise,
    )
    vix = DatedSeries(market.vix.dates, np.where(market.vix.dates > cutoff, 99.0, market.vix.values))
    turnover = DatedSeries(market.turnover.dates, np.where(market.turnover.dates > cutoff, 1e9, market.turnover.values))
    base = sentiment_for(market.days)
    is_late = base.dates > cutoff
    sentiment = DailySentiment(
        base.dates,
        np.where(is_late, 50.0, base.mean_sentiment),
        np.where(is_late, 50.0, base.negative_share),
        np.where(is_late, 50.0, base.log_count),
    )
    return minute, vix, turnover, sentiment


def rows_up_to(table: FeatureTable, cutoff: np.datetime64) -> np.ndarray:
    return table.X[table.dates <= cutoff]


def test_corrupting_the_future_leaves_past_features_unchanged(market, expiry):
    clean = build(market, expiry)
    minute, vix, turnover, sentiment = corrupt_after(market, CUTOFF)
    corrupted = build(market, expiry, minute, vix, turnover, sentiment)
    np.testing.assert_array_equal(rows_up_to(clean, CUTOFF), rows_up_to(corrupted, CUTOFF))
    # The corruption really did reach the later rows, so the check above is not vacuous.
    assert not np.array_equal(clean.X[clean.dates > CUTOFF], corrupted.X[corrupted.dates > CUTOFF])


def test_truncating_the_data_leaves_past_rows_unchanged(market, expiry):
    """Rows built without any later data must match rows built with all of it."""
    clean = build(market, expiry)
    keep = market.minute.timestamps.astype("datetime64[D]") <= CUTOFF
    truncated_minute = MinuteData(
        market.minute.timestamps[keep], market.minute.open[keep], market.minute.high[keep],
        market.minute.low[keep], market.minute.close[keep],
    )
    truncated = build(market, expiry, minute=truncated_minute)
    assert truncated.dates.size > 50
    in_both = np.isin(clean.dates, truncated.dates)
    np.testing.assert_array_equal(clean.X[in_both], truncated.X)
    np.testing.assert_array_equal(clean.target[in_both], truncated.target)


def test_the_corruption_check_catches_a_deliberately_leaky_feature(market, expiry):
    """Prove the test above can fail: a feature using t+1 data must be flagged."""
    def leaky_rows(table: FeatureTable) -> np.ndarray:
        log_rv = table.X[:, table.columns.index("log_rv")]
        return np.append(log_rv[1:], np.nan)[table.dates <= CUTOFF]  # peeks at t+1

    minute, vix, turnover, sentiment = corrupt_after(market, CUTOFF)
    clean = leaky_rows(build(market, expiry))
    corrupted = leaky_rows(build(market, expiry, minute, vix, turnover, sentiment))
    assert not np.array_equal(clean, corrupted)


def test_target_is_the_next_days_log_rv(market, expiry):
    daily, _ = daily_from_minute(market.minute)
    table = build(market, expiry)
    positions = np.searchsorted(daily.dates, table.dates)
    np.testing.assert_allclose(table.target, np.log(daily.rv[positions + 1]))


def test_last_session_has_no_row_because_it_has_no_target(market, expiry):
    daily, _ = daily_from_minute(market.minute)
    table = build(market, expiry)
    assert table.dates[-1] == daily.dates[-2]
    assert np.isfinite(table.target).all()


def test_expiry_flag_marks_the_row_before_an_expiry_date(market):
    daily, _ = daily_from_minute(market.minute)
    expiry_date = daily.dates[100]
    table = build(market, np.array([expiry_date]))
    flag = table.X[:, table.columns.index("expiry_next")]
    assert table.dates[flag == 1].tolist() == [daily.dates[99]]


def make_minute(day: str, closes: np.ndarray) -> MinuteData:
    """A session with a given close on the last minute of each 5-minute slot.

    A 09:15 print equal to the first close is included, as every real session opens then.
    """
    opening = np.datetime64(day).astype("datetime64[m]") + np.timedelta64(SESSION_START_MINUTE, "m")
    slot_ends = opening + np.timedelta64(4, "m") + np.arange(closes.size) * np.timedelta64(5, "m")
    stamps = np.insert(slot_ends, 0, opening)
    prices = np.insert(closes, 0, closes[0])
    return MinuteData(stamps, prices, prices, prices, prices)


def join(a: MinuteData, b: MinuteData) -> MinuteData:
    return MinuteData(*(np.concatenate([x, y]) for x, y in zip(dataclasses.astuple(a), dataclasses.astuple(b))))


def test_overnight_jump_is_excluded_from_realized_variance():
    closes = 100.0 * np.exp(np.cumsum(np.full(BARS_PER_DAY, 0.001)))
    first = make_minute("2021-03-01", closes)
    small_gap = daily_from_minute(join(first, make_minute("2021-03-02", closes)))[0]
    huge_gap = daily_from_minute(join(first, make_minute("2021-03-02", closes * 1.5)))[0]
    np.testing.assert_allclose(small_gap.rv, huge_gap.rv)
    # 75 closes give 74 within-day returns of 0.001 each.
    np.testing.assert_allclose(small_gap.rv, 74 * 0.001**2)


def test_last_close_in_each_slot_is_used():
    """Only the slot's final minute should matter, not the prices inside the slot."""
    closes = 100.0 * np.exp(np.cumsum(np.full(BARS_PER_DAY, 0.001)))
    base = make_minute("2021-03-01", closes)
    inner_stamp = base.timestamps[10] - np.timedelta64(2, "m")  # mid-slot, before that slot's close
    noisy = MinuteData(
        np.append(base.timestamps, inner_stamp), np.append(base.open, 500.0), np.append(base.high, 500.0),
        np.append(base.low, 500.0), np.append(base.close, 500.0),
    )
    np.testing.assert_allclose(daily_from_minute(noisy)[0].rv, daily_from_minute(base)[0].rv)


def test_days_with_too_few_bars_are_dropped_and_logged():
    closes = 100.0 * np.exp(np.cumsum(np.full(BARS_PER_DAY, 0.001)))
    full = make_minute("2021-03-01", closes)
    combined = join(join(full, make_minute("2021-03-02", closes[:59])), make_minute("2021-03-03", closes[:60]))
    daily, dropped = daily_from_minute(combined)
    assert dropped.dates.tolist() == [np.datetime64("2021-03-02").item()]
    assert dropped.n_bars.tolist() == [59]
    assert daily.dates.size == 2


def test_session_entirely_outside_market_hours_is_logged_as_dropped():
    """The Diwali Muhurat session runs in the evening, so it has no bars in 09:15-15:30."""
    closes = 100.0 * np.exp(np.cumsum(np.full(BARS_PER_DAY, 0.001)))
    full = make_minute("2020-11-13", closes)
    evening = np.datetime64("2020-11-14T18:15").astype("datetime64[m]") + np.arange(60) * np.timedelta64(1, "m")
    muhurat = MinuteData(evening, *(np.full(60, 100.0) for _ in range(4)))
    daily, dropped = daily_from_minute(join(join(full, muhurat), make_minute("2020-11-16", closes)))
    assert daily.dates.astype(str).tolist() == ["2020-11-13", "2020-11-16"]
    assert dropped.dates.astype(str).tolist() == ["2020-11-14"]
    assert dropped.n_bars.tolist() == [0]
    assert "09:15-15:30" in dropped.reasons[0]


def test_utc_timestamps_are_rejected():
    from src.data.loaders import DataFileError

    closes = 100.0 * np.exp(np.cumsum(np.full(BARS_PER_DAY, 0.001)))
    ist = make_minute("2021-03-01", closes)
    utc = MinuteData(ist.timestamps - np.timedelta64(330, "m"), ist.open, ist.high, ist.low, ist.close)
    with pytest.raises(DataFileError, match="UTC"):
        daily_from_minute(utc)


def test_trailing_mean_uses_only_past_values():
    x = np.arange(1.0, 11.0)
    out = trailing_mean(x, 5)
    assert np.isnan(out[:4]).all()
    np.testing.assert_allclose(out[4:], [3, 4, 5, 6, 7, 8])
    x_future_changed = x.copy()
    x_future_changed[7:] = 1e6
    np.testing.assert_array_equal(out[:7], trailing_mean(x_future_changed, 5)[:7])


def test_alignment_forward_fills_one_day_and_never_backward():
    calendar = np.array(["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08"], dtype="datetime64[D]")
    series = DatedSeries(np.array(["2021-01-05", "2021-01-08"], dtype="datetime64[D]"), np.array([10.0, 20.0]))
    out = align_to_calendar(calendar, series)
    # 01-04: nothing earlier, and the 01-05 value must not be pulled backward.
    # 01-06: one-day forward fill. 01-07: value is two sessions old, so it stays missing.
    np.testing.assert_array_equal(out, [np.nan, 10.0, 10.0, np.nan, 20.0])


def test_headlines_roll_forward_and_never_backward():
    trading_days = np.array(["2021-01-08", "2021-01-11", "2021-01-12"], dtype="datetime64[D]")
    headline_days = np.array(
        ["2021-01-07", "2021-01-08", "2021-01-09", "2021-01-10", "2021-01-11", "2021-01-13"], dtype="datetime64[D]"
    )
    index = roll_to_trading_day(headline_days, trading_days)
    assert index.tolist() == [0, 0, 1, 1, 1, -1]  # weekend goes to Monday, after the last day is unused
    assigned = index >= 0
    assert (headline_days[assigned] <= trading_days[index[assigned]]).all()


def test_days_without_headlines_are_neutral_and_not_filled(market, expiry):
    daily, _ = daily_from_minute(market.minute)
    only_day = daily.dates[60]
    sentiment = DailySentiment(np.array([only_day]), np.array([0.5]), np.array([0.25]), np.array([2.0]))
    table = build(market, expiry, sentiment=sentiment)
    column = table.X[:, table.columns.index("news_mean_sentiment")]
    assert column[table.dates == only_day].tolist() == [0.5]
    assert (column[table.dates != only_day] == 0.0).all()


def test_missing_turnover_sets_the_feature_to_zero(market, expiry):
    daily, _ = daily_from_minute(market.minute)
    table = build_feature_table(daily, market.vix, None, expiry, sentiment_for(market.days), START, END)
    assert (table.X[:, table.columns.index("log_turnover_ratio")] == 0.0).all()


def test_early_rows_are_dropped_until_the_monthly_window_is_full(market, expiry):
    daily, _ = daily_from_minute(market.minute)
    table = build(market, expiry)
    assert table.dates[0] >= daily.dates[MONTHLY_WINDOW - 1]


def test_splits_are_chronological_and_disjoint():
    dates = np.arange(np.datetime64("2019-12-30"), np.datetime64("2021-01-03"))
    labels = split_labels(dates)
    assert labels[dates == TRAIN_END][0] == "train"
    assert labels[dates == TRAIN_END + 1][0] == "val"
    assert labels[dates == VAL_END][0] == "val"
    assert labels[dates == VAL_END + 1][0] == "test"
    assert dates[labels == "train"].max() < dates[labels == "val"].min()
    assert dates[labels == "val"].max() < dates[labels == "test"].min()


def test_scaler_ignores_later_rows(market, expiry):
    table = build(market, expiry)
    n_train = table.dates.size // 2
    train_only = Scaler.fit(table.X[:n_train])
    changed = table.X.copy()
    changed[n_train:] *= 1000.0
    refit = Scaler.fit(changed[:n_train])
    np.testing.assert_array_equal(train_only.mean, refit.mean)
    np.testing.assert_array_equal(train_only.std, refit.std)
    np.testing.assert_allclose(train_only.transform(table.X[:n_train]).mean(axis=0), 0.0, atol=1e-12)


def test_constant_column_does_not_divide_by_zero():
    scaler = Scaler.fit(np.zeros((10, 1)))
    assert np.isfinite(scaler.transform(np.ones((3, 1)))).all()


def test_no_market_feature_is_missing_after_build(market, expiry):
    table = build(market, expiry)
    assert np.isfinite(table.X[:, : len(MARKET_FEATURES)]).all()
