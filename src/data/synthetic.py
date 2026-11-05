"""Clearly labelled SYNTHETIC data, used to test the pipeline when raw files are absent.

These numbers are simulated and say nothing about the Nifty 50. Files carry a SYNTHETIC_
prefix and live in data/synthetic/raw/, and their outputs go to directories separate
from the real ones (see config.RunPaths). They must never appear in reported results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import SESSION_START_MINUTE
from src.data.loaders import DatedSeries, MinuteData, clean_series, write_table

SYNTHETIC_SEED = 20240229
MINUTES_PER_DAY = 375
MEAN_LOG_VARIANCE = -9.6  # about 0.8% daily volatility


@dataclass(frozen=True)
class SyntheticMarket:
    minute: MinuteData
    vix: DatedSeries
    turnover: DatedSeries
    days: np.ndarray
    log_variance: np.ndarray  # the simulated "true" daily log variance


def simulate_market(start: str = "2015-01-01", end: str = "2023-12-29", seed: int = SYNTHETIC_SEED) -> SyntheticMarket:
    """Simulate a persistent log-variance process and minute prices with a U-shaped day."""
    rng = np.random.default_rng(seed)
    all_days = np.arange(np.datetime64(start), np.datetime64(end) + 1)
    weekdays = all_days[(all_days.astype(np.int64) + 3) % 7 < 5]
    days = weekdays[rng.random(weekdays.size) > 0.025]  # about 2.5% fake holidays
    n = days.size

    log_var = np.empty(n)
    log_var[0] = MEAN_LOG_VARIANCE
    shocks = rng.normal(0.0, 0.35, n)
    # A crisis period, so the validation year contains a volatility spike as 2020 did.
    shocks[(days >= np.datetime64("2020-03-02")) & (days <= np.datetime64("2020-03-13"))] += 0.8
    for t in range(1, n):
        log_var[t] = MEAN_LOG_VARIANCE + 0.9 * (log_var[t - 1] - MEAN_LOG_VARIANCE) + shocks[t]
    daily_sigma = np.exp(log_var / 2.0)

    # Intraday U-shape: minute variance is higher at the open and the close.
    position = np.linspace(-1.0, 1.0, MINUTES_PER_DAY)
    shape = 0.6 + position**2
    shape /= shape.sum()
    minute_returns = daily_sigma[:, None] * np.sqrt(shape)[None, :] * rng.standard_normal((n, MINUTES_PER_DAY))
    gaps = rng.normal(0.0, 0.4, n) * daily_sigma

    intraday = np.cumsum(minute_returns, axis=1)
    day_close = np.log(8300.0) + np.cumsum(gaps + intraday[:, -1])
    day_open = day_close - intraday[:, -1]
    close = np.exp(day_open[:, None] + intraday)
    open_ = np.column_stack([np.exp(day_open), close[:, :-1]])
    wick = 1.0 + 0.3 * np.abs(minute_returns)
    high = np.maximum(open_, close) * wick
    low = np.minimum(open_, close) / wick

    stamps = days[:, None].astype("datetime64[m]") + (SESSION_START_MINUTE + np.arange(MINUTES_PER_DAY)).astype("timedelta64[m]")[None, :]

    # A few truncated sessions so that the thin-day filter gets exercised.
    present = np.ones((n, MINUTES_PER_DAY), dtype=bool)
    short_days = rng.choice(n, size=min(15, n // 10), replace=False)
    present[short_days, 120:] = False
    minute = MinuteData(stamps[present], open_[present], high[present], low[present], close[present])

    vix_close = 100.0 * np.sqrt(252.0 * np.exp(0.8 * log_var + 0.2 * MEAN_LOG_VARIANCE)) * np.exp(rng.normal(0.0, 0.05, n))
    turnover = np.exp(10.0 + 0.3 * (log_var - MEAN_LOG_VARIANCE) + rng.normal(0.0, 0.2, n))
    # A few missing daily values, so that the forward-fill and drop paths get exercised.
    vix_keep = rng.random(n) > 0.003
    turnover_keep = rng.random(n) > 0.003
    return SyntheticMarket(
        minute,
        clean_series(days[vix_keep], vix_close[vix_keep]),
        clean_series(days[turnover_keep], turnover[turnover_keep]),
        days,
        log_var,
    )


NEGATIVE_LINES = (
    "{co} shares fall {n} percent after profit warning", "{co} reports loss as sales decline",
    "markets slump on weak earnings from {co}", "{co} cuts jobs amid falling demand",
    "{co} profit drops {n} percent", "investors worried as {co} misses estimates",
)
POSITIVE_LINES = (
    "{co} shares rise {n} percent on strong results", "{co} profit jumps {n} percent",
    "markets rally as {co} beats estimates", "{co} wins major order and raises guidance",
    "{co} reports record sales growth", "investors cheer strong earnings from {co}",
)
NEUTRAL_LINES = (
    "{co} to hold annual meeting in {n} days", "{co} appoints new chief financial officer",
    "{co} announces board meeting schedule", "{co} operates {n} plants across the country",
    "{co} files quarterly report with regulator", "{co} moves head office to a new city",
)
COMPANIES = ("Alpha Steel", "Bharat Motors", "Coastal Bank", "Delta Power", "Eastern Textiles", "Fusion Tech")


def _sentence(rng: np.random.Generator, lines: tuple[str, ...]) -> str:
    return lines[rng.integers(len(lines))].format(co=COMPANIES[rng.integers(len(COMPANIES))], n=int(rng.integers(2, 40)))


def _write_headlines(path: Path, market: SyntheticMarket, rng: np.random.Generator) -> None:
    """Headlines every calendar day, more negative when simulated volatility is high."""
    dates, categories, texts = [], [], []
    for day in np.arange(market.days[0], np.datetime64("2023-07-01")):
        recent = max(int(np.searchsorted(market.days, day, side="right")) - 1, 0)
        stress = np.clip((market.log_variance[recent] - MEAN_LOG_VARIANCE) / 2.0, -0.5, 1.0)
        p_negative = float(np.clip(0.25 + 0.3 * stress, 0.05, 0.8))
        for _ in range(int(rng.integers(15, 45))):
            if rng.random() < 0.25:  # other sections, which the 'business' filter must remove
                dates.append(day)
                categories.append("sports.cricket")
                texts.append("local team wins final match")
                continue
            draw = rng.random()
            lines = NEGATIVE_LINES if draw < p_negative else NEUTRAL_LINES if draw < p_negative + 0.45 else POSITIVE_LINES
            dates.append(day)
            categories.append("business.india-business")
            texts.append(_sentence(rng, lines))
    compact = np.char.replace(np.array(dates).astype(str), "-", "")
    write_table(path, {"publish_date": compact, "headline_category": np.array(categories), "headline_text": np.array(texts)})


def _write_phrasebank(path: Path, rng: np.random.Generator, n: int = 2400) -> None:
    labelled = (("negative", NEGATIVE_LINES, 0.15), ("neutral", NEUTRAL_LINES, 0.6), ("positive", POSITIVE_LINES, 0.25))
    rows = []
    for label, lines, share in labelled:
        rows += [f"{_sentence(rng, lines)} .@{label}" for _ in range(int(n * share))]
    path.write_text("\n".join(rows) + "\n", encoding="latin-1")


def generate_synthetic_raw(raw_dir: Path, seed: int = SYNTHETIC_SEED) -> None:
    """Write the full synthetic raw data set into raw_dir."""
    print(f"Generating SYNTHETIC raw data in {raw_dir} ...")
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "README_SYNTHETIC.txt").write_text(
        "All files here are simulated for pipeline testing. They are not market data.\n"
        "Never report numbers derived from them.\n",
        encoding="utf-8",
    )
    market = simulate_market(seed=seed)
    minute = market.minute
    stamps = np.char.replace(np.datetime_as_string(minute.timestamps, unit="s"), "T", " ")
    write_table(
        raw_dir / "SYNTHETIC_nifty50_minute.csv",
        {"date": stamps, "open": minute.open, "high": minute.high, "low": minute.low, "close": minute.close},
        float_format="%.2f",
    )
    write_table(raw_dir / "SYNTHETIC_india_vix.csv", {"date": market.vix.dates, "close": market.vix.values}, float_format="%.4f")
    write_table(raw_dir / "SYNTHETIC_nifty50_turnover.csv", {"date": market.turnover.dates, "turnover": market.turnover.values}, float_format="%.2f")
    rng = np.random.default_rng(seed + 1)
    _write_headlines(raw_dir / "SYNTHETIC_india_news_headlines.csv", market, rng)
    _write_phrasebank(raw_dir / "SYNTHETIC_Sentences_66Agree.txt", rng)
