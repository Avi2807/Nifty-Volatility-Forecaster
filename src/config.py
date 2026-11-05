"""Central configuration: dates, session constants, feature names, run paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Study window and date-based splits. A row belongs to the split of its feature date t.
WINDOW_START = np.datetime64("2015-01-01")
WINDOW_END = np.datetime64("2023-06-30")
TRAIN_END = np.datetime64("2019-12-31")
VAL_END = np.datetime64("2020-12-31")

# NSE session: 09:15 to 15:30 IST is 375 minutes, i.e. 75 five-minute slots.
SESSION_START_MINUTE = 9 * 60 + 15
SESSION_END_MINUTE = 15 * 60 + 30
BAR_MINUTES = 5
BARS_PER_DAY = 75
MIN_BARS_PER_DAY = 60

WEEKLY_WINDOW = 5
MONTHLY_WINDOW = 22
TURNOVER_WINDOW = 20

# Weekday one-hot, Monday first. With 5 columns the model has 15 market features and 18
# in total (the brief's 14/17 did not match its own feature list; 15/18 was chosen).
# A special Saturday session is all zeros. Model input size follows this tuple.
DOW_FEATURES: tuple[str, ...] = ("dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri")

MARKET_FEATURES: tuple[str, ...] = (
    "log_rv",  # log realized variance of day t
    "log_rv_w",  # mean log RV over t-4..t
    "log_rv_m",  # mean log RV over t-21..t
    "parkinson",  # (ln H - ln L)^2 / (4 ln 2)
    "ret",  # ln(close_t / close_t-1)
    "ret_neg",  # min(ret, 0)
    "gap",  # ln(open_t / close_t-1)
    "log_vix",  # log India VIX close
    "log_turnover_ratio",  # log(turnover_t / trailing 20-day mean)
    "expiry_next",  # 1 if day t+1 is an expiry date (calendar is known in advance)
    *DOW_FEATURES,
)
NEWS_FEATURES: tuple[str, ...] = (
    "news_mean_sentiment",  # mean of P(pos) - P(neg)
    "news_negative_share",  # share of headlines whose top class is negative
    "news_log_count",  # log(1 + headline count)
)
ALL_FEATURES: tuple[str, ...] = MARKET_FEATURES + NEWS_FEATURES

EXPIRY_FILE = ROOT / "config" / "expiry_dates.csv"


@dataclass(frozen=True)
class RunPaths:
    """Where one pipeline run reads and writes.

    Real and synthetic runs use disjoint directories, so synthetic numbers can never
    end up in data/processed or artifacts/.
    """

    raw: Path
    processed: Path
    artifacts: Path
    is_synthetic: bool

    @property
    def data_source(self) -> str:
        return "synthetic" if self.is_synthetic else "real"


REAL_PATHS = RunPaths(ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "artifacts", False)
SYNTHETIC_PATHS = RunPaths(
    ROOT / "data" / "synthetic" / "raw",
    ROOT / "data" / "synthetic" / "processed",
    ROOT / "artifacts_synthetic",
    True,
)
