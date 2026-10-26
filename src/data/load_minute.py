"""Load minute OHLC data and resample it inside Indian market hours."""
import pandas as pd


def load_minute(path, ts_col: str = "date", utc_offset_hours: float = 0.0) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    timestamp = pd.to_datetime(df[ts_col.lower()]) + pd.to_timedelta(
        utc_offset_hours, unit="h"
    )
    df = df.set_index(timestamp).sort_index()
    required = ["open", "high", "low", "close"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"minute file is missing OHLC columns: {missing}")
    return df[required].loc[~df.index.duplicated(keep="first")].between_time(
        "09:15", "15:29"
    )


def to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Create 09:15, 09:20, ..., 15:25 bars from minute OHLC."""
    return df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()
