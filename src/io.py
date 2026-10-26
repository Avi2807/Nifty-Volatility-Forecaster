"""Shared CSV formats and the log-variance to volatility conversion."""
import numpy as np
import pandas as pd

from src.config import DATA_PROC, FORECASTS


def log_rv_to_vol(log_rv):
    """Convert log realized variance to standard deviation."""
    return np.exp(0.5 * np.asarray(log_rv, dtype=float))


def read_features() -> pd.DataFrame:
    return pd.read_csv(
        DATA_PROC / "daily_features.csv",
        parse_dates=["date", "target_date"],
    ).sort_values("date").reset_index(drop=True)


def write_forecast(name: str, rows: pd.DataFrame, forecast_log_rv, subdir: str = "") -> None:
    """Write the stable forecast contract used by all evaluation code."""
    out = pd.DataFrame({
        "date": rows["date"].to_numpy(),
        "target_date": rows["target_date"].to_numpy(),
        "forecast_log_rv": np.asarray(forecast_log_rv, dtype=float),
    })
    folder = FORECASTS / subdir
    folder.mkdir(parents=True, exist_ok=True)
    out.to_csv(folder / f"{name}.csv", index=False)


def read_forecast(name: str) -> pd.DataFrame:
    return pd.read_csv(
        FORECASTS / f"{name}.csv",
        parse_dates=["date", "target_date"],
    )
