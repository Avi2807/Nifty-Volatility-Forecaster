"""Compute daily realized variance without crossing the overnight gap."""
import numpy as np
import pandas as pd

from src.config import MIN_BARS


def daily_from_bars(bars: pd.DataFrame):
    rows, dropped = [], []
    for day, group in bars.groupby(bars.index.date):
        if len(group) < MIN_BARS:
            dropped.append((day, len(group)))
            continue
        close = group["close"].to_numpy(dtype=float)
        day_open = float(group["open"].iloc[0])
        returns = np.diff(np.log(np.r_[day_open, close]))
        rows.append({
            "date": pd.Timestamp(day),
            "n_bars": len(group),
            "open": day_open,
            "high": float(group["high"].max()),
            "low": float(group["low"].min()),
            "close": float(close[-1]),
            "rv": float(np.sum(returns ** 2)),
            "oc_ret": float(np.sum(returns)),
        })
    if not rows:
        return pd.DataFrame(), dropped
    return pd.DataFrame(rows).set_index("date"), dropped
