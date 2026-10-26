"""Build leak-free daily features and next-day targets."""
import numpy as np
import pandas as pd

from src.config import BASE_FEATURES, SENT_FEATURES, TARGET, TARGET_OC


def build_features(
    daily: pd.DataFrame,
    vix: pd.DataFrame,
    turnover: pd.DataFrame,
    expiry_dates: set,
) -> pd.DataFrame:
    df = daily.join(vix[["vix_close"]], how="left").join(
        turnover[["turnover"]], how="left"
    )
    df[["vix_close", "turnover"]] = df[["vix_close", "turnover"]].ffill(limit=1)
    df = df.dropna(subset=["vix_close", "turnover"])

    features = pd.DataFrame(index=df.index)
    previous_close = df["close"].shift(1)
    features["log_rv"] = np.log(df["rv"])
    features["log_rv_w"] = np.log(df["rv"].rolling(5).mean())
    features["log_rv_m"] = np.log(df["rv"].rolling(22).mean())
    features["log_parkinson"] = np.log(
        np.log(df["high"] / df["low"]) ** 2 / (4 * np.log(2)) + 1e-12
    )
    features["ret"] = np.log(df["close"] / previous_close)
    features["neg_ret"] = features["ret"].clip(upper=0.0)
    features["gap"] = np.log(df["open"] / previous_close)
    features["log_vix"] = np.log(df["vix_close"])
    features["turnover_chg"] = np.log(
        df["turnover"] / df["turnover"].rolling(20).mean()
    )
    next_day = pd.Series(df.index, index=df.index).shift(-1)
    features["expiry_next"] = next_day.isin(expiry_dates).astype(float)
    for day in range(4):
        features[f"dow_{day}"] = (df.index.dayofweek == day).astype(float)
    for column in SENT_FEATURES:
        features[column] = np.nan

    features["oc_ret"] = df["oc_ret"]
    features[TARGET] = features["log_rv"].shift(-1)
    features[TARGET_OC] = df["oc_ret"].shift(-1)
    features["target_date"] = next_day
    features = features.dropna(
        subset=BASE_FEATURES + [TARGET, TARGET_OC, "target_date"]
    )
    features.index.name = "date"
    return features.reset_index()
