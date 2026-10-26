"""Load auxiliary daily data indexed by normalized trading date."""
import pandas as pd


def _by_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    result = df.copy()
    result.index = pd.to_datetime(result[date_col]).dt.normalize()
    return result[~result.index.duplicated(keep="first")].sort_index()


def load_vix(path, date_col: str = "Date", close_col: str = "Close") -> pd.DataFrame:
    df = _by_date(pd.read_csv(path), date_col)
    return df[[close_col]].rename(columns={close_col: "vix_close"})


def load_turnover(path, date_col: str = "Date", turnover_col: str = "Turnover") -> pd.DataFrame:
    df = _by_date(pd.read_csv(path), date_col)
    return df[[turnover_col]].rename(columns={turnover_col: "turnover"})


def load_expiry_dates(path, date_col: str = "date") -> set:
    return set(pd.to_datetime(pd.read_csv(path)[date_col]).dt.normalize())
