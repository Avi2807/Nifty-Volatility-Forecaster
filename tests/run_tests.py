"""Small smoke runner for the Phase 0 scaffold.

The full gradient, leakage, metrics, and import-policy tests will be added
alongside their corresponding model modules in later phases.
"""
import sys

import numpy as np
import pandas as pd

from src.config import BASE_FEATURES
from src.data.features import build_features
from src.data.split import Scaler, make_windows, split_masks
from src.io import log_rv_to_vol


def test_data_contract_smoke():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=30)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(dates))))
    daily = pd.DataFrame({
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "rv": rng.uniform(1e-5, 1e-4, len(dates)),
        "oc_ret": rng.normal(0.0, 0.01, len(dates)),
    }, index=dates)
    vix = pd.DataFrame({"vix_close": 18.0}, index=dates)
    turnover = pd.DataFrame({"turnover": 1000.0}, index=dates)
    features = build_features(daily, vix, turnover, set())
    assert set(BASE_FEATURES).issubset(features.columns)
    assert features["target_date"].notna().all()
    assert np.all(log_rv_to_vol(features["log_rv"]) > 0)


def test_split_and_window_contract():
    dates = pd.bdate_range("2019-12-20", periods=30)
    frame = pd.DataFrame({
        "target_date": dates,
    })
    masks = split_masks(frame)
    assert sum(mask.sum() for mask in masks.values()) == len(frame)
    X = np.arange(20, dtype=float).reshape(10, 2)
    scaler = Scaler().fit(X[:5])
    assert np.allclose(scaler.transform(X[:5]).mean(axis=0), 0.0)
    windows, targets, ends = make_windows(X, np.arange(10), 3)
    assert windows.shape == (8, 3, 2)
    assert targets.shape == ends.shape == (8,)


def main() -> int:
    tests = [test_data_contract_smoke, test_split_and_window_contract]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
