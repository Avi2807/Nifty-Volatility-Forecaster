"""Reference forecasts for evaluation only. These are not part of the project's models."""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def naive_forecast(log_rv: np.ndarray) -> np.ndarray:
    """Tomorrow's log RV equals today's."""
    return log_rv.copy()


def har_fit(har_inputs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS of y on [1, log RV_t, weekly mean, monthly mean], solving the normal equations."""
    design = np.column_stack([np.ones(y.size), har_inputs])
    return np.linalg.solve(design.T @ design, design.T @ y)


def har_forecast(har_inputs: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(har_inputs.shape[0]), har_inputs]) @ coefficients


def vix_implied_forecast(log_vix: np.ndarray) -> np.ndarray:
    """log((VIX / 100)^2 / 252): the VIX converted to a daily log variance."""
    return np.log((np.exp(log_vix) / 100.0) ** 2 / TRADING_DAYS)
