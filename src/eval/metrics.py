"""Forecast losses and statistical tests, written directly from their formulas."""

from __future__ import annotations

import math

import numpy as np

DM_LAGS = 5
VAR_MULTIPLIER = 2.33  # one-sided 99% point of the normal distribution, as the brief sets it
VAR_RATE = 0.01
KUPIEC_CRITICAL = 3.84  # chi-squared with 1 degree of freedom at the 5% level


def squared_errors(actual_log_rv: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    return (actual_log_rv - forecast) ** 2


def qlike_losses(actual_log_rv: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """RV/h - log(RV/h) - 1, with h = exp(forecast). Under-forecasting costs more than over-forecasting."""
    log_ratio = actual_log_rv - forecast
    return np.exp(log_ratio) - log_ratio - 1.0


def newey_west_variance(series: np.ndarray, lags: int) -> float:
    """Long-run variance with Bartlett weights, which keep the estimate non-negative."""
    centred = series - series.mean()
    n = centred.size
    variance = float(centred @ centred) / n
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        variance += 2.0 * weight * float(centred[lag:] @ centred[:-lag]) / n
    return variance


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = DM_LAGS) -> dict[str, float]:
    """Test of equal loss. A negative statistic means model A had the lower loss.

    The p-value is two-sided from the normal distribution: P(|Z| > |s|) = erfc(|s| / sqrt 2).
    """
    difference = loss_a - loss_b
    variance = newey_west_variance(difference, lags)
    if variance <= 0.0:  # identical forecasts: no evidence either way
        return {"mean_loss_difference": float(difference.mean()), "statistic": 0.0, "p_value": 1.0}
    statistic = float(difference.mean()) / math.sqrt(variance / difference.size)
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return {"mean_loss_difference": float(difference.mean()), "statistic": statistic, "p_value": p_value}


def var99(forecast: np.ndarray) -> np.ndarray:
    """99% one-day VaR as a positive log-return loss: 2.33 x sqrt(exp(forecast))."""
    return VAR_MULTIPLIER * np.sqrt(np.exp(forecast))


def var_breaches(forecast: np.ndarray, return_oc: np.ndarray) -> np.ndarray:
    """True where the open-to-close loss exceeded the VaR. Gains can't breach."""
    return -return_oc > var99(forecast)


def _binomial_log_likelihood(breaches: int, n: int, rate: float) -> float:
    """Treats 0 x log 0 as 0, so that zero breaches and all breaches stay defined."""
    total = 0.0
    if breaches > 0:
        total += breaches * math.log(rate)
    if n - breaches > 0:
        total += (n - breaches) * math.log(1.0 - rate)
    return total


def kupiec_lr(breaches: int, n: int, rate: float = VAR_RATE) -> float:
    """Likelihood ratio that the true breach rate equals the VaR's stated rate."""
    return 2.0 * (_binomial_log_likelihood(breaches, n, breaches / n) - _binomial_log_likelihood(breaches, n, rate))


def var_diagnostics(actual_log_rv: np.ndarray, return_oc: np.ndarray) -> dict:
    """How well RV measures open-to-close risk, whatever the forecast.

    RV leaves out the move from the open to the first 5-minute close, and intraday trends make
    the day's squared return exceed the sum of its squared 5-minute returns. A ratio above 1
    therefore makes any VaR built on an RV forecast too low.
    """
    return {
        "mean_squared_return_over_mean_rv": float(np.mean(return_oc ** 2) / np.mean(np.exp(actual_log_rv))),
        "breaches_if_rv_were_known": int(var_breaches(actual_log_rv, return_oc).sum()),
    }


def score_model(actual_log_rv: np.ndarray, forecast: np.ndarray, return_oc: np.ndarray) -> dict:
    """Every per-model number the report needs, for one period."""
    n = actual_log_rv.size
    breaches = int(var_breaches(forecast, return_oc).sum())
    likelihood_ratio = kupiec_lr(breaches, n)
    return {
        "mse_log_rv": float(squared_errors(actual_log_rv, forecast).mean()),
        "qlike": float(qlike_losses(actual_log_rv, forecast).mean()),
        "mean_error_log_rv": float((forecast - actual_log_rv).mean()),  # positive means forecasts ran high
        "var99": {
            "breaches": breaches,
            "expected_breaches": n * VAR_RATE,
            "breach_rate": breaches / n,
            "kupiec_lr": likelihood_ratio,
            "kupiec_rejected": likelihood_ratio > KUPIEC_CRITICAL,
        },
    }
