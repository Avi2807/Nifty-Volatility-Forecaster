"""GARCH(1,1): the recursion, its constraints, recovering known parameters, and what it may see."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.garch import (
    ALPHA_SHARE_GRID,
    PERCENT,
    PERSISTENCE_GRID,
    GarchFit,
    constrained,
    fit_garch,
    grid_start,
    log_variance_forecasts,
    negative_log_likelihood,
    numeric_gradient,
    variance_path,
)

FIT = GarchFit(omega=0.1, alpha=0.1, beta=0.8, initial_variance=1.0, negative_log_likelihood=0.0)


def simulate(omega: float, alpha: float, beta: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    variance = omega / (1.0 - alpha - beta)
    returns = np.empty(n)
    for t in range(n):
        returns[t] = np.sqrt(variance) * rng.normal()
        variance = omega + alpha * returns[t] ** 2 + beta * variance
    return returns


def test_recursion_matches_the_formula_by_hand():
    path = variance_path(np.array([1.0, -2.0]), 0.1, 0.2, 0.7, 0.5)
    assert path[0] == 0.5
    assert path[1] == pytest.approx(0.1 + 0.2 * 1.0 + 0.7 * 0.5)
    assert path[2] == pytest.approx(0.1 + 0.2 * 4.0 + 0.7 * path[1])


@pytest.mark.parametrize("theta", [[-30.0, -30.0, -30.0], [5.0, 30.0, 30.0], [0.0, 0.0, 0.0], [-3.0, 4.0, -2.0]])
def test_any_real_parameters_meet_the_constraints(theta):
    omega, alpha, beta = constrained(np.array(theta, dtype=float))
    assert omega > 0.0 and alpha >= 0.0 and beta >= 0.0
    assert alpha + beta < 1.0


def test_the_likelihood_matches_the_formula_by_hand():
    returns = np.array([1.0, -1.0])
    theta = np.array([np.log(0.2), 0.0, 0.0])  # omega 0.2, persistence 0.5, alpha 0.25, beta 0.25
    second = 0.2 + 0.25 * 1.0 + 0.25 * 0.5
    expected = 0.5 * ((np.log(2 * np.pi) + np.log(0.5) + 1.0 / 0.5) + (np.log(2 * np.pi) + np.log(second) + 1.0 / second))
    assert negative_log_likelihood(theta, returns, 0.5) == pytest.approx(expected)


def test_the_grid_searches_the_values_the_brief_sets():
    assert PERSISTENCE_GRID[0] == 0.90 and PERSISTENCE_GRID[-1] == 0.99 and PERSISTENCE_GRID.size == 10
    assert ALPHA_SHARE_GRID[0] == 0.05 and ALPHA_SHARE_GRID[-1] == 0.30


def test_fit_recovers_known_parameters_and_beats_the_grid():
    returns = simulate(0.05, 0.10, 0.85, 6000)
    fit = fit_garch(returns)
    assert fit.alpha == pytest.approx(0.10, abs=0.025)
    assert fit.beta == pytest.approx(0.85, abs=0.04)
    assert fit.omega == pytest.approx(0.05, abs=0.03)
    initial = float(returns.var(ddof=1))
    assert fit.negative_log_likelihood <= negative_log_likelihood(grid_start(returns, initial), returns, initial)


def test_fit_ends_at_a_stationary_point():
    returns = simulate(0.05, 0.10, 0.85, 3000, seed=4)
    fit = fit_garch(returns)
    persistence = fit.alpha + fit.beta
    theta = np.array([np.log(fit.omega), np.log(persistence / (1 - persistence)), np.log(fit.alpha / fit.beta)])
    assert np.abs(numeric_gradient(theta, returns, fit.initial_variance)).max() < 0.5


def test_long_run_volatility_is_annualised_from_percent():
    assert FIT.long_run_annual_vol_pct == pytest.approx(np.sqrt(1.0 * 252))


def test_forecasts_are_for_the_next_session_in_log_decimal_units():
    dates = np.datetime64("2021-01-04") + np.arange(4)
    returns = np.array([1.0, -2.0, 0.5, 3.0])
    path = variance_path(returns, 0.1, 0.1, 0.8, 1.0)
    forecast = log_variance_forecasts(FIT, dates, returns, dates[[1]])
    assert forecast[0] == pytest.approx(np.log(path[2] / PERCENT**2))


def test_a_forecast_never_sees_its_own_target_session_or_later():
    dates = np.datetime64("2021-01-04") + np.arange(6)
    returns = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 2.0])
    base = log_variance_forecasts(FIT, dates, returns, dates)
    changed = returns.copy()
    changed[3:] = 99.0
    after = log_variance_forecasts(FIT, dates, changed, dates)
    np.testing.assert_array_equal(base[:3], after[:3])  # forecasts made on days 0-2 use returns 0-2 only
    assert after[3] != base[3]


def test_fit_on_training_returns_ignores_anything_later():
    returns = simulate(0.05, 0.10, 0.85, 1200, seed=2)
    first = fit_garch(returns[:1000], max_steps=30)
    returns[1000:] = 50.0
    second = fit_garch(returns[:1000], max_steps=30)
    assert first == second


def test_an_unknown_forecast_date_is_refused():
    dates = np.datetime64("2021-01-04") + np.arange(3)
    with pytest.raises(RuntimeError, match="not a session"):
        log_variance_forecasts(FIT, dates, np.zeros(3), dates[:1] - 1)
    with pytest.raises(RuntimeError, match="not a session"):
        log_variance_forecasts(FIT, dates, np.zeros(3), dates[-1:] + 1)
