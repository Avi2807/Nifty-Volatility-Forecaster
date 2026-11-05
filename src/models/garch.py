"""GARCH(1,1) on open-to-close returns, fitted to the training period only.

Usage: python -m src.models.garch

s2[t+1] = omega + alpha * r[t]^2 + beta * s2[t], with r in percent, so it covers the same 09:15-15:30
window as realized volatility. It models the variance of the return, not realized variance, so its
forecast isn't guaranteed to sit at the same level as log RV. The comparison reports that gap.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from src.config import REAL_PATHS, RunPaths
from src.data.loaders import read_table
from src.eval.metrics import squared_errors
from src.eval.references import TRADING_DAYS
from src.models.comparison import model_folder, record_training
from src.nn.optim import AdamW

PERCENT = 100.0
PERSISTENCE_GRID = np.round(np.arange(0.90, 0.995, 0.01), 2)  # 0.90 to 0.99
ALPHA_SHARE_GRID = np.round(np.arange(0.05, 0.301, 0.05), 2)  # 0.05 to 0.30
LEARNING_RATE = 0.01
MAX_STEPS = 5000
GRADIENT_STEP = 1e-5
STOP_AFTER = 200  # steps without a real improvement
MIN_IMPROVEMENT = 1e-7


@dataclass(frozen=True)
class GarchFit:
    omega: float
    alpha: float
    beta: float
    initial_variance: float  # sample variance of the training returns, in percent^2
    negative_log_likelihood: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_annual_vol_pct(self) -> float:
        return float(np.sqrt(self.omega / (1.0 - self.persistence) * TRADING_DAYS))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def constrained(theta: np.ndarray) -> tuple[float, float, float]:
    """(a, b, c) -> (omega, alpha, beta). Any real values give omega > 0, alpha, beta >= 0 and alpha + beta < 1."""
    persistence = _sigmoid(theta[1])
    alpha = persistence * _sigmoid(theta[2])
    return float(np.exp(theta[0])), alpha, persistence - alpha


def variance_path(returns: np.ndarray, omega: float, alpha: float, beta: float, initial: float) -> np.ndarray:
    """s2[0..n], one longer than returns: s2[i] uses returns before i, so s2[n] forecasts the next session."""
    path = np.empty(returns.size + 1)
    path[0] = current = initial
    for i, squared in enumerate((returns**2).tolist()):
        current = omega + alpha * squared + beta * current
        path[i + 1] = current
    return path


def negative_log_likelihood(theta: np.ndarray, returns: np.ndarray, initial: float) -> float:
    variance = variance_path(returns, *constrained(theta), initial)[:-1]
    return float(0.5 * np.sum(np.log(2.0 * np.pi) + np.log(variance) + returns**2 / variance))


def numeric_gradient(theta: np.ndarray, returns: np.ndarray, initial: float, step: float = GRADIENT_STEP) -> np.ndarray:
    """Central differences over (a, b, c)."""
    gradient = np.zeros(3)
    for i in range(3):
        shift = np.zeros(3)
        shift[i] = step
        above = negative_log_likelihood(theta + shift, returns, initial)
        below = negative_log_likelihood(theta - shift, returns, initial)
        gradient[i] = (above - below) / (2.0 * step)
    return gradient


def grid_start(returns: np.ndarray, initial: float) -> np.ndarray:
    """The best coarse grid point. Omega comes from setting the long-run variance to the sample variance."""
    best, best_loss = np.zeros(3), np.inf
    for persistence in PERSISTENCE_GRID:
        for share in ALPHA_SHARE_GRID:
            theta = np.array([np.log(initial * (1.0 - persistence)), _logit(persistence), _logit(share)])
            loss = negative_log_likelihood(theta, returns, initial)
            if loss < best_loss:
                best, best_loss = theta, loss
    return best


def fit_garch(returns: np.ndarray, max_steps: int = MAX_STEPS) -> GarchFit:
    """Adam (the project's AdamW with no decay) from the grid's best point, keeping the best point seen."""
    initial = float(returns.var(ddof=1))
    theta = grid_start(returns, initial)
    gradient = np.zeros(3)
    optimiser = AdamW([(theta, gradient)], lr=LEARNING_RATE, weight_decay=0.0)
    best_theta, best_loss = theta.copy(), negative_log_likelihood(theta, returns, initial)
    since_improved = 0
    for _ in range(max_steps):
        gradient[...] = numeric_gradient(theta, returns, initial)
        optimiser.step()
        loss = negative_log_likelihood(theta, returns, initial)
        since_improved = 0 if loss < best_loss - MIN_IMPROVEMENT else since_improved + 1
        if loss < best_loss:
            best_theta, best_loss = theta.copy(), loss
        if since_improved >= STOP_AFTER:
            break
    return GarchFit(*constrained(best_theta), initial, best_loss)


def load_returns(paths: RunPaths) -> tuple[np.ndarray, np.ndarray]:
    """Every session's date and open-to-close log return, in percent."""
    bars = read_table(paths.processed / "daily_bars.csv")
    return bars["date"], np.log(bars["close"].astype(np.float64) / bars["open"].astype(np.float64)) * PERCENT


def log_variance_forecasts(fit: GarchFit, session_dates: np.ndarray, returns: np.ndarray, forecast_dates: np.ndarray) -> np.ndarray:
    """The log variance, in decimal units like log RV, for the session after each forecast date."""
    path = variance_path(returns, fit.omega, fit.alpha, fit.beta, fit.initial_variance)
    position = np.searchsorted(session_dates, forecast_dates)
    if (position >= session_dates.size).any() or not np.array_equal(session_dates[position], forecast_dates):
        raise RuntimeError("A forecast date is not a session in daily_bars.csv")
    return np.log(path[position + 1] / PERCENT**2)


def save_fit(paths: RunPaths, fit: GarchFit) -> None:
    (model_folder(paths) / "garch.json").write_text(json.dumps(asdict(fit), indent=2), encoding="utf-8")


def load_fit(paths: RunPaths) -> GarchFit:
    return GarchFit(**json.loads((model_folder(paths) / "garch.json").read_text(encoding="utf-8")))


def run_garch(paths: RunPaths) -> GarchFit:
    """Fit on training returns, then report on validation. Test returns are never read here."""
    print(f"\n=== GARCH(1,1) ({paths.data_source} data) ===")
    started = time.perf_counter()
    dates, returns = load_returns(paths)
    table = read_table(paths.processed / "daily_features.csv")
    labels = table["split"].astype(str)
    # The cut-offs come from the same split labels every other model trains on.
    in_train = dates <= table["date"][labels == "train"].max()
    fit = fit_garch(returns[in_train])
    seconds = time.perf_counter() - started
    save_fit(paths, fit)

    in_val = labels == "val"
    val_dates, actual = table["date"][in_val], table["target_log_rv"][in_val].astype(np.float64)
    known = dates <= val_dates.max()  # the last validation forecast needs nothing after its own date
    forecast = log_variance_forecasts(fit, dates[known], returns[known], val_dates)
    val_mse = float(squared_errors(actual, forecast).mean())
    record_training(paths, "garch", {
        "n_parameters": 3, "n_train_returns": int(in_train.sum()), "seconds": seconds, "val_mse_log_rv": val_mse,
        "omega": fit.omega, "alpha": fit.alpha, "beta": fit.beta, "persistence": fit.persistence,
        "long_run_annual_vol_pct": fit.long_run_annual_vol_pct, "negative_log_likelihood": fit.negative_log_likelihood,
    })
    print(f"Training returns: {int(in_train.sum())}, {dates[in_train][0]} to {dates[in_train][-1]}")
    print(f"omega {fit.omega:.6f}  alpha {fit.alpha:.6f}  beta {fit.beta:.6f}  alpha + beta {fit.persistence:.6f}")
    print(f"Long-run annualised volatility: {fit.long_run_annual_vol_pct:.2f}%")
    print(f"Negative log-likelihood {fit.negative_log_likelihood:.4f}, validation MSE on log RV {val_mse:.4f}, {seconds:.1f}s")
    return fit


if __name__ == "__main__":
    run_garch(REAL_PATHS)
