"""The simpler models that can run live, scored on the later check, and the calibrated VaR multiplier.

Usage: python -m src.eval.challengers

The study's MLP and LSTM read news, whose archive ends in June 2023, so they can't run live. Here they
are trained again without it, with the same recipes and seeds and no tuning. GARCH reads no news and is
used as it stands. None of them sees the test period here, and no existing score or checkpoint changes.

The calibrated multiplier is set on the later check, so that period can't also judge it. It is post-test
and is judged only on the forward record's new days, in src/forward.py.
"""

from __future__ import annotations

import numpy as np

from src.config import REAL_PATHS, RunPaths
from src.data.loaders import read_table, write_table
from src.eval.evaluate import write_json
from src.eval.metrics import DM_LAGS, VAR_MULTIPLIER, VAR_RATE, diebold_mariano, qlike_losses, score_model, squared_errors
from src.models.comparison import verify_comparison_frozen
from src.models.dataset import load_dataset
from src.models.garch import load_fit, load_returns, log_variance_forecasts
from src.models.lstm import RECIPE as LSTM_RECIPE, build_lstm, whole_window
from src.models.mlp import RECIPE as MLP_RECIPE, build_mlp, last_days_flat
from src.models.neural import neural_forecasts, train_neural

NEURAL = {"mlp_nosent": (build_mlp, last_days_flat, MLP_RECIPE), "lstm_nosent": (build_lstm, whole_window, LSTM_RECIPE)}
CHALLENGERS = ("garch", *NEURAL)
BASE = "transformer_nosent"
LOSSES = (("squared_error_log_rv", squared_errors), ("qlike", qlike_losses))
FIVE_PERCENT = 0.05


def later_forecasts(paths: RunPaths, dates: np.ndarray) -> dict[str, np.ndarray]:
    """Each challenger's log variance forecast for the later check's dates."""
    verify_comparison_frozen(paths)
    data = load_dataset(paths, with_sentiment=False)
    windows = data.splits["later"]
    if not np.array_equal(windows.dates, dates):
        raise RuntimeError("The challengers' windows cover different dates from the later check")
    forecasts = {name: neural_forecasts(paths, name, build, prepare, data, windows.x).mean(axis=0) for name, (build, prepare, _) in NEURAL.items()}
    session_dates, returns = load_returns(paths)
    return {"garch": log_variance_forecasts(load_fit(paths), session_dates, returns, dates), **forecasts}


def calibrated_multiplier(forecast: np.ndarray, return_oc: np.ndarray) -> float:
    """The multiple of the forecast standard deviation that open-to-close losses passed on 1% of days."""
    return float(np.quantile(-return_oc / np.sqrt(np.exp(forecast)), 1.0 - VAR_RATE))


def run_challengers(paths: RunPaths) -> dict:
    """Train both networks, then score all three on the later check and set the calibrated multiplier."""
    for name, (build, prepare, recipe) in NEURAL.items():
        train_neural(paths, name, build, prepare, recipe, with_sentiment=False)

    later = read_table(paths.artifacts / "forecasts_later.csv")
    actual, return_oc = later["actual_log_rv"].astype(np.float64), later["return_oc"].astype(np.float64)
    base = later[BASE].astype(np.float64)
    forecasts = later_forecasts(paths, later["date"])
    write_table(paths.artifacts / "challengers_later.csv", {"date": later["date"], "target_date": later["target_date"], **forecasts})

    tests = []
    for name, forecast in forecasts.items():
        for loss_name, loss in LOSSES:
            result = diebold_mariano(loss(actual, base), loss(actual, forecast), DM_LAGS)
            tests.append({"model_a": BASE, "model_b": name, "loss": loss_name, **result})
    multiplier = calibrated_multiplier(base, return_oc)
    summary = {
        "data_source": paths.data_source,
        "later_check": {
            "n_days": int(actual.size),
            "target_dates": [str(later["target_date"][0]), str(later["target_date"][-1])],
            "models": {name: score_model(actual, forecast, return_oc) for name, forecast in forecasts.items()},
            "diebold_mariano": tests,
            "bonferroni_limit": FIVE_PERCENT / len(tests),
        },
        "calibrated_var": {
            "model": BASE,
            "multiplier": multiplier,
            "rule_multiplier": VAR_MULTIPLIER,
            "n_days": int(actual.size),
            "fitted_on": [str(later["target_date"][0]), str(later["target_date"][-1])],
        },
    }
    write_json(paths.artifacts / "challengers.json", summary)
    print(f"\nLater check, {actual.size} sessions. Calibrated VaR multiplier {multiplier:.3f}, against the rule's {VAR_MULTIPLIER}")
    for name, score in summary["later_check"]["models"].items():
        print(f"  {name:12s} MSE {score['mse_log_rv']:.4f}  QLIKE {score['qlike']:.4f}  breaches {score['var99']['breaches']}")
    return summary


if __name__ == "__main__":
    run_challengers(REAL_PATHS)
