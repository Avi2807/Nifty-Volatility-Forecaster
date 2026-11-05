"""Compare every model on the test set. Run after the comparison models are trained and frozen.

Usage: python -m src.eval.compare

The only module that makes test forecasts for GARCH, the MLP and the LSTM. It adds to the existing
results and doesn't change them: it refuses to run if the Transformer's recomputed forecasts differ
from the saved ones, and existing cells in the CSV files keep their exact text.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from src.config import REAL_PATHS, RunPaths
from src.data.loaders import append_columns, read_table
from src.eval.evaluate import MODELS, build_evaluation_rows, verify_frozen, write_json
from src.eval.export import gradcheck_summary
from src.eval.figures import LABELS, model_ladder, rolling_qlike
from src.eval.metrics import DM_LAGS, diebold_mariano, qlike_losses, score_model, squared_errors, var99, var_breaches
from src.eval.references import har_fit
from src.models.comparison import COMPARISON_MODELS, read_log, verify_comparison_frozen
from src.models.dataset import load_dataset
from src.models.garch import load_fit, load_returns, log_variance_forecasts
from src.models.lstm import build_lstm, whole_window
from src.models.mlp import build_mlp, last_days_flat
from src.models.neural import neural_forecasts

# From simplest to most complex, as the model ladder shows them.
LADDER = ("naive", "vix_implied", "har", "garch", "mlp", "lstm", "transformer_nosent", "transformer")
ROLLING_WINDOW = 63
ROLLING_MODELS = ("har", "garch", "lstm", "transformer")
LOSSES = (("squared_error_log_rv", squared_errors, "mse_log_rv"), ("qlike", qlike_losses, "qlike"))
REGIME_NAMES = ("calm", "middle", "volatile")
FIVE_PERCENT = 0.05
CONCENTRATION_DAYS = 10


@dataclass(frozen=True)
class Rows:
    """Every row of the June 2023 window. dates[i] is the day a forecast was made, target_dates[i] the day it's for."""

    dates: np.ndarray
    target_dates: np.ndarray
    labels: np.ndarray
    actual: np.ndarray
    return_oc: np.ndarray
    forecasts: dict[str, np.ndarray]
    seed_forecasts: dict[str, np.ndarray]  # model -> (n_seeds, n_rows)

    @property
    def is_test(self) -> np.ndarray:
        return self.labels == "test"


def build_rows(paths: RunPaths) -> Rows:
    evaluation, _ = build_evaluation_rows(paths)
    main = evaluation.labels != "later"
    forecasts = {name: evaluation.forecasts[name][main] for name in MODELS}
    dates = evaluation.dates[main]

    saved = read_table(paths.artifacts / "forecasts.csv")
    if not np.array_equal(saved["date"], dates):
        raise RuntimeError("forecasts.csv covers different dates from the frozen models")
    for name in MODELS:
        if not np.allclose(saved[name].astype(np.float64), forecasts[name], rtol=0.0, atol=1e-9):
            raise RuntimeError(f"The saved {name} forecasts differ from what the frozen model gives. Nothing was written.")

    data = load_dataset(paths, with_sentiment=True)
    windows = sorted(data.splits.values(), key=lambda window: window.dates[0])
    if not np.array_equal(np.concatenate([window.dates for window in windows]), dates):
        raise RuntimeError("The neural models' windows cover different dates from the Transformer's")
    x = np.concatenate([window.x for window in windows])
    mlp = neural_forecasts(paths, "mlp", build_mlp, last_days_flat, data, x)
    lstm = neural_forecasts(paths, "lstm", build_lstm, whole_window, data, x)

    session_dates, returns = load_returns(paths)
    forecasts |= {
        "garch": log_variance_forecasts(load_fit(paths), session_dates, returns, dates),
        "mlp": mlp.mean(axis=0),
        "lstm": lstm.mean(axis=0),
    }
    seeds = {
        "transformer": evaluation.seed_forecasts["transformer"][:, main],
        "transformer_nosent": evaluation.seed_forecasts["transformer_nosent"][:, main],
        "mlp": mlp,
        "lstm": lstm,
    }
    return Rows(
        dates, evaluation.target_dates[main], evaluation.labels[main], evaluation.actual[main],
        evaluation.return_oc[main], forecasts, seeds,
    )


def _seed_scores(rows: Rows) -> dict[str, dict]:
    """Each seed's own score, mean and spread (sample standard deviation). The reported forecast is the mean of seeds."""
    test = rows.is_test
    actual = rows.actual[test]
    scores = {}
    for name, seeds in rows.seed_forecasts.items():
        mse = [float(squared_errors(actual, seed[test]).mean()) for seed in seeds]
        qlike = [float(qlike_losses(actual, seed[test]).mean()) for seed in seeds]
        scores[name] = {
            "mse_log_rv": mse, "mse_mean": float(np.mean(mse)), "mse_spread": float(np.std(mse, ddof=1)),
            "qlike": qlike, "qlike_mean": float(np.mean(qlike)), "qlike_spread": float(np.std(qlike, ddof=1)),
        }
    return scores


def _costs(paths: RunPaths) -> dict[str, dict]:
    """Parameter count and training seconds. Naive and VIX-implied have nothing to fit."""
    log = read_log(paths)["models"]
    transformer = json.loads((paths.artifacts / "training_log.json").read_text(encoding="utf-8"))["variants"]
    table = read_table(paths.processed / "daily_features.csv")
    train = table["split"].astype(str) == "train"
    inputs = np.column_stack([table[c][train].astype(np.float64) for c in ("log_rv", "log_rv_w", "log_rv_m")])
    started = time.perf_counter()
    har_fit(inputs, table["target_log_rv"][train].astype(np.float64))
    costs = {
        "naive": {"n_parameters": 0, "training_seconds": None},
        "vix_implied": {"n_parameters": 0, "training_seconds": None},
        "har": {"n_parameters": 4, "training_seconds": time.perf_counter() - started},
    }
    for name in COMPARISON_MODELS:
        costs[name] = {"n_parameters": log[name]["n_parameters"], "training_seconds": log[name]["seconds"]}
    for name, variant in transformer.items():
        costs[name] = {
            "n_parameters": variant["n_parameters"],
            "training_seconds": float(sum(run["seconds"] for run in variant["seed_runs"])),
        }
    return costs


def _dm_tests(actual: np.ndarray, forecasts: dict[str, np.ndarray], base: str, others: tuple[str, ...]) -> list[dict]:
    """Both losses, for one Transformer against each other model. The limit covers just that Transformer's tests."""
    limit = FIVE_PERCENT / (len(others) * len(LOSSES))
    tests = []
    for other in others:
        for loss_name, loss, _ in LOSSES:
            result = diebold_mariano(loss(actual, forecasts[base]), loss(actual, forecasts[other]), DM_LAGS)
            tests.append({
                "model_a": base, "model_b": other, "loss": loss_name, **result,
                "lower_loss": base if result["mean_loss_difference"] < 0 else other,
                "significant_5pct": result["p_value"] < FIVE_PERCENT,
                "bonferroni_limit": limit,
                "significant_bonferroni": result["p_value"] < limit,
            })
    return tests


def _ranking(scores: dict[str, dict], key: str) -> list[str]:
    return sorted(scores, key=lambda name: scores[name][key])


def _regimes(rows: Rows, paths: RunPaths) -> list[dict]:
    """Calm, middle and volatile test days, by the 22-day mean of log RV known when the forecast was made.

    Sorting on a measure known in advance, not on the day's outcome, avoids flattering whichever model
    forecasts high or low. The cut-offs are test-period terciles, used only to describe results.
    """
    test = rows.is_test
    features = read_table(paths.processed / "daily_features.csv")
    level = features["log_rv_m"].astype(np.float64)[np.searchsorted(features["date"], rows.dates[test])]
    low, high = np.quantile(level, [1.0 / 3.0, 2.0 / 3.0])
    masks = (level <= low, (level > low) & (level <= high), level > high)
    actual = rows.actual[test]
    forecasts = {name: rows.forecasts[name][test] for name in LADDER}
    regimes = []
    for name, mask in zip(REGIME_NAMES, masks):
        chosen = {model: forecast[mask] for model, forecast in forecasts.items()}
        scores = {
            model: {
                "mse_log_rv": float(squared_errors(actual[mask], forecast).mean()),
                "qlike": float(qlike_losses(actual[mask], forecast).mean()),
            }
            for model, forecast in chosen.items()
        }
        regimes.append({
            "name": name,
            "n_days": int(mask.sum()),
            "annualised_vol_pct_range": [
                float(np.sqrt(252.0 * np.exp(level[mask].min())) * 100.0),
                float(np.sqrt(252.0 * np.exp(level[mask].max())) * 100.0),
            ],
            "scores": scores,
            "ranking_mse": _ranking(scores, "mse_log_rv"),
            "ranking_qlike": _ranking(scores, "qlike"),
            "diebold_mariano": _dm_tests(actual[mask], chosen, "transformer", tuple(m for m in LADDER if m != "transformer")),
        })
    return regimes


def _rolling(rows: Rows) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """63-day mean QLIKE for each model, and how often the Transformer was ahead of each of the others."""
    test = rows.is_test
    actual = rows.actual[test]
    window = min(ROLLING_WINDOW, actual.size)  # only a very short test run has fewer days than this
    kernel = np.ones(window) / window
    series = {name: np.convolve(qlike_losses(actual, rows.forecasts[name][test]), kernel, "valid") for name in LADDER}
    days = rows.target_dates[test][window - 1 :]
    ours = series["transformer"]
    summary: dict = {"window_days": window, "n_windows": int(ours.size), "against": {}}
    for name, values in series.items():
        if name == "transformer":
            continue
        gap = values - ours  # above zero: the Transformer's loss was lower
        summary["against"][name] = {
            "share_of_windows_ahead": float((gap > 0).mean()),
            "largest_lead": float(gap.max()),
            "largest_lead_window_end": str(days[int(gap.argmax())]),
            "largest_deficit": float(-gap.min()),
            "largest_deficit_window_end": str(days[int(gap.argmin())]),
        }
    return days, series, summary


def _lead_over_har(rows: Rows) -> dict:
    """How widely the QLIKE lead over HAR is spread across days. A description, since it looks only at the best days."""
    test = rows.is_test
    gap = qlike_losses(rows.actual[test], rows.forecasts["har"][test]) - qlike_losses(rows.actual[test], rows.forecasts["transformer"][test])
    best = np.argsort(-gap)[:CONCENTRATION_DAYS]
    rest = np.delete(gap, best)
    return {
        "share_of_days_ahead": float((gap > 0).mean()),
        "n_best_days": int(best.size),
        "share_of_lead_from_best_days": float(gap[best].sum() / gap.sum()),
        "best_days": [str(day) for day in rows.target_dates[test][best]],
        "mean_lead": float(gap.mean()),
        "mean_lead_without_best_days": float(rest.mean()),
    }


def build_comparison(paths: RunPaths, rows: Rows, scores: dict[str, dict]) -> dict:
    test = rows.is_test
    actual = rows.actual[test]
    forecasts = {name: rows.forecasts[name][test] for name in LADDER}
    others = tuple(name for name in LADDER if name != "transformer")
    others_nosent = tuple(name for name in others if name != "transformer_nosent")
    costs = _costs(paths)
    seeds = _seed_scores(rows)
    log = read_log(paths)["models"]
    _, _, rolling = _rolling(rows)
    return {
        "n_days": int(test.sum()),
        "ladder": list(LADDER),
        "models": {
            name: {
                **costs[name],
                "seed_scores": seeds.get(name),
                # What MSE would be with the forecast's average level gap taken out. A description only: no model uses this.
                "mse_without_mean_error": scores[name]["mse_log_rv"] - scores[name]["mean_error_log_rv"] ** 2,
            }
            for name in LADDER
        },
        "ranking_mse": _ranking(scores, "mse_log_rv"),
        "ranking_qlike": _ranking(scores, "qlike"),
        "diebold_mariano": _dm_tests(actual, forecasts, "transformer", others)
        + _dm_tests(actual, forecasts, "transformer_nosent", others_nosent),
        "regimes": _regimes(rows, paths),
        "rolling_qlike": rolling,
        "qlike_lead_over_har": _lead_over_har(rows),
        "garch": {key: log["garch"][key] for key in ("omega", "alpha", "beta", "persistence", "long_run_annual_vol_pct")},
        "recipes": {
            name: {key: log[name][key] for key in ("learning_rate", "weight_decay", "warmup_epochs", "clip_norm")}
            for name in ("mlp", "lstm")
        },
        "added_training_seconds": float(sum(log[name]["seconds"] for name in COMPARISON_MODELS)),
    }


def _p(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def _seconds(value: float | None) -> str:
    if value is None:
        return "nothing to fit"
    return "under 0.001s" if value < 0.001 else f"{value:.3f}s" if value < 1.0 else f"{value:.1f}s"


def _ordinal(position: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(position, f"{position}th")


def _name(model: str) -> str:
    """A model's label for use inside a sentence, where "Transformer, no news" would read as two items."""
    return "the no-news Transformer" if model == "transformer_nosent" else LABELS[model]


def _list(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _rank_sentence(ranking: list[str], score_name: str) -> str:
    position = ranking.index("transformer") + 1
    if position == 1:
        return f"On {score_name} it ranks 1st of {len(ranking)}."
    return f"On {score_name} it ranks {_ordinal(position)} of {len(ranking)}, behind {_list([_name(name) for name in ranking[: position - 1]])}."


def _lead_sentence(test: dict) -> str:
    other = _name(test["model_b"])
    if test["lower_loss"] == "transformer":
        strength = "holds up after the Bonferroni adjustment" if test["significant_bonferroni"] else (
            "is significant at 5% but not after the Bonferroni adjustment" if test["significant_5pct"] else "isn't statistically significant"
        )
        return f"ahead of {other}, and the lead {strength}"
    strength = "and that gap holds up after the Bonferroni adjustment" if test["significant_bonferroni"] else (
        "and that gap is significant at 5% but not after the Bonferroni adjustment" if test["significant_5pct"] else "though the gap isn't statistically significant"
    )
    return f"behind {other}, {strength}"


def _seed_sentences(comparison: dict, scores: dict[str, dict]) -> list[str]:
    """The ranks above score forecasts averaged over seeds. This says how each model did one seed at a time."""
    seeds = {name: model["seed_scores"] for name, model in comparison["models"].items() if model["seed_scores"]}
    sentences = []
    for label, key in (("MSE", "mse"), ("QLIKE", "qlike")):
        order = sorted(seeds, key=lambda name: seeds[name][f"{key}_mean"])
        listing = ", ".join(f"{_name(name)} {seeds[name][f'{key}_mean']:.4f}" for name in order)
        sentences.append(f"One seed at a time, the mean {label} runs in this order: {listing}.")
    spread = seeds["transformer"]["mse_spread"]
    close = [
        _name(name) for name in seeds
        if name != "transformer" and abs(scores[name]["mse_log_rv"] - scores["transformer"]["mse_log_rv"]) < spread
    ]
    if close:
        sentences.append(
            f"The Transformer's MSE varies by {spread:.4f} between seeds, more than its gap to {_list(close)}, so those positions could change with other seeds. "
            "The Diebold-Mariano tests treat each averaged forecast as fixed, so they don't include this."
        )
    return sentences


def write_verdict(paths: RunPaths, comparison: dict, scores: dict[str, dict], metrics: dict) -> str:
    """artifacts/comparison.md. Every number and claim is computed from the results, so it can't drift from them."""
    tests = [t for t in comparison["diebold_mariano"] if t["model_a"] == "transformer"]
    by_key = {(t["model_b"], t["loss"]): t for t in tests}
    ranks_mse, ranks_qlike = comparison["ranking_mse"], comparison["ranking_qlike"]
    period = metrics["test"]
    garch = scores["garch"]
    garch_cost = comparison["models"]["garch"]
    simpler = LADDER[: LADDER.index("transformer")]  # every other model is lower on the ladder
    lost_to_simpler = [
        (_name(name), label) for name in simpler for key, label in (("mse_log_rv", "MSE"), ("qlike", "QLIKE"))
        if scores[name][key] < scores["transformer"][key]
    ]
    lines = [
        "# Model comparison: the verdict",
        "",
        f"Test period: {comparison['n_days']} forecasts for sessions from {period['target_dates'][0]} to {period['target_dates'][1]}. "
        "Lower is better on both scores. \"The Transformer\" means the with-news model unless it says otherwise.",
        "",
        "## Where the Transformer ranks",
        "",
        _rank_sentence(ranks_mse, "MSE on log RV"),
        "",
        _rank_sentence(ranks_qlike, "QLIKE"),
        "",
    ]
    if lost_to_simpler:
        lines += ["**It loses to a simpler model on: " + "; ".join(f"{name} on {label}" for name, label in lost_to_simpler) + ".**", ""]
    else:
        lines += ["It beats every simpler model on both scores.", ""]
    lines += [" ".join(_seed_sentences(comparison, scores)), ""]

    lines += ["## Is each lead statistically significant?", ""]
    lines += [
        f"Diebold-Mariano tests with {DM_LAGS} Newey-West lags. A negative statistic means the Transformer had the lower loss. "
        f"These are {len(tests)} tests, so a Bonferroni limit of {tests[0]['bonferroni_limit']:.4f} applies to them as a family.",
        "",
        "| Against | MSE statistic | MSE p | Lower MSE | QLIKE statistic | QLIKE p | Lower QLIKE |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in LADDER:
        if name == "transformer":
            continue
        mse, qlike = by_key[(name, "squared_error_log_rv")], by_key[(name, "qlike")]
        lines.append(
            f"| {LABELS[name]} | {mse['statistic']:.2f} | {_p(mse['p_value'])} | {LABELS[mse['lower_loss']]} "
            f"| {qlike['statistic']:.2f} | {_p(qlike['p_value'])} | {LABELS[qlike['lower_loss']]} |"
        )
    lines += ["", "In words:", ""]
    for name in LADDER:
        if name == "transformer":
            continue
        lines.append(
            f"- **{LABELS[name]}.** On MSE the Transformer is {_lead_sentence(by_key[(name, 'squared_error_log_rv')])}. "
            f"On QLIKE it is {_lead_sentence(by_key[(name, 'qlike')])}."
        )

    lines += [
        "",
        "## When it does better or worse",
        "",
        "Test days are sorted into three equal groups by the 22-day average of log RV known when the forecast was made, "
        "so the grouping can't favour a model that happened to forecast high or low.",
        "",
        "| Period | Days | 22-day volatility, annualised | Transformer MSE (rank) | HAR MSE | p vs HAR | Transformer QLIKE (rank) | HAR QLIKE | p vs HAR | Best on QLIKE |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for regime in comparison["regimes"]:
        regime_tests = {(t["model_b"], t["loss"]): t for t in regime["diebold_mariano"]}
        ours, har = regime["scores"]["transformer"], regime["scores"]["har"]
        low, high = regime["annualised_vol_pct_range"]
        lines.append(
            f"| {regime['name'].capitalize()} | {regime['n_days']} | {low:.1f}% to {high:.1f}% "
            f"| {ours['mse_log_rv']:.4f} ({regime['ranking_mse'].index('transformer') + 1}) | {har['mse_log_rv']:.4f} "
            f"| {_p(regime_tests[('har', 'squared_error_log_rv')]['p_value'])} "
            f"| {ours['qlike']:.4f} ({regime['ranking_qlike'].index('transformer') + 1}) | {har['qlike']:.4f} "
            f"| {_p(regime_tests[('har', 'qlike')]['p_value'])} | {LABELS[regime['ranking_qlike'][0]]} |"
        )
    calm, volatile = comparison["regimes"][0], comparison["regimes"][-1]
    gap_calm = calm["scores"]["har"]["qlike"] - calm["scores"]["transformer"]["qlike"]
    gap_volatile = volatile["scores"]["har"]["qlike"] - volatile["scores"]["transformer"]["qlike"]
    against_har = comparison["rolling_qlike"]["against"]["har"]
    lead = comparison["qlike_lead_over_har"]
    lines += [
        "",
        f"Against HAR, the Transformer's QLIKE gap is {gap_calm:+.4f} in calm periods and {gap_volatile:+.4f} in volatile ones, where a positive gap means the Transformer is ahead. "
        f"Across rolling {comparison['rolling_qlike']['window_days']}-day windows it's ahead of HAR in {against_har['share_of_windows_ahead'] * 100:.0f}% of them. "
        f"Its largest lead is {against_har['largest_lead']:.3f} in the window ending {against_har['largest_lead_window_end']}, and "
        f"its largest deficit is {against_har['largest_deficit']:.3f} in the window ending {against_har['largest_deficit_window_end']}. "
        f"Day by day it's ahead of HAR on QLIKE only {lead['share_of_days_ahead'] * 100:.0f}% of the time, and its {lead['n_best_days']} best days supply {lead['share_of_lead_from_best_days'] * 100:.0f}% of its lead "
        f"({lead['mean_lead']:.4f} a day with them, {lead['mean_lead_without_best_days']:.4f} without). So the lead comes mainly from a few sessions, the largest on {lead['best_days'][0]} and {lead['best_days'][1]}, not from being a little better every day. "
        "These groups are small and were split after the fact, so treat them as description, not proof. "
        f"\"Volatile\" is relative to this period too: the 22-day figure tops out at {volatile['annualised_vol_pct_range'][1]:.1f}% annualised, "
        "because the 2020 crash fell in the validation period. So this says little about how any model would do in a crisis.",
        "",
        "## 99% VaR",
        "",
        f"The same rule for every model: 2.33 x sqrt(exp(forecast)). A sound VaR would breach about {period['models']['transformer']['var99']['expected_breaches']:.1f} times. Kupiec fails above {metrics['kupiec_critical_value']}.",
        "",
        "| Model | Breaches | Kupiec LR | Result |",
        "|---|---|---|---|",
    ]
    for name in LADDER:
        var = scores[name]["var99"]
        lines.append(f"| {LABELS[name]} | {var['breaches']} | {var['kupiec_lr']:.2f} | {'fail' if var['kupiec_rejected'] else 'pass'} |")
    passing = [_name(name) for name in LADDER if not scores[name]["var99"]["kupiec_rejected"]]
    ratio = period["var_diagnostics"]["mean_squared_return_over_mean_rv"]
    lines += [
        "",
        (f"Passing: {_list(passing)}." if passing else "No model passes.")
        + (f" A forecast of RV runs low for this purpose however accurate it is, because the squared open-to-close return averages {ratio:.2f} times RV." if ratio > 1.05 else "")
        + (f" GARCH targets the return's own variance, and its forecasts sit {garch['mean_error_log_rv']:+.3f} log units from log RV on average, which is why its line is crossed less often." if not garch["var99"]["kupiec_rejected"] and garch["mean_error_log_rv"] > 0 else "")
        + " A model that passes here isn't necessarily the better volatility forecaster: VaR only counts how often one line was crossed.",
        "",
        "## All the scores",
        "",
        "For the neural models, the score is that of the forecast averaged over 3 seeds, and \"seed mean ± spread\" is each seed's own score, with the sample standard deviation.",
        "",
        "| Model | MSE | MSE, seed mean ± spread | QLIKE | QLIKE, seed mean ± spread | Parameters | Training time |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in LADDER:
        cost = comparison["models"][name]
        seed = cost["seed_scores"]
        seconds = _seconds(cost["training_seconds"])
        lines.append(
            f"| {LABELS[name]} | {scores[name]['mse_log_rv']:.4f} | {f'''{seed['mse_mean']:.4f} ± {seed['mse_spread']:.4f}''' if seed else 'n/a'} "
            f"| {scores[name]['qlike']:.4f} | {f'''{seed['qlike_mean']:.4f} ± {seed['qlike_spread']:.4f}''' if seed else 'n/a'} "
            f"| {cost['n_parameters']:,} | {seconds} |"
        )
    recipe = comparison["recipes"]["mlp"]
    lines += [
        "",
        f"Added training time for GARCH, the MLP and the LSTM: {comparison['added_training_seconds']:.1f}s.",
        "",
        "## Read this before relying on it",
        "",
        f"- **The training settings aren't identical.** The Transformer's were chosen on validation data: learning rate 5e-4, weight decay 1e-3, a 5-epoch warm-up, and last-day pooling after mean pooling did poorly. The MLP and LSTM use the settings the brief gave (learning rate {recipe['learning_rate']:g}, weight decay {recipe['weight_decay']:g}, no warm-up) with no tuning, as it asked. A tuned MLP or LSTM might do better, so this tests them as specified, not at their best.",
        f"- **GARCH is scored on a target it doesn't model.** It forecasts the variance of the open-to-close return, not log RV. Its forecasts average {garch['mean_error_log_rv']:+.3f} log units from log RV, and without that average gap its MSE would be {garch_cost['mse_without_mean_error']:.4f} rather than {garch['mse_log_rv']:.4f}. That figure only describes the gap, because no forecast could have known it in advance.",
        "- **The MLP and LSTM use 18 features, not the brief's 17,** following the project's earlier decision on the weekday columns. So the MLP has 90 inputs rather than 85. They use the news features, like the with-news Transformer.",
        "- **The Transformer's test scores were already known** when these models were added. Nothing about the new models was chosen after seeing test results, and their checkpoints were checksummed before any test forecast was made.",
        "- **Unadjusted p-values overstate the evidence** when many tests are run, which is why the Bonferroni limit is given.",
        "- **The later check isn't extended.** These models were only scored on the test period, so nothing here says how they did after June 2023.",
        "",
    ]
    text = "\n".join(lines)
    (paths.artifacts / "comparison.md").write_text(text, encoding="utf-8")
    return text


def update_artifacts(paths: RunPaths, rows: Rows, comparison: dict, new_scores: dict[str, dict], metrics: dict) -> None:
    """Add to forecasts.csv, var_series.csv, metrics.json and the model card. Existing values stay as they are."""
    forecast_columns: dict[str, np.ndarray] = {name: rows.forecasts[name] for name in COMPARISON_MODELS}
    for name in COMPARISON_MODELS:
        forecast_columns[f"var99_{name}"] = var99(rows.forecasts[name])
        forecast_columns[f"breach_{name}"] = var_breaches(rows.forecasts[name], rows.return_oc).astype(int)
    append_columns(paths.artifacts / "forecasts.csv", forecast_columns)

    test = rows.is_test
    append_columns(
        paths.artifacts / "var_series.csv",
        {name: values[test] for name, values in forecast_columns.items() if name.startswith(("var99_", "breach_"))},
    )

    metrics["test"]["models"].update(new_scores)
    log = read_log(paths)["models"]
    metrics["validation_mse_log_rv"].update({name: log[name]["val_mse_log_rv"] for name in COMPARISON_MODELS})
    metrics["comparison"] = comparison
    write_json(paths.artifacts / "metrics.json", metrics)

    card = json.loads((paths.artifacts / "model_card.json").read_text(encoding="utf-8"))
    card["parameter_count"].update({name: log[name]["n_parameters"] for name in COMPARISON_MODELS})
    card["gradcheck"] = gradcheck_summary()
    write_json(paths.artifacts / "model_card.json", card)


def _print_summary(comparison: dict, scores: dict[str, dict]) -> None:
    print(f"\n  {'model':20s} {'MSE':>8s} {'QLIKE':>8s} {'bias':>8s} {'params':>7s} {'seconds':>9s} {'breaches':>9s} {'Kupiec':>8s}")
    for name in comparison["ladder"]:
        score, cost = scores[name], comparison["models"][name]
        seconds = "-" if cost["training_seconds"] is None else f"{cost['training_seconds']:.3f}"
        var = score["var99"]
        print(
            f"  {name:20s} {score['mse_log_rv']:8.4f} {score['qlike']:8.4f} {score['mean_error_log_rv']:8.3f} {cost['n_parameters']:7d} "
            f"{seconds:>9s} {var['breaches']:9d} {var['kupiec_lr']:7.2f}{'*' if var['kupiec_rejected'] else ' '}"
        )
    print("  Diebold-Mariano, Transformer against each (negative: the Transformer had the lower loss):")
    for test in comparison["diebold_mariano"]:
        if test["model_a"] == "transformer":
            print(f"    vs {test['model_b']:20s} {test['loss']:22s} statistic {test['statistic']:7.2f}, p = {_p(test['p_value']):>7s}, lower loss: {test['lower_loss']}")
    print(f"Added training time: {comparison['added_training_seconds']:.1f}s")


def run_comparison(paths: RunPaths) -> dict:
    """Check both freezes, forecast the new models, score everything, and write the outputs."""
    print(f"\n=== Model comparison ({paths.data_source} data) ===")
    verify_frozen(paths)
    verify_comparison_frozen(paths)
    rows = build_rows(paths)
    metrics = json.loads((paths.artifacts / "metrics.json").read_text(encoding="utf-8"))
    test = rows.is_test
    new_scores = {name: score_model(rows.actual[test], rows.forecasts[name][test], rows.return_oc[test]) for name in COMPARISON_MODELS}
    scores = {**{name: metrics["test"]["models"][name] for name in MODELS}, **new_scores}
    comparison = build_comparison(paths, rows, scores)

    days, series, rolling = _rolling(rows)
    rolling_qlike(paths, days, {name: series[name] for name in ROLLING_MODELS}, rolling["window_days"])
    model_ladder(paths, list(LADDER), [scores[name]["qlike"] for name in LADDER])
    write_verdict(paths, comparison, scores, metrics)  # first, so a failure here leaves the results as they were
    update_artifacts(paths, rows, comparison, new_scores, metrics)
    _print_summary(comparison, scores)
    print(f"Wrote comparison.md, metrics.json, forecasts.csv, var_series.csv and two figures to {paths.artifacts}")
    return comparison


if __name__ == "__main__":
    run_comparison(REAL_PATHS)
