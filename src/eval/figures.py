"""Report figures. matplotlib is used here and nowhere else."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # files only, no window

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.config import RunPaths
from src.eval.evaluate import Evaluation
from src.eval.metrics import squared_errors, var99, var_breaches
from src.models.transformer import WINDOW_DAYS

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated for colour-blind readers, all pairs
NEUTRAL_LINE, NEUTRAL_BAR, CRITICAL = "#a8a7a0", "#c3c2b7", "#d03b3b"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f2f7fd", "#9ec5f4", "#2a78d6", "#104281"])
LABELS = {
    "transformer": "Transformer",
    "transformer_nosent": "Transformer, no news",
    "har": "HAR-RV",
    "naive": "Naive",
    "vix_implied": "VIX-implied",
    "garch": "GARCH(1,1)",
    "mlp": "MLP",
    "lstm": "LSTM",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "axes.titlelocation": "left",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})


def _save(fig: plt.Figure, paths: RunPaths, name: str) -> None:
    if paths.is_synthetic:
        fig.text(0.99, 1.03, "SYNTHETIC DATA: NOT REPORTABLE", ha="right", va="bottom", color=CRITICAL, fontweight="bold")
    folder = paths.artifacts / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _days(values: np.ndarray) -> np.ndarray:
    return values.astype("datetime64[D]").astype("O")


def forecast_vs_actual(paths: RunPaths, ev: Evaluation) -> None:
    test = ev.labels == "test"
    days = _days(ev.target_dates[test])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(days, ev.actual[test], color=NEUTRAL_LINE, linewidth=0.8, label="Actual")
    ax.plot(days, ev.forecasts["har"][test], color=ORANGE, linewidth=1.2, label=LABELS["har"])
    ax.plot(days, ev.forecasts["transformer"][test], color=BLUE, linewidth=1.2, label=LABELS["transformer"])
    ax.set_title("Next-day log realized variance, test period")
    ax.set_ylabel("log RV")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    _save(fig, paths, "forecast_vs_actual")


def cumulative_loss_difference(paths: RunPaths, ev: Evaluation) -> None:
    test = ev.labels == "test"
    days = _days(ev.target_dates[test])
    ours = squared_errors(ev.actual[test], ev.forecasts["transformer"][test])
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, colour in (("har", ORANGE), ("naive", AQUA), ("transformer_nosent", BLUE)):
        line = np.cumsum(squared_errors(ev.actual[test], ev.forecasts[name][test]) - ours)
        ax.plot(days, line, color=colour, linewidth=1.5, label=f"Against {LABELS[name]}")
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.set_title("Cumulative squared-error advantage of the Transformer, test period")
    ax.set_ylabel("Above zero: Transformer ahead")
    ax.legend(loc="best")
    _save(fig, paths, "cumulative_loss_difference")


def model_errors(paths: RunPaths, ev: Evaluation) -> None:
    models = ev.metrics["test"]["models"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    panels = (("mse_log_rv", "MSE on log RV (lower is better)"), ("qlike", "QLIKE (lower is better)"))
    for ax, (key, title) in zip(axes, panels):
        names = sorted(models, key=lambda name: -models[name][key])  # the best model ends up on top
        values = [models[name][key] for name in names]
        colours = [BLUE if name == "transformer" else NEUTRAL_BAR for name in names]
        ax.barh([LABELS[name] for name in names], values, color=colours, edgecolor=SURFACE, linewidth=2)
        for row, value in enumerate(values):
            ax.annotate(f"{value:.3f}", (value, row), xytext=(4, 0), textcoords="offset points", va="center", color=INK)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0.0, max(values) * 1.18)
    fig.tight_layout()
    _save(fig, paths, "model_errors")


def var_chart(paths: RunPaths, ev: Evaluation) -> None:
    test = ev.labels == "test"
    days = _days(ev.target_dates[test])
    forecast, return_oc = ev.forecasts["transformer"][test], ev.return_oc[test]
    breached = var_breaches(forecast, return_oc)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(days, -return_oc * 100.0, color=NEUTRAL_LINE, linewidth=0.7, label="Open-to-close loss")
    ax.plot(days, var99(forecast) * 100.0, color=BLUE, linewidth=1.5, label="99% VaR")
    ax.scatter(
        days[breached], -return_oc[breached] * 100.0, s=36, color=CRITICAL, edgecolor=SURFACE, linewidth=1, zorder=3,
        label=f"Breach ({int(breached.sum())})",
    )
    ax.set_title("Transformer 99% VaR against the actual loss, test period")
    ax.set_ylabel("Loss, % (a gain is below zero)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    _save(fig, paths, "var_breaches")


def attention_heatmap(paths: RunPaths, ev: Evaluation) -> None:
    """Colours span the observed range, because weights sit close to uniform (1/22, about 0.045)."""
    mean_attention = ev.test_attention.mean(axis=0)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1.0]})
    image = left.imshow(mean_attention, cmap=SEQUENTIAL)
    ticks = np.arange(WINDOW_DAYS - 1, -1, -7)
    labels = ["t" if tick == WINDOW_DAYS - 1 else f"t-{WINDOW_DAYS - 1 - tick}" for tick in ticks]
    left.set_xticks(ticks, labels)
    left.set_yticks(ticks, labels)
    left.grid(False)
    left.set_xlabel("Day attended to")
    left.set_ylabel("Day paying attention")
    left.set_title("Mean attention over all test windows")
    fig.colorbar(image, ax=left, label="Attention weight (uniform = 0.045)", shrink=0.85)

    right.bar(np.arange(WINDOW_DAYS), mean_attention[-1], color=BLUE, edgecolor=SURFACE, linewidth=1)
    right.axhline(1.0 / WINDOW_DAYS, color=MUTED, linewidth=0.8, linestyle="--", label="Uniform")
    right.set_xticks(ticks, labels)
    right.grid(axis="x", visible=False)
    right.set_xlabel("Day attended to")
    right.set_ylabel("Attention weight")
    right.set_title("Day t's row, the one that feeds the forecast directly")
    right.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, paths, "attention_heatmap")


def training_curves(paths: RunPaths) -> None:
    log = json.loads((paths.artifacts / "training_log.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, (name, variant) in zip(axes, log["variants"].items()):
        for run, colour in zip(variant["seed_runs"], (BLUE, ORANGE, AQUA)):
            epochs = np.arange(1, run["epochs_run"] + 1)
            ax.plot(epochs, run["val_curve"], color=colour, linewidth=1.5, label=f"Seed {run['seed']}")
            best = run["best_epoch"]
            ax.scatter(best, run["val_curve"][best - 1], s=36, color=colour, edgecolor=SURFACE, zorder=3)
        ax.set_title(LABELS[name])
        ax.set_xlabel("Epoch (the dot marks the kept weights)")
    axes[0].set_ylabel("Validation MSE, standardised")
    axes[0].legend()
    _save(fig, paths, "training_curves")


def rolling_qlike(paths: RunPaths, days: np.ndarray, series: dict[str, np.ndarray], window: int) -> None:
    """The references stay grey, told apart by line style and a label at the end of each line."""
    styles = {"har": ("#52514e", "--"), "garch": ("#8f8e88", ":"), "lstm": ("#6f6e69", "-")}
    x = _days(days)
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, values in series.items():
        colour, style = styles.get(name, (BLUE, "-"))
        ax.plot(x, values, color=colour, linestyle=style, linewidth=2.0 if name == "transformer" else 1.3, label=LABELS[name])
    ax.set_title(f"Rolling {window}-day QLIKE, test period (lower is better)")
    ax.set_ylabel("QLIKE")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(series))
    _save(fig, paths, "rolling_qlike")


def model_ladder(paths: RunPaths, names: list[str], qlike: list[float]) -> None:
    """Simplest model at the top, most complex at the bottom, so the eye reads down the ladder."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colours = [BLUE if name.startswith("transformer") else NEUTRAL_BAR for name in names]
    ax.barh([LABELS[name] for name in names], qlike, color=colours, edgecolor=SURFACE, linewidth=2)
    for row, value in enumerate(qlike):
        ax.annotate(f"{value:.3f}", (value, row), xytext=(4, 0), textcoords="offset points", va="center", color=INK)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.0, max(qlike) * 1.12)
    ax.set_title("Test QLIKE, from the simplest model to the most complex (lower is better)")
    fig.tight_layout()
    _save(fig, paths, "model_ladder")


def write_figures(paths: RunPaths, evaluation: Evaluation) -> None:
    forecast_vs_actual(paths, evaluation)
    cumulative_loss_difference(paths, evaluation)
    model_errors(paths, evaluation)
    var_chart(paths, evaluation)
    attention_heatmap(paths, evaluation)
    training_curves(paths)
    print(f"Figures written to {paths.artifacts / 'figures'}")
