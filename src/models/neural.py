"""Train and reload the MLP and LSTM with the Transformer's training loop, seeds and data.

Test windows are never passed to train_neural. Forecasts for them are made once, by src/eval/compare.py.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np

from src.config import RunPaths
from src.models.comparison import model_folder, record_training
from src.models.dataset import Dataset, Windows, load_dataset
from src.models.train import SEEDS, Recipe, forecast_only, load_checkpoint, save_checkpoint, train_model

# The brief's numbers. The Transformer itself ended up on 5e-4 and 1e-3 with a 5-epoch warm-up, chosen on
# validation data. The comparison reports this, because these two models didn't get that tuning.
NEURAL_RECIPE = Recipe(learning_rate=1e-3, weight_decay=1e-4, warmup_epochs=1, clip_norm=math.inf)

Prepare = Callable[[Windows], Windows]
Build = Callable[[int, int], object]


def _checkpoint(paths: RunPaths, name: str, seed: int):
    return model_folder(paths) / f"{name}_seed{seed}.npz"


def train_neural(paths: RunPaths, name: str, build: Build, prepare: Prepare, recipe: Recipe) -> dict:
    """Every seed on train, stopping on validation. Saves checkpoints, logs and prints the timing."""
    print(f"\n=== {name.upper()} ({paths.data_source} data) ===")
    started = time.perf_counter()
    data = load_dataset(paths, with_sentiment=True)
    train, val = prepare(data.splits["train"]), prepare(data.splits["val"])
    n_features = len(data.feature_names)
    n_parameters = build(n_features, SEEDS[0]).n_parameters()
    print(f"{n_features} features, input {train.x.shape[1:]}, {n_parameters} parameters, "
          f"{train.y.shape[0]} train and {val.y.shape[0]} validation rows")

    actual = data.to_log_rv(val.y)
    seed_logs, val_forecasts = [], []
    for seed in SEEDS:
        model = build(n_features, seed)
        run = train_model(model, train, val, seed, recipe=recipe)
        save_checkpoint(_checkpoint(paths, name, seed), model, data, seed)
        forecast = data.to_log_rv(forecast_only(model, val.x))
        val_forecasts.append(forecast)
        mse = float(np.mean((forecast - actual) ** 2))
        seed_logs.append({
            "seed": seed, "best_epoch": run.best_epoch, "epochs_run": run.epochs_run, "val_mse_log_rv": mse,
            "max_grad_norm_before_clip": run.max_grad_norm, "seconds": run.seconds,
            "train_curve": run.train_curve, "val_curve": run.val_curve,
        })
        print(f"  seed {seed}: best epoch {run.best_epoch:3d} of {run.epochs_run:3d}, validation MSE {mse:.4f}, {run.seconds:.1f}s")

    ensemble = float(np.mean((np.mean(val_forecasts, axis=0) - actual) ** 2))
    seconds = time.perf_counter() - started
    print(f"  mean of seeds: validation MSE {ensemble:.4f}. Total {seconds:.1f}s")
    entry = {
        "n_parameters": n_parameters, "features": list(data.feature_names), "input_shape": list(train.x.shape[1:]),
        "n_train_rows": train.y.shape[0], "n_val_rows": val.y.shape[0], "seconds": seconds, "val_mse_log_rv": ensemble,
        "learning_rate": recipe.learning_rate, "weight_decay": recipe.weight_decay,
        "warmup_epochs": recipe.warmup_epochs, "clip_norm": None if math.isinf(recipe.clip_norm) else recipe.clip_norm,
        "seed_runs": seed_logs,
    }
    record_training(paths, name, entry)
    return entry


def neural_forecasts(paths: RunPaths, name: str, build: Build, prepare: Prepare, data: Dataset, x: np.ndarray) -> np.ndarray:
    """Forecasts (n_seeds, n) in log RV, for 22-day windows x, after checking the scaling is the same."""
    forecasts = []
    windows = prepare(Windows(x, np.empty((x.shape[0], 1)), np.empty(x.shape[0], dtype="datetime64[D]")))
    for seed in SEEDS:
        model, meta = load_checkpoint(_checkpoint(paths, name, seed), build)
        if meta["feature_names"] != data.feature_names or not np.allclose(meta["scaler_mean"], data.scaler.mean):
            raise RuntimeError(f"{name} seed {seed} was trained on different features or scaling")
        forecasts.append(data.to_log_rv(forecast_only(model, windows.x))[:, 0])
    return np.array(forecasts)
