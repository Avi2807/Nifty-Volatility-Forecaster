"""Train both Transformer variants on train and validation windows only.

Test windows are never passed to anything in this module. Forecasts are made after training
is frozen, in the evaluation stage.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

import numpy as np

from src.config import RunPaths
from src.data.loaders import read_table
from src.eval.references import har_fit, har_forecast, naive_forecast
from src.models.dataset import Dataset, Windows, load_dataset
from src.models.transformer import TransformerForecaster
from src.nn.losses import MSELoss
from src.nn.optim import AdamW, clip_grad_norm, warmup_lr

BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
WARMUP_EPOCHS = 5
CLIP_NORM = 1.0
PATIENCE = 15
MAX_EPOCHS = 200
SEEDS = (1, 2, 3)
PREDICT_BATCH = 512

VARIANTS = {"transformer": True, "transformer_nosent": False}  # name -> uses the news features


@dataclass(frozen=True)
class Recipe:
    """What the loop lets a model vary. The defaults are the frozen Transformer's settings."""

    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    warmup_epochs: int = WARMUP_EPOCHS  # 1 means no warm-up
    clip_norm: float = CLIP_NORM  # math.inf means no clipping


@dataclass
class TrainingRun:
    best_epoch: int = 0
    epochs_run: int = 0
    best_val_mse: float = np.inf  # in standardised target units
    seconds: float = 0.0
    max_grad_norm: float = 0.0
    train_curve: list[float] = field(default_factory=list)
    val_curve: list[float] = field(default_factory=list)


def forecast_only(model, x: np.ndarray) -> np.ndarray:
    """Forecasts (n, 1) with dropout switched off, for any model."""
    model.set_training(False)
    return np.vstack([model.forward(x[start : start + PREDICT_BATCH]) for start in range(0, x.shape[0], PREDICT_BATCH)])


def predict(model: TransformerForecaster, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts (n, 1) and head-averaged attention (n, 22, 22), with dropout switched off."""
    model.set_training(False)
    forecasts, attention = [], []
    for start in range(0, x.shape[0], PREDICT_BATCH):
        forecasts.append(model.forward(x[start : start + PREDICT_BATCH]))
        attention.append(model.attention_weights())
    return np.vstack(forecasts), np.vstack(attention)


def train_model(
    model,
    train: Windows,
    val: Windows,
    seed: int,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    recipe: Recipe = Recipe(),
) -> TrainingRun:
    """AdamW with warm-up and clipping. Stops on validation MSE and restores the best weights."""
    started = time.perf_counter()
    shuffle_rng = np.random.default_rng(seed + 1000)
    optimiser = AdamW(model.params(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay)
    loss = MSELoss()
    run = TrainingRun()
    best_weights = model.get_weights()

    for epoch in range(max_epochs):
        optimiser.lr = warmup_lr(recipe.learning_rate, epoch, recipe.warmup_epochs)
        model.set_training(True)
        order = shuffle_rng.permutation(train.y.shape[0])  # windows shuffle; the split itself never does
        batch_losses = []
        for start in range(0, order.size, BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            batch_losses.append(loss.forward(model.forward(train.x[batch]), train.y[batch]))
            model.backward(loss.backward())
            run.max_grad_norm = max(run.max_grad_norm, clip_grad_norm(model.params(), recipe.clip_norm))
            optimiser.step()

        val_mse = float(np.mean((forecast_only(model, val.x) - val.y) ** 2))
        if not np.isfinite(val_mse):
            raise FloatingPointError(f"Validation loss became {val_mse} at epoch {epoch + 1}, seed {seed}")
        run.train_curve.append(float(np.mean(batch_losses)))
        run.val_curve.append(val_mse)
        run.epochs_run = epoch + 1
        if val_mse < run.best_val_mse:
            run.best_val_mse, run.best_epoch = val_mse, epoch + 1
            best_weights = model.get_weights()
        elif epoch + 1 - run.best_epoch >= patience:
            break

    model.set_weights(best_weights)
    run.seconds = time.perf_counter() - started
    return run


def save_checkpoint(path, model, data: Dataset, seed: int) -> None:
    """Weights plus everything needed to reuse them later, including the training-period scaling."""
    arrays = {f"param_{i}": value for i, value in enumerate(model.get_weights())}
    np.savez(
        path, **arrays, feature_names=np.array(data.feature_names), scaler_mean=data.scaler.mean,
        scaler_std=data.scaler.std, target_mean=data.target_mean, target_std=data.target_std, seed=seed,
    )


def load_checkpoint(path, build=TransformerForecaster) -> tuple:
    """`build(n_features, seed)` makes the empty model. It defaults to the Transformer."""
    with np.load(path) as saved:
        names = tuple(str(name) for name in saved["feature_names"])
        model = build(len(names), int(saved["seed"]))
        model.set_weights([saved[f"param_{i}"] for i in range(len(model.params()))])
        meta = {
            "feature_names": names, "scaler_mean": saved["scaler_mean"], "scaler_std": saved["scaler_std"],
            "target_mean": float(saved["target_mean"]), "target_std": float(saved["target_std"]),
        }
    model.set_training(False)
    return model, meta


def _validation_references(paths: RunPaths) -> dict[str, float]:
    """Naive and HAR validation MSE in log RV units. A yardstick only: nothing is selected on it."""
    table = read_table(paths.processed / "daily_features.csv")
    labels = table["split"].astype(str)
    har_inputs = np.column_stack([table[c].astype(np.float64) for c in ("log_rv", "log_rv_w", "log_rv_m")])
    y = table["target_log_rv"].astype(np.float64)
    coefficients = har_fit(har_inputs[labels == "train"], y[labels == "train"])
    is_val = labels == "val"
    return {
        "naive": float(np.mean((naive_forecast(har_inputs[is_val, 0]) - y[is_val]) ** 2)),
        "har": float(np.mean((har_forecast(har_inputs[is_val], coefficients) - y[is_val]) ** 2)),
    }


def write_freeze_record(paths: RunPaths) -> None:
    """Checksums of every checkpoint, so anyone can verify the models weren't changed after testing."""
    checkpoints = sorted((paths.artifacts / "models").glob("*.npz"))
    record = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints}
    (paths.artifacts / "frozen_models.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def run_training(paths: RunPaths) -> dict:
    """Train every variant and seed, save checkpoints, write the log and print the timing."""
    print(f"\n=== Training ({paths.data_source} data) ===")
    started = time.perf_counter()
    checkpoint_dir = paths.artifacts / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log: dict = {"data_source": paths.data_source, "variants": {}, "seeds": list(SEEDS)}
    references = _validation_references(paths)

    for name, with_sentiment in VARIANTS.items():
        data = load_dataset(paths, with_sentiment)
        train, val = data.splits["train"], data.splits["val"]
        n_parameters = TransformerForecaster(len(data.feature_names), SEEDS[0]).n_parameters()
        print(f"{name}: {len(data.feature_names)} features, {n_parameters} parameters, "
              f"{train.y.shape[0]} train and {val.y.shape[0]} validation windows")
        seed_logs, val_forecasts = [], []
        for seed in SEEDS:
            model = TransformerForecaster(len(data.feature_names), seed)
            run = train_model(model, train, val, seed)
            save_checkpoint(checkpoint_dir / f"{name}_seed{seed}.npz", model, data, seed)
            forecast = data.to_log_rv(predict(model, val.x)[0])
            val_forecasts.append(forecast)
            actual = data.to_log_rv(val.y)
            mse = float(np.mean((forecast - actual) ** 2))
            seed_logs.append({
                "seed": seed, "best_epoch": run.best_epoch, "epochs_run": run.epochs_run, "val_mse_log_rv": mse,
                "max_grad_norm_before_clip": run.max_grad_norm, "seconds": run.seconds,
                "train_curve": run.train_curve, "val_curve": run.val_curve,
            })
            print(f"  seed {seed}: best epoch {run.best_epoch:3d} of {run.epochs_run:3d}, "
                  f"validation MSE {mse:.4f}, {run.seconds:.1f}s")
        ensemble_mse = float(np.mean((np.mean(val_forecasts, axis=0) - data.to_log_rv(val.y)) ** 2))
        print(f"  mean of seeds: validation MSE {ensemble_mse:.4f}")
        log["variants"][name] = {
            "features": list(data.feature_names), "n_parameters": n_parameters, "n_train_windows": train.y.shape[0],
            "n_val_windows": val.y.shape[0], "seed_runs": seed_logs, "ensemble_val_mse_log_rv": ensemble_mse,
        }

    log["validation_references_mse_log_rv"] = references
    log["training_seconds"] = time.perf_counter() - started
    (paths.artifacts / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    write_freeze_record(paths)
    print(f"Validation references: naive {references['naive']:.4f}, HAR {references['har']:.4f}")
    print(f"Training time for all {len(VARIANTS) * len(SEEDS)} models: {log['training_seconds']:.1f}s")
    return log
