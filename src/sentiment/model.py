"""Softmax regression (vocabulary -> 3 classes), its training loop and its metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.nn.functional import stable_softmax
from src.nn.optim import AdamW
from src.sentiment.text import SparseRows

CLASSES = ("negative", "neutral", "positive")
NEGATIVE, NEUTRAL, POSITIVE = 0, 1, 2

# Fixed in advance. These are not tuned on the evaluation split.
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 15


class SoftmaxRegression:
    def __init__(self, n_features: int, n_classes: int = len(CLASSES)) -> None:
        # Cross-entropy is convex in these weights, so a zero start is safe.
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.W, self.dW), (self.b, self.db)]

    def logits(self, x: SparseRows) -> np.ndarray:
        return x.dot(self.W) + self.b

    def predict_proba(self, x: SparseRows) -> np.ndarray:
        return stable_softmax(self.logits(x))

    def loss(self, x: SparseRows, y: np.ndarray) -> float:
        """Mean cross-entropy."""
        probabilities = self.predict_proba(x)
        return float(-np.mean(np.log(probabilities[np.arange(y.size), y] + 1e-300)))

    def loss_and_grad(self, x: SparseRows, y: np.ndarray) -> float:
        """Mean cross-entropy, storing its gradient: dL/dW = X^T (p - onehot(y)) / n."""
        probabilities = self.predict_proba(x)
        residual = probabilities.copy()
        residual[np.arange(y.size), y] -= 1.0
        self.dW[...] = x.transpose_dot(residual) / y.size
        self.db[...] = residual.mean(axis=0)
        return float(-np.mean(np.log(probabilities[np.arange(y.size), y] + 1e-300)))


@dataclass(frozen=True)
class TrainResult:
    best_epoch: int
    epochs_run: int
    best_val_loss: float


def train(
    model: SoftmaxRegression,
    x_train: SparseRows,
    y_train: np.ndarray,
    x_val: SparseRows,
    y_val: np.ndarray,
    seed: int,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> TrainResult:
    """Train with AdamW, stop when validation loss stalls, and restore the best weights."""
    rng = np.random.default_rng(seed)
    optimiser = AdamW(model.params(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_loss, best_epoch = np.inf, 0
    best_W, best_b = model.W.copy(), model.b.copy()
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(y_train.size)
        for start in range(0, order.size, BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            model.loss_and_grad(x_train.select(batch), y_train[batch])
            optimiser.step()
        val_loss = model.loss(x_val, y_val)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_W[...], best_b[...] = model.W, model.b
        elif epoch - best_epoch >= patience:
            break
    model.W[...], model.b[...] = best_W, best_b
    return TrainResult(best_epoch, epoch, float(best_loss))


def stratified_split(labels: np.ndarray, holdout_share: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(train rows, holdout rows), with each class keeping its share in both."""
    rng = np.random.default_rng(seed)
    train_rows, holdout_rows = [], []
    for label in np.unique(labels):
        rows = rng.permutation(np.flatnonzero(labels == label))
        n_holdout = int(round(rows.size * holdout_share))
        holdout_rows.append(rows[:n_holdout])
        train_rows.append(rows[n_holdout:])
    return np.sort(np.concatenate(train_rows)), np.sort(np.concatenate(holdout_rows))


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = len(CLASSES)) -> dict:
    """Accuracy, macro F1, per-class precision/recall/F1 and the confusion matrix."""
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted = confusion.sum(axis=0)
    actual = confusion.sum(axis=1)
    precision = np.divide(true_positive, predicted, out=np.zeros(n_classes), where=predicted > 0)
    recall = np.divide(true_positive, actual, out=np.zeros(n_classes), where=actual > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros(n_classes), where=denominator > 0)
    return {
        "accuracy": float(true_positive.sum() / confusion.sum()),
        "macro_f1": float(f1.mean()),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "confusion_rows_true_cols_pred": confusion.tolist(),
    }
