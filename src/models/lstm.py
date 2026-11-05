"""LSTM: 22 days of the same standardised features, last hidden state, to next-day log RV.

Usage: python -m src.models.lstm

One cell holds a single stacked weight of shape (n_in + n_hidden, 4 * n_hidden). With z = [x, h]:

    z W + b splits into four gates, in the order forget f, input i, candidate g, output o
    f, i, o = sigmoid(...)    g = tanh(...)
    c' = f c + i g            h' = o tanh(c')
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.config import REAL_PATHS, RunPaths
from src.models.dataset import Windows
from src.models.neural import NEURAL_RECIPE, train_neural
from src.models.train import Recipe
from src.nn.layers import Dense, Dropout, LastDayPool, Layer, ParamPairs, Sequential

N_HIDDEN = 32
DROPOUT = 0.1
RECIPE: Recipe = replace(NEURAL_RECIPE, clip_norm=1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class CellStep:
    """What backward needs from one forward step."""

    z: np.ndarray
    c: np.ndarray
    forget: np.ndarray
    input: np.ndarray
    candidate: np.ndarray
    output: np.ndarray
    tanh_c: np.ndarray


class LSTMCell:
    def __init__(self, n_in: int, n_hidden: int, rng: np.random.Generator) -> None:
        self.n_in, self.n_hidden = n_in, n_hidden
        # Xavier normal, counting each gate as n_hidden outputs.
        self.W = rng.normal(0.0, np.sqrt(2.0 / (n_in + 2 * n_hidden)), (n_in + n_hidden, 4 * n_hidden))
        self.b = np.zeros(4 * n_hidden)
        self.b[:n_hidden] = 1.0  # the forget gate starts open, so memory survives early training
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.step: CellStep | None = None

    def run_step(self, x: np.ndarray, h: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, CellStep]:
        z = np.concatenate([x, h], axis=-1)
        gates = z @ self.W + self.b
        forget, in_gate, candidate, out_gate = np.split(gates, 4, axis=-1)
        step = CellStep(z, c, _sigmoid(forget), _sigmoid(in_gate), np.tanh(candidate), _sigmoid(out_gate), np.empty(0))
        new_c = step.forget * c + step.input * step.candidate
        tanh_c = np.tanh(new_c)
        return step.output * tanh_c, new_c, replace(step, tanh_c=tanh_c)

    def back_step(self, step: CellStep, dh: np.ndarray, dc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """From dL/dh' and dL/dc' to (dx, dh, dc, gate gradients), leaving the weights' gradients to the caller."""
        total_dc = dc + dh * step.output * (1.0 - step.tanh_c**2)  # h' depends on c' too
        d_gates = np.concatenate([
            total_dc * step.c * step.forget * (1.0 - step.forget),
            total_dc * step.candidate * step.input * (1.0 - step.input),
            total_dc * step.input * (1.0 - step.candidate**2),
            dh * step.tanh_c * step.output * (1.0 - step.output),
        ], axis=-1)
        dz = d_gates @ self.W.T
        return dz[..., : self.n_in], dz[..., self.n_in :], total_dc * step.forget, d_gates

    def add_weight_gradients(self, step: CellStep, d_gates: np.ndarray) -> None:
        self.dW += step.z.reshape(-1, self.n_in + self.n_hidden).T @ d_gates.reshape(-1, 4 * self.n_hidden)
        self.db += d_gates.reshape(-1, 4 * self.n_hidden).sum(axis=0)

    def zero_gradients(self) -> None:
        self.dW[...] = 0.0
        self.db[...] = 0.0

    def forward(self, x: np.ndarray, h: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        new_h, new_c, self.step = self.run_step(x, h, c)
        return new_h, new_c

    def backward(self, dh: np.ndarray, dc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gradients for the one step of the last forward. Returns (dx, dh, dc)."""
        self.zero_gradients()
        dx, dh_before, dc_before, d_gates = self.back_step(self.step, dh, dc)
        self.add_weight_gradients(self.step, d_gates)
        return dx, dh_before, dc_before

    def params(self) -> ParamPairs:
        return [(self.W, self.dW), (self.b, self.db)]


class LSTM(Layer):
    """(batch, time, n_in) -> (batch, time, n_hidden). Backward is full backpropagation through time.

    Every step shares one cell, so its weight gradients add up across the steps.
    """

    def __init__(self, n_in: int, n_hidden: int, rng: np.random.Generator) -> None:
        self.cell = LSTMCell(n_in, n_hidden, rng)
        self.steps: list[CellStep] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        n_batch, n_time, _ = x.shape
        h = np.zeros((n_batch, self.cell.n_hidden))
        c = np.zeros((n_batch, self.cell.n_hidden))
        outputs = np.empty((n_batch, n_time, self.cell.n_hidden))
        self.steps = []
        for t in range(n_time):
            h, c, step = self.cell.run_step(x[:, t], h, c)
            outputs[:, t] = h
            self.steps.append(step)
        return outputs

    def backward(self, dout: np.ndarray) -> np.ndarray:
        n_batch, n_time, n_hidden = dout.shape
        self.cell.zero_gradients()
        dx = np.empty((n_batch, n_time, self.cell.n_in))
        dh = np.zeros((n_batch, n_hidden))  # the gradient arriving from later steps
        dc = np.zeros((n_batch, n_hidden))
        for t in range(n_time - 1, -1, -1):
            dx[:, t], dh, dc, d_gates = self.cell.back_step(self.steps[t], dout[:, t] + dh, dc)
            self.cell.add_weight_gradients(self.steps[t], d_gates)
        return dx

    def params(self) -> ParamPairs:
        return self.cell.params()


def build_lstm(n_features: int, seed: int) -> Sequential:
    rng = np.random.default_rng(seed)
    return Sequential([LSTM(n_features, N_HIDDEN, rng), LastDayPool(), Dropout(DROPOUT, rng), Dense(N_HIDDEN, 1, rng)])


def whole_window(windows: Windows) -> Windows:
    """The LSTM reads the same 22-day windows as the Transformer."""
    return windows


def run_lstm(paths: RunPaths) -> dict:
    return train_neural(paths, "lstm", build_lstm, whole_window, RECIPE)


if __name__ == "__main__":
    run_lstm(REAL_PATHS)
