"""Neural-network layers with hand-written gradients.

Every layer follows one contract:
    forward(x)     computes the output and caches what backward needs.
    backward(dout) takes dL/d(output), stores dL/d(parameters) in place, returns dL/d(input).
    params()       returns (value, grad) array pairs. The optimiser keeps these references,
                   so gradients must always be written in place.
"""

from __future__ import annotations

import numpy as np

from src.nn.functional import stable_softmax

ParamPairs = list[tuple[np.ndarray, np.ndarray]]


class Layer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> ParamPairs:
        return []


class Dense(Layer):
    """y = x W + b, applied to the last axis of an input of any rank."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, bias: bool = True) -> None:
        self.n_in, self.n_out = n_in, n_out
        # Xavier normal initialisation keeps the output variance close to the input's.
        self.W = rng.normal(0.0, np.sqrt(2.0 / (n_in + n_out)), (n_in, n_out))
        self.dW = np.zeros_like(self.W)
        self.b = np.zeros(n_out) if bias else None
        self.db = np.zeros(n_out) if bias else None
        self.x = np.empty(0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        out = x @ self.W
        return out if self.b is None else out + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Flatten the leading axes: every position shares the same W and b, so their
        # contributions to the gradient add up.
        inputs = self.x.reshape(-1, self.n_in)
        outputs = dout.reshape(-1, self.n_out)
        self.dW[...] = inputs.T @ outputs
        if self.db is not None:
            self.db[...] = outputs.sum(axis=0)
        return dout @ self.W.T

    def params(self) -> ParamPairs:
        pairs = [(self.W, self.dW)]
        if self.b is not None:
            pairs.append((self.b, self.db))
        return pairs


class GELU(Layer):
    """Tanh approximation: GELU(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))."""

    ROOT_TWO_OVER_PI = np.sqrt(2.0 / np.pi)
    CUBIC = 0.044715

    def __init__(self) -> None:
        self.x = np.empty(0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return 0.5 * x * (1.0 + np.tanh(self.ROOT_TWO_OVER_PI * (x + self.CUBIC * x**3)))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.x
        tanh_u = np.tanh(self.ROOT_TWO_OVER_PI * (x + self.CUBIC * x**3))
        du_dx = self.ROOT_TWO_OVER_PI * (1.0 + 3.0 * self.CUBIC * x**2)
        # Product rule, using d tanh(u)/du = 1 - tanh(u)^2.
        return dout * (0.5 * (1.0 + tanh_u) + 0.5 * x * (1.0 - tanh_u**2) * du_dx)


class ReLU(Layer):
    """max(x, 0). The gradient at exactly 0 is taken as 0."""

    def __init__(self) -> None:
        self.active = np.empty(0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.active = x > 0.0
        return x * self.active

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.active


class Dropout(Layer):
    """Zero each value with probability `rate` while training, scaling the rest by 1/(1-rate)."""

    def __init__(self, rate: float, rng: np.random.Generator) -> None:
        self.rate = rate
        self.rng = rng
        self.training = True
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.rate == 0.0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.rate) / (1.0 - self.rate)
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout if self.mask is None else dout * self.mask


class LayerNorm(Layer):
    """Normalise the last axis of each position, then apply a learned scale and shift.

    x_hat = (x - mean) / sigma, sigma = sqrt(var + eps), y = gamma x_hat + beta.
    Backward, for dx_hat = dy gamma:
    dx = (dx_hat - mean(dx_hat) - x_hat mean(dx_hat x_hat)) / sigma.
    """

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        self.eps = eps
        self.gamma = np.ones(d)
        self.beta = np.zeros(d)
        self.dgamma = np.zeros(d)
        self.dbeta = np.zeros(d)
        self.x_hat = np.empty(0)
        self.sigma = np.empty(0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        self.sigma = np.sqrt(x.var(axis=-1, keepdims=True) + self.eps)
        self.x_hat = (x - mean) / self.sigma
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        leading_axes = tuple(range(dout.ndim - 1))
        self.dgamma[...] = (dout * self.x_hat).sum(axis=leading_axes)
        self.dbeta[...] = dout.sum(axis=leading_axes)
        dx_hat = dout * self.gamma
        mean_dx_hat = dx_hat.mean(axis=-1, keepdims=True)
        mean_product = (dx_hat * self.x_hat).mean(axis=-1, keepdims=True)
        return (dx_hat - mean_dx_hat - self.x_hat * mean_product) / self.sigma

    def params(self) -> ParamPairs:
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


class Softmax(Layer):
    """Row-wise softmax over the last axis. Backward: dx_i = p_i (dp_i - sum_j p_j dp_j)."""

    def __init__(self) -> None:
        self.probabilities = np.empty(0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.probabilities = stable_softmax(x, axis=-1)
        return self.probabilities

    def backward(self, dout: np.ndarray) -> np.ndarray:
        p = self.probabilities
        return p * (dout - (p * dout).sum(axis=-1, keepdims=True))


class PositionalEmbedding(Layer):
    """Add a learned vector for each position to inputs shaped (batch, time, d)."""

    def __init__(self, n_positions: int, d: int, rng: np.random.Generator, init_std: float = 0.02) -> None:
        self.P = rng.normal(0.0, init_std, (n_positions, d))
        self.dP = np.zeros_like(self.P)
        self.n_time = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.n_time = x.shape[1]
        return x + self.P[: self.n_time]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dP[...] = 0.0
        self.dP[: self.n_time] = dout.sum(axis=0)  # the batch shares one table
        return dout

    def params(self) -> ParamPairs:
        return [(self.P, self.dP)]


class LastDayPool(Layer):
    """Keep only the final time step: (batch, time, d) -> (batch, d).

    The final step still sees every earlier day through attention, so history is not ignored.
    """

    def __init__(self) -> None:
        self.input_shape: tuple[int, ...] = ()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_shape = x.shape
        return x[:, -1, :]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        gradient = np.zeros(self.input_shape)
        gradient[:, -1, :] = dout
        return gradient


class MeanPool(Layer):
    """Average over the time axis: (batch, time, d) -> (batch, d)."""

    def __init__(self) -> None:
        self.n_time = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.n_time = x.shape[1]
        return x.mean(axis=1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return np.repeat(dout[:, None, :], self.n_time, axis=1) / self.n_time


class Sequential:
    """A model that is a plain stack of layers, with what the training loop needs from a model."""

    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def params(self) -> ParamPairs:
        return [pair for layer in self.layers for pair in layer.params()]

    def n_parameters(self) -> int:
        return sum(value.size for value, _ in self.params())

    def set_training(self, training: bool) -> None:
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = training

    def get_weights(self) -> list[np.ndarray]:
        return [value.copy() for value, _ in self.params()]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """Copy values in place, so the references held by an optimiser stay valid."""
        for (value, _), saved in zip(self.params(), weights):
            value[...] = saved
