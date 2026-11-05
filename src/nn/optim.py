"""AdamW optimiser, gradient clipping and learning-rate warm-up, written by hand."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def clip_grad_norm(params: Sequence[tuple[np.ndarray, np.ndarray]], max_norm: float) -> float:
    """Scale all gradients in place so their joint L2 norm is at most max_norm.

    Returns the norm measured before clipping.
    """
    norm = float(np.sqrt(sum(np.sum(grad**2) for _, grad in params)))
    if norm > max_norm:
        for _, grad in params:
            grad *= max_norm / norm
    return norm


def warmup_lr(base_lr: float, epoch: int, warmup_epochs: int) -> float:
    """Linear warm-up by epoch: epoch 0 uses base_lr / warmup_epochs, reaching base_lr at the end."""
    return base_lr * min(1.0, (epoch + 1) / warmup_epochs)


class AdamW:
    """Adam with decoupled weight decay (Loshchilov and Hutter).

    Works on (value, grad) array pairs and updates each value in place. Decay applies to
    matrices only (ndim >= 2), so biases and other vectors are not shrunk toward zero.
    """

    def __init__(
        self,
        params: Sequence[tuple[np.ndarray, np.ndarray]],
        lr: float,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.first_moment = [np.zeros_like(value) for value, _ in self.params]
        self.second_moment = [np.zeros_like(value) for value, _ in self.params]
        self.steps = 0

    def step(self) -> None:
        self.steps += 1
        correct_first = 1.0 - self.beta1**self.steps
        correct_second = 1.0 - self.beta2**self.steps
        for (value, grad), m1, m2 in zip(self.params, self.first_moment, self.second_moment):
            m1 *= self.beta1
            m1 += (1.0 - self.beta1) * grad
            m2 *= self.beta2
            m2 += (1.0 - self.beta2) * grad**2
            update = (m1 / correct_first) / (np.sqrt(m2 / correct_second) + self.eps)
            if value.ndim >= 2 and self.weight_decay:
                # Decoupled decay: applied to the weight directly, not added to the gradient.
                update = update + self.weight_decay * value
            value -= self.lr * update
