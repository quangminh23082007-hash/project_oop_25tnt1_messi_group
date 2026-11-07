"""Text-aware BPR using frozen item vectors and a learned text projection."""
from __future__ import annotations

import math

import numpy as np

if __package__:
    from .matrix_factorization import BPRMatrixFactorization
else:
    from matrix_factorization import BPRMatrixFactorization


def text_bpr_loss_and_gradients(p, qi, qj, W, xi, xj, reg, text_reg):
    """L = softplus(-p@(qi-qj+W@(xi-xj))) + per-triple L2 penalties.

    All gradients use pre-update parameters. X stays frozen.
    """
    dx = xi - xj
    difference = qi - qj + W @ dx
    delta = float(p @ difference)
    g = math.exp(-float(np.logaddexp(0.0, delta)))
    loss = float(np.logaddexp(0.0, -delta))
    loss += 0.5 * reg * float(p @ p + qi @ qi + qj @ qj)
    loss += 0.5 * text_reg * float(np.sum(W * W))
    return loss, (-g * difference + reg * p, -g * p + reg * qi,
                  g * p + reg * qj, -g * np.outer(p, dx) + text_reg * W)


class TextBPRMatrixFactorization(BPRMatrixFactorization):
    """One score: s(u,i) = p_u @ (q_i + W @ x_i)."""
    parameter_names = ("P", "Q", "W")

    def __init__(self, factors=32, learning_rate=0.03, reg=0.01,
                 epochs=30, seed=42, patience=5, text_reg=0.001):
        super().__init__(factors, learning_rate, reg, epochs, seed, patience)
        if not math.isfinite(text_reg) or text_reg < 0:
            raise ValueError("text_reg must be finite and >= 0")
        self.text_reg = text_reg
        self._effective_items = None

    def config(self):
        return {**super().config(), "text_reg": self.text_reg}

    def fit(self, interactions, n_users, n_items, *, item_features,
            observed=None, validation_score=None, verbose=True):
        self._is_fitted = False
        X = np.asarray(item_features, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != n_items or X.shape[1] == 0:
            raise ValueError("item_features must have shape (n_items, embedding_dimension)")
        if not np.isfinite(X).all():
            raise ValueError("item_features must be finite")
        norms = np.linalg.norm(X, axis=1)
        if not np.all(np.isclose(norms, 0, atol=1e-6) | np.isclose(norms, 1, atol=1e-5)):
            raise ValueError("Item vectors must have unit norm, or be zero for missing text")
        self.X = X.copy()
        self._invalidate_scores()
        return super().fit(interactions, n_users, n_items, observed=observed,
                           validation_score=validation_score, verbose=verbose)

    def _initialize_parameters(self, rng):
        super()._initialize_parameters(rng)
        # Initial scores match BPR; dW can still be nonzero on the first step.
        self.W = np.zeros((self.factors, self.X.shape[1]), dtype=np.float64)

    def _sgd_step(self, u, i, j):
        p, qi, qj = self.P[u].copy(), self.Q[i].copy(), self.Q[j].copy()
        loss, (dp, di, dj, dW) = text_bpr_loss_and_gradients(
            p, qi, qj, self.W, self.X[i], self.X[j], self.reg, self.text_reg)
        self.P[u] -= self.learning_rate * dp
        self.Q[i] -= self.learning_rate * di
        self.Q[j] -= self.learning_rate * dj
        self.W -= self.learning_rate * dW
        return loss

    def _invalidate_scores(self):
        self._effective_items = None

    def score_all(self, user_idx):
        self._check_user(user_idx)
        if self._effective_items is None:
            self._effective_items = self.Q + self.X @ self.W.T
        return self._effective_items @ self.P[user_idx]

    def score(self, user_idx, item_idx):
        self._check_pair(user_idx, item_idx)
        return float(self.P[user_idx] @ (self.Q[item_idx] + self.W @ self.X[item_idx]))
