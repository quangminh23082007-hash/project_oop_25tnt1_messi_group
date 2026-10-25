"""Models only: no raw-file parsing, recommendation filtering or metrics."""
from __future__ import annotations

from abc import ABC, abstractmethod
import math
from numbers import Integral
from typing import Callable

import numpy as np


def positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_pairs(values, n_users: int, n_items: int, allow_empty=False):
    pairs = np.asarray(values)
    if pairs.size == 0 and allow_empty:
        return np.empty((0, 2), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
        raise ValueError("Interactions must be a nonempty (N, 2) array")
    if not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("Mapped user/item IDs must be integers")
    if (pairs < 0).any() or (pairs[:, 0] >= n_users).any() or (pairs[:, 1] >= n_items).any():
        raise ValueError("Mapped ID is outside the declared user/item range")
    return np.unique(pairs.astype(np.int64), axis=0)


def pair_sets(pairs) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for u, i in pairs:
        result.setdefault(int(u), set()).add(int(i))
    return result


class BaseRecommender(ABC):
    def __init__(self):
        self._is_fitted = False

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit() or load a trained model first")

    def _check_user(self, user_idx):
        self._check_fitted()
        if isinstance(user_idx, bool) or not isinstance(user_idx, Integral):
            raise ValueError("user_idx must be an integer")
        if not 0 <= user_idx < self.n_users:
            raise ValueError("Unknown user index; use the service fallback")

    def score(self, user_idx: int, item_idx: int) -> float:
        self._check_pair(user_idx, item_idx)
        return float(self.score_all(user_idx)[item_idx])

    def _check_pair(self, user_idx, item_idx):
        self._check_user(user_idx)
        if isinstance(item_idx, bool) or not isinstance(item_idx, Integral):
            raise ValueError("item_idx must be an integer")
        if not 0 <= item_idx < self.n_items:
            raise ValueError("Unknown item index")

    @abstractmethod
    def fit(self, interactions, n_users: int, n_items: int, **kwargs):
        """Implementations learn only from the supplied training data."""
        raise NotImplementedError

    @abstractmethod
    def score_all(self, user_idx: int) -> np.ndarray:
        """Return one score per TRAIN catalog item."""
        raise NotImplementedError


class PopularityRecommender(BaseRecommender):
    def fit(self, interactions, n_users: int, n_items: int, **kwargs):
        self._is_fitted = False
        self.n_users = positive_int(n_users, "n_users")
        self.n_items = positive_int(n_items, "n_items")
        pairs = validate_pairs(interactions, n_users, n_items)
        # Each distinct (user, item) contributes once; refit replaces old counts.
        self.counts = np.bincount(pairs[:, 1], minlength=n_items).astype(float)
        self._is_fitted = True
        return self

    def score_all(self, user_idx):
        self._check_user(user_idx)
        return self.counts.copy()

    def score(self, user_idx, item_idx):
        self._check_pair(user_idx, item_idx)
        return float(self.counts[item_idx])


def bpr_loss_and_gradients(p, qi, qj, reg: float):
    """One sampled triple, L = softplus(-delta) + reg/2 * sum(norm^2).

    All gradients use the SAME pre-update vectors. Regularization here is
    per sampled triple, so frequent factors receive more shrinkage updates.
    """
    delta = float(p @ (qi - qj))
    g = math.exp(-float(np.logaddexp(0.0, delta)))  # sigmoid(-delta)
    loss = float(np.logaddexp(0.0, -delta))
    loss += 0.5 * reg * float(p @ p + qi @ qi + qj @ qj)
    return loss, (-g * (qi - qj) + reg * p,
                  -g * p + reg * qi,
                  g * p + reg * qj)


class BPRMatrixFactorization(BaseRecommender):
    parameter_names = ("P", "Q")
    def __init__(self, factors=32, learning_rate=0.03, reg=0.01,
                 epochs=30, seed=42, patience=5):
        super().__init__()
        self.factors = positive_int(factors, "factors")
        self.epochs = positive_int(epochs, "epochs")
        self.patience = positive_int(patience, "patience")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and > 0")
        if not math.isfinite(reg) or reg < 0:
            raise ValueError("reg must be finite and >= 0")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.learning_rate, self.reg, self.seed = learning_rate, reg, int(seed)
        self.history: list[dict] = []

    def config(self):
        return {name: getattr(self, name) for name in
                ("factors", "learning_rate", "reg", "epochs", "seed", "patience")}

    def _initialize_parameters(self, rng):
        self.P = rng.normal(0.0, 0.1, (self.n_users, self.factors))
        self.Q = rng.normal(0.0, 0.1, (self.n_items, self.factors))

    def _sgd_step(self, u, i, j):
        p, qi, qj = self.P[u].copy(), self.Q[i].copy(), self.Q[j].copy()
        loss, (dp, di, dj) = bpr_loss_and_gradients(p, qi, qj, self.reg)
        self.P[u] -= self.learning_rate * dp
        self.Q[i] -= self.learning_rate * di
        self.Q[j] -= self.learning_rate * dj
        return loss

    def _invalidate_scores(self):
        """Hook for models with cached projected item factors."""

    def fit(self, interactions, n_users, n_items, *, observed=None,
            validation_score: Callable | None = None, verbose=True):
        self._is_fitted = False
        self.n_users = positive_int(n_users, "n_users")
        self.n_items = positive_int(n_items, "n_items")
        pairs = validate_pairs(interactions, n_users, n_items)
        observed = pairs if observed is None else validate_pairs(observed, n_users, n_items)
        blocked = pair_sets(observed)
        for u, i in pairs:
            blocked.setdefault(int(u), set()).add(int(i))
        # No rejection-sampling infinite loop when a user has seen every item.
        eligible = np.array([len(blocked[int(u)]) < n_items for u in pairs[:, 0]])
        pairs = pairs[eligible]
        if not len(pairs):
            raise ValueError("No valid BPR triples: all users have seen every catalog item")
        self.skipped_pairs = int((~eligible).sum())
        rng = np.random.default_rng(self.seed)
        self._initialize_parameters(rng)
        # Dense-history users get a finite complement; sparse users use rejection.
        dense = {u: np.array([i for i in range(n_items) if i not in seen])
                 for u, seen in blocked.items() if n_items / 2 < len(seen) < n_items}
        self.history = []
        self.best_epoch = 0
        best_value, stale, best_weights = -math.inf, 0, None
        self._is_fitted = True
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            # One negative draw per distinct positive pair per epoch.
            for index in rng.permutation(len(pairs)):
                u, i = map(int, pairs[index])
                if u in dense:
                    j = int(rng.choice(dense[u]))
                else:
                    j = int(rng.integers(n_items))
                    while j in blocked[u]:
                        j = int(rng.integers(n_items))
                total_loss += self._sgd_step(u, i, j)
            self._invalidate_scores()
            if not all(np.isfinite(getattr(self, name)).all() for name in self.parameter_names):
                self._is_fitted = False
                raise FloatingPointError("Training diverged; lower learning_rate")
            row = {"epoch": epoch, "sampled_loss": total_loss / len(pairs)}
            if validation_score is not None:
                value = float(validation_score(self))
                if not math.isfinite(value):
                    raise ValueError("Validation metric is not finite")
                row["validation_ndcg"] = value
                if value > best_value + 1e-12:
                    best_value, stale = value, 0
                    self.best_epoch = epoch
                    best_weights = {name: getattr(self, name).copy() for name in self.parameter_names}
                else:
                    stale += 1
            else:
                self.best_epoch = epoch
            self.history.append(row)
            if verbose:
                print(row, flush=True)
            if validation_score is not None and stale >= self.patience:
                break
        if best_weights is not None:
            for name, value in best_weights.items():
                setattr(self, name, value)
        self._invalidate_scores()
        return self

    def score_all(self, user_idx):
        self._check_user(user_idx)
        return self.Q @ self.P[user_idx]

    def score(self, user_idx, item_idx):
        self._check_pair(user_idx, item_idx)
        return float(self.P[user_idx] @ self.Q[item_idx])


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
