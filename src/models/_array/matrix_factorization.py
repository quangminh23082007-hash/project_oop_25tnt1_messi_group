"""BPR matrix factorization trained from scratch with stochastic gradient descent."""
from __future__ import annotations

import math
from numbers import Integral
from typing import Callable

import numpy as np

if __package__:
    from .base import BaseRecommender
    from ._utils import pair_sets, positive_int, validate_pairs
else:
    from base import BaseRecommender
    from _utils import pair_sets, positive_int, validate_pairs


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


# Name used in the team layout. An alias preserves the existing class identity
# and checkpoint model_type while exposing the same BPR/SGD implementation.
MatrixFactorizationRecommender = BPRMatrixFactorization
