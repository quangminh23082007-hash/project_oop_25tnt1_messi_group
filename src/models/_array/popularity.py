"""Popularity baseline based on distinct training users per item."""
from __future__ import annotations

import numpy as np

if __package__:
    from .base import BaseRecommender
    from ._utils import positive_int, validate_pairs
else:
    from base import BaseRecommender
    from _utils import positive_int, validate_pairs


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
