"""Common fit and scoring interface for the existing array-based models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral

import numpy as np


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
