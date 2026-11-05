"""Validation and mapped interaction helpers shared by the models."""
from __future__ import annotations

from numbers import Integral

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
