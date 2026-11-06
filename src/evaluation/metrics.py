from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from math import log2
from numbers import Integral

import numpy as np

from src.contracts.exceptions import DataValidationError, InvalidParameterError
from src.contracts.interfaces import BaseMetric


__all__ = [
    "RankingMetric",
    "RecallAtK",
    "NDCGAtK",
]


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise InvalidParameterError(f"{name} must be an integer.")
    if value < minimum:
        raise InvalidParameterError(f"{name} must be at least {minimum}.")
    return int(value)


def _item_id(value: object, num_total_items: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DataValidationError("Item IDs must be integers.")
    item_id = int(value)
    if item_id < 0:
        raise DataValidationError("Item IDs must be nonnegative.")
    if num_total_items is not None and item_id >= num_total_items:
        raise DataValidationError("Item ID is outside the catalog range.")
    return item_id


class RankingMetric(BaseMetric):
    def __init__(self, k: int = 10) -> None:
        super().__init__(k=_integer(k, "k", 1))

    @property
    def k(self) -> int:
        return self._k

    def compute(
        self,
        ground_truth: Iterable[int],
        ranked_items: Sequence[int] | np.ndarray,
        num_total_items: int | None = None,
    ) -> float:
        if num_total_items is not None:
            num_total_items = _integer(num_total_items, "num_total_items")

        if isinstance(ground_truth, (str, bytes, bytearray, Mapping)):
            raise TypeError("ground_truth must be an iterable of integer item IDs.")

        try:
            ground_truth_iterator = iter(ground_truth)
        except TypeError as error:
            raise TypeError(
                "ground_truth must be an iterable of integer item IDs."
            ) from error

        if isinstance(ranked_items, np.ndarray):
            if ranked_items.ndim != 1:
                raise DataValidationError("ranked_items must be one-dimensional.")
        elif (
            isinstance(ranked_items, (str, bytes, bytearray, Mapping))
            or not isinstance(ranked_items, Sequence)
        ):
            raise TypeError(
                "ranked_items must be an ordered sequence of integer item IDs."
            )

        relevant_items = {
            _item_id(item_id, num_total_items)
            for item_id in ground_truth_iterator
        }
        ranking = [
            _item_id(item_id, num_total_items)
            for item_id in ranked_items
        ]
        top_items = ranking[:self.k]

        if len(top_items) != len(set(top_items)):
            raise DataValidationError(
                "The top-K ranking contains duplicate item IDs."
            )

        if not relevant_items or not top_items:
            return 0.0

        return self._compute_score(relevant_items, top_items)

    @abstractmethod
    def _compute_score(
        self,
        ground_truth: set[int],
        top_items: Sequence[int],
    ) -> float:
        ...


class RecallAtK(RankingMetric):
    @property
    def name(self) -> str:
        return f"Recall@{self.k}"

    def _compute_score(
        self,
        ground_truth: set[int],
        top_items: Sequence[int],
    ) -> float:
        hits = len(ground_truth.intersection(top_items))
        return hits / len(ground_truth)


class NDCGAtK(RankingMetric):
    @property
    def name(self) -> str:
        return f"NDCG@{self.k}"

    def _compute_score(
        self,
        ground_truth: set[int],
        top_items: Sequence[int],
    ) -> float:
        dcg = sum(
            1.0 / log2(rank + 1)
            for rank, item_id in enumerate(top_items, start=1)
            if item_id in ground_truth
        )
        ideal_hits = min(self.k, len(ground_truth))
        idcg = sum(
            1.0 / log2(rank + 1)
            for rank in range(1, ideal_hits + 1)
        )
        return dcg / idcg