from abc import ABC, abstractmethod
from math import log2
from numbers import Integral
from typing import Iterable, Optional, Sequence, Set


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _item_id(value: object, num_total_items: Optional[int]) -> int:
    item_id = _integer(value, "item ID")
    if num_total_items is not None and item_id >= num_total_items:
        raise ValueError("Item ID is outside the catalog range.")
    return item_id


class RankingMetric(ABC):
    def __init__(self, k: int = 10) -> None:
        self._k = _integer(k, "k", 1)

    @property
    def k(self) -> int:
        return self._k

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def compute(
        self,
        ground_truth: Iterable[int],
        ranked_items: Sequence[int],
        num_total_items: Optional[int] = None,
    ) -> float:
        if num_total_items is not None:
            num_total_items = _integer(num_total_items, "num_total_items")
        if isinstance(ground_truth, (str, bytes, bytearray)):
            raise TypeError("ground_truth must contain integer item IDs.")
        if isinstance(ranked_items, (str, bytes, bytearray)) or not isinstance(ranked_items, Sequence):
            raise TypeError("ranked_items must be an ordered sequence of integer item IDs.")
        relevant_items = {_item_id(item, num_total_items) for item in ground_truth}
        ranking = [_item_id(item, num_total_items) for item in ranked_items]
        top_items = ranking[:self.k]
        if len(top_items) != len(set(top_items)):
            raise ValueError("The top-K ranking contains duplicate item IDs.")
        if not relevant_items:
            return 0.0
        return self._compute_score(relevant_items, top_items)

    @abstractmethod
    def _compute_score(self, ground_truth: Set[int], top_items: Sequence[int]) -> float:
        raise NotImplementedError


class RecallAtK(RankingMetric):
    @property
    def name(self) -> str:
        return f"Recall@{self.k}"

    def _compute_score(self, ground_truth: Set[int], top_items: Sequence[int]) -> float:
        hits = len(ground_truth.intersection(top_items))
        return hits / len(ground_truth)


class NDCGAtK(RankingMetric):
    @property
    def name(self) -> str:
        return f"NDCG@{self.k}"

    def _compute_score(self, ground_truth: Set[int], top_items: Sequence[int]) -> float:
        dcg = sum(
            1.0 / log2(rank + 1)
            for rank, item_id in enumerate(top_items, start=1)
            if item_id in ground_truth
        )
        ideal_hits = min(self.k, len(ground_truth))
        idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
        return dcg / idcg
