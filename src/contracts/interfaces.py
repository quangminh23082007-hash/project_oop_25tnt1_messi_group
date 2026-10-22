"""
Abstract Base Classes defining strict boundaries between the 4 team members.

Design Justifications:
- Abstraction: Defines clean, minimal, non-leaky contracts for DataLoader, Recommender, and Metric.
- Template Method Pattern (BaseRecommender.fit): High-level invariant checks (dataset validation,
  dimension caching, fitted flag transition) are controlled by the abstract base class, while
  algorithmic logic is deferred to polymorphic `_fit_internal()` subclasses.
- Open-Closed Principle: New algorithms or evaluation metrics can be added without modifying
  existing interfaces or consumer services.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Set, TYPE_CHECKING
import numpy as np

from src.contracts.dto import InteractionRecord, ScoredItem
from src.contracts.exceptions import (
    EmptyDatasetError,
    InvalidParameterError,
    ModelNotFittedError,
)

if TYPE_CHECKING:
    from src.data.dataset import InteractionDataset
    from src.data.entities import GameItem


class BaseDataLoader(ABC):
    """[Member 1 Contract] Abstract interface for loading game metadata and interactions."""

    @abstractmethod
    def load_interactions(self, source_path: str) -> List[InteractionRecord]:
        """Load and validate user-game interactions from a file."""

    @abstractmethod
    def load_games(self, source_path: str) -> Dict[str, "GameItem"]:
        """Load game catalog metadata keyed by raw game_id."""


class BaseRecommender(ABC):
    """[Member 2 Contract] Abstract base class for all recommendation models."""

    def __init__(self, name: str) -> None:
        self._name: str = name
        self._is_fitted: bool = False
        self._num_users: int = 0
        self._num_items: int = 0

    @property
    def name(self) -> str:
        """Name of the recommendation algorithm."""
        return self._name

    @property
    def is_fitted(self) -> bool:
        """True if the model has undergone successful training."""
        return self._is_fitted

    def _check_fitted(self) -> None:
        """Guard clause raising ModelNotFittedError if model has not been trained."""
        if not self._is_fitted:
            raise ModelNotFittedError(
                f"Model '{self._name}' has not been fitted yet. Call fit(dataset) first."
            )

    def fit(self, dataset: "InteractionDataset") -> "BaseRecommender":
        """Template method: validates dataset, delegates to _fit_internal, marks fitted."""
        if dataset is None or dataset.num_interactions == 0:
            raise EmptyDatasetError("Cannot train model on an empty InteractionDataset.")

        self._num_users = dataset.num_users
        self._num_items = dataset.num_items
        self._fit_internal(dataset)
        self._is_fitted = True
        return self

    @abstractmethod
    def _fit_internal(self, dataset: "InteractionDataset") -> None:
        """Concrete subclasses implement their core training algorithm here."""

    @abstractmethod
    def score_items(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Predict relevance scores for a given internal user_idx across item_indices."""

    def recommend(
        self,
        user_idx: int,
        dataset: "InteractionDataset",
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> List[ScoredItem]:
        """Generate top-K ScoredItems for an internal user_idx."""
        self._check_fitted()
        if top_k <= 0:
            raise InvalidParameterError(f"top_k must be a positive integer, got {top_k}.")
        if not (0 <= user_idx < self._num_users):
            raise InvalidParameterError(
                f"user_idx {user_idx} out of range [0, {self._num_users})."
            )

        all_items = np.arange(self._num_items, dtype=np.int64)
        if exclude_seen:
            seen_items = dataset.get_user_seen_items(user_idx)
            if seen_items:
                mask = np.ones(self._num_items, dtype=bool)
                mask[list(seen_items)] = False
                candidate_items = all_items[mask]
            else:
                candidate_items = all_items
        else:
            candidate_items = all_items

        if candidate_items.size == 0:
            return []

        scores = self.score_items(user_idx, candidate_items)
        k = min(top_k, candidate_items.size)
        # Sort descending by score (with deterministic tie-breaking by item_idx)
        ranked_order = np.lexsort((candidate_items, -scores))[:k]
        return [
            ScoredItem(item_idx=int(candidate_items[i]), score=float(scores[i]))
            for i in ranked_order
        ]


class BaseMetric(ABC):
    """[Member 3 Contract] Abstract base class for ranking evaluation metrics."""

    def __init__(self, k: Optional[int] = None) -> None:
        if k is not None and k <= 0:
            raise InvalidParameterError(f"Metric cutoff k must be > 0, got {k}.")
        self._k = k

    @property
    def k(self) -> Optional[int]:
        """Rank cutoff threshold k."""
        return self._k

    @property
    @abstractmethod
    def name(self) -> str:
        """Return formatted metric name (e.g., 'Recall@10', 'NDCG@10', 'AUC')."""

    @abstractmethod
    def compute(
        self,
        ground_truth: Set[int],
        ranked_items: Sequence[int],
        num_total_items: int,
    ) -> float:
        """Compute metric score in [0.0, 1.0] for a single user."""


__all__ = [
    "BaseDataLoader",
    "BaseRecommender",
    "BaseMetric",
]
