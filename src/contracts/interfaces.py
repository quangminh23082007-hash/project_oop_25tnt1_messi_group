"""Public boundaries from Contracts v1; no dependency on a Data implementation."""
from abc import ABC, abstractmethod
from numbers import Integral
from typing import Iterable, Optional, Sequence, Set, Protocol
import numpy as np
from .dto import GameItem, Interaction, InteractionRecord, ReviewRecord, ScoredItem
from .exceptions import DataValidationError, InvalidParameterError, ModelNotFittedError


class IdMapper(Protocol):
    size: int
    def contains_raw(self, raw_id: str) -> bool: ...
    def to_idx(self, raw_id: str) -> int: ...
    def to_raw(self, index: int) -> str: ...


class InteractionDataset(Protocol):
    split_name: str
    interactions: Sequence[Interaction]
    num_interactions: int
    num_users: int
    num_items: int
    user_mapper: IdMapper
    item_mapper: IdMapper
    mapping_fingerprint: str
    fingerprint: str
    review_fingerprint: str
    def get_game_by_raw_id(self, raw_game_id: str) -> GameItem: ...
    def get_user_seen_items(self, user_id: int) -> frozenset[int]: ...
    def get_user_positive_items(self, user_id: int) -> frozenset[int]: ...
    def iter_reviews(self) -> Iterable[ReviewRecord]: ...


class BaseDataLoader(ABC):
    @abstractmethod
    def load_interactions(self, source_path: str) -> list[InteractionRecord]: ...

    @abstractmethod
    def load_games(self, source_path: str) -> dict[str, GameItem]: ...


def checked_indices(item_ids, num_items):
    try:
        values = np.asarray(item_ids)
        # Preserve type information for mixed Python lists (e.g. [1, True]).
        original = values if isinstance(item_ids, np.ndarray) else np.asarray(item_ids, dtype=object)
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError("item_ids must be a one-dimensional integer sequence") from exc
    if values.ndim != 1:
        raise InvalidParameterError("item_ids must be a one-dimensional integer sequence")
    if not values.size:
        return np.empty(0, dtype=np.int64)
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in original)
            or (values < 0).any() or (values >= num_items).any()):
        raise InvalidParameterError("item_ids must contain valid internal item indices")
    return values.astype(np.int64)


def checked_scores(scores, count):
    try:
        values = np.asarray(scores)
        if values.dtype.kind not in "fiu" or values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError("invalid scores")
    except (TypeError, ValueError) as exc:
        raise DataValidationError("Scores must be a finite real vector matching candidate order/length") from exc
    return values.astype(float, copy=True)


class BaseRecommender(ABC):
    def __init__(self, name):
        self._name = name
        self._is_fitted = False
        self._num_users = self._num_items = 0
        self._mapping_fingerprint = self._train_fingerprint = None

    @property
    def name(self): return self._name

    @property
    def is_fitted(self): return self._is_fitted

    @property
    def num_users(self): return self._num_users

    @property
    def num_items(self): return self._num_items

    @property
    def mapping_fingerprint(self): return self._mapping_fingerprint

    @property
    def train_fingerprint(self): return self._train_fingerprint

    def _check_fitted(self):
        if not self.is_fitted:
            raise ModelNotFittedError(f"Model {self.name!r} is not fitted")

    def _check_user(self, user_id):
        if isinstance(user_id, bool) or not isinstance(user_id, Integral) or not 0 <= user_id < self.num_users:
            raise InvalidParameterError("user_id is outside the fitted mapping")

    @abstractmethod
    def fit(self, train_dataset, *, validation_score=None, item_features=None, verbose=False): ...

    @abstractmethod
    def score_items(self, user_id, item_ids): ...

    @abstractmethod
    def config(self) -> dict: ...

    def recommend(self, user_id, train_dataset, top_k=10, exclude_seen=True):
        self._check_fitted()
        self._check_user(user_id)
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k <= 0:
            raise InvalidParameterError("top_k must be a positive integer")
        if not isinstance(exclude_seen, bool):
            raise InvalidParameterError("exclude_seen must be boolean")
        if (getattr(train_dataset, "split_name", None) != "train"
                or getattr(train_dataset, "num_users", None) != self.num_users
                or getattr(train_dataset, "num_items", None) != self.num_items
                or getattr(train_dataset, "mapping_fingerprint", None) != self.mapping_fingerprint
                or getattr(train_dataset, "fingerprint", None) != self.train_fingerprint):
            raise DataValidationError("recommend requires the fitted train content and mapping fingerprints")
        candidates = np.arange(self.num_items, dtype=np.int64)
        if exclude_seen:
            seen = checked_indices(list(train_dataset.get_user_seen_items(int(user_id))), self.num_items)
            candidates = np.setdiff1d(candidates, seen, assume_unique=True)
        if not candidates.size:
            return []
        scores = checked_scores(self.score_items(user_id, candidates), len(candidates))
        order = np.lexsort((candidates, -scores))[:top_k]
        return [ScoredItem(int(candidates[i]), float(scores[i])) for i in order]


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, model, train_dataset, evaluation_dataset, *, additional_seen_dataset=None): ...


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


__all__ = ["BaseDataLoader", "BaseRecommender", "BaseMetric", "BaseEvaluator",
           "InteractionDataset", "IdMapper"]