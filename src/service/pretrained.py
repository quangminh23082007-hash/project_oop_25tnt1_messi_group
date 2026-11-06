"""Pretrained service facade and runtime backend for production / UI inference."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set
import numpy as np

from src.contracts.dto import GameItem, Interaction, RecommendationResult, ScoredItem
from src.contracts.exceptions import (
    ColdStartError,
    ConfigurationError,
    DataValidationError,
    InvalidParameterError,
    ModelNotFittedError,
    RecSysError,
    UnknownItemError,
    UnknownUserError,
)
from src.contracts.interfaces import (
    BaseRecommender,
    IdMapper,
    checked_indices,
    checked_scores,
)
from src.service.filters import FilterCriteria, game_passes_filters, parse_filter_criteria
from src.service.formatters import format_recommendations_table
from src.service.mappers import to_recommendation_result
from src.service.validators import validate_fallback_model, validate_id, validate_top_k


class PretrainedDatasetProxy:
    """Read-only InteractionDataset proxy backed by pre-computed metadata bundles."""

    def __init__(
        self,
        user_mapper: IdMapper,
        item_mapper: IdMapper,
        catalog: Dict[str, GameItem],
        user_seen_items: Dict[int, Set[int]],
        user_positive_items: Optional[Dict[int, Set[int]]] = None,
        mapping_fingerprint: str = "mfp-pretrained",
        fingerprint: str = "fp-pretrained",
        review_fingerprint: str = "rfp-pretrained",
    ) -> None:
        self.split_name = "train"
        self.user_mapper = user_mapper
        self.item_mapper = item_mapper
        self.num_users = user_mapper.size
        self.num_items = item_mapper.size
        self.mapping_fingerprint = mapping_fingerprint
        self.fingerprint = fingerprint
        self.review_fingerprint = review_fingerprint

        self._catalog = dict(catalog)
        self._user_seen = {u: frozenset(items) for u, items in user_seen_items.items()}
        self._user_pos = (
            {u: frozenset(items) for u, items in user_positive_items.items()}
            if user_positive_items is not None
            else self._user_seen
        )

        # Build minimal synthetic interactions sequence to fulfill protocol
        interactions_list: List[Interaction] = []
        for u, seen in self._user_seen.items():
            pos = self._user_pos.get(u, frozenset())
            for i in seen:
                rating = 1.0 if i in pos else 0.0
                interactions_list.append(Interaction(user_id=u, item_id=i, rating=rating, timestamp=0))
        self.interactions: Sequence[Interaction] = tuple(interactions_list)
        self.num_interactions = len(self.interactions)

    def get_game_by_raw_id(self, raw_game_id: str) -> GameItem:
        if raw_game_id not in self._catalog:
            raise UnknownItemError(f"Game '{raw_game_id}' is not in catalog.")
        return self._catalog[raw_game_id]

    def get_user_seen_items(self, user_id: int) -> FrozenSet[int]:
        return self._user_seen.get(user_id, frozenset())

    def get_user_positive_items(self, user_id: int) -> FrozenSet[int]:
        return self._user_pos.get(user_id, frozenset())

    def iter_reviews(self) -> Iterable[Any]:
        return iter(())


class PretrainedServiceFacade:
    """Production and UI facade coordinating inference from pre-trained model bundles."""

    def __init__(
        self,
        model: BaseRecommender,
        user_mapper: IdMapper,
        item_mapper: IdMapper,
        catalog: Dict[str, GameItem],
        user_seen_items: Dict[int, Set[int]],
        user_positive_items: Optional[Dict[int, Set[int]]] = None,
        *,
        fallback_model: Optional[BaseRecommender] = None,
        allow_cold_start_fallback: bool = False,
        mapping_fingerprint: str = "mfp-pretrained",
        train_fingerprint: str = "fp-pretrained",
    ) -> None:
        if not isinstance(model, BaseRecommender):
            raise InvalidParameterError("model must be an instance of BaseRecommender.")
        if not bool(getattr(model, "is_fitted", False)):
            raise ModelNotFittedError(f"Model '{model.name}' must be fitted before initialization.")
        self._model = model
        self._fallback_model = fallback_model
        self._allow_cold_start_fallback = allow_cold_start_fallback
        self._user_mapper = user_mapper
        self._item_mapper = item_mapper
        self._catalog = dict(catalog)
        self._user_seen_items = {u: set(items) for u, items in user_seen_items.items()}
        self._user_positive_items = (
            {u: set(items) for u, items in user_positive_items.items()}
            if user_positive_items is not None
            else {u: set(items) for u, items in user_seen_items.items()}
        )
        self._extra_seen: Dict[int, Set[int]] = defaultdict(set)

        self._dataset_proxy = PretrainedDatasetProxy(
            user_mapper=user_mapper,
            item_mapper=item_mapper,
            catalog=self._catalog,
            user_seen_items=self._user_seen_items,
            user_positive_items=self._user_positive_items,
            mapping_fingerprint=mapping_fingerprint,
            fingerprint=train_fingerprint,
        )

        if fallback_model is not None:
            validate_fallback_model(fallback_model)
            if not bool(getattr(fallback_model, "is_fitted", False)):
                fallback_model.fit(self._dataset_proxy)

    @property
    def model(self) -> BaseRecommender:
        return self._model

    @property
    def model_name(self) -> str:
        return self._model.name

    @property
    def fallback_model(self) -> Optional[BaseRecommender]:
        return self._fallback_model

    @property
    def allow_cold_start_fallback(self) -> bool:
        return self._allow_cold_start_fallback

    @allow_cold_start_fallback.setter
    def allow_cold_start_fallback(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise InvalidParameterError("allow_cold_start_fallback must be boolean.")
        self._allow_cold_start_fallback = value

    @property
    def dataset(self) -> PretrainedDatasetProxy:
        return self._dataset_proxy

    def record_user_interaction(
        self,
        raw_user_id: str,
        raw_game_id: str,
        rating: float = 1.0,
    ) -> None:
        """Record dynamic runtime interaction (extra_seen) from session/UI."""
        raw_user_id = validate_id(raw_user_id, "raw_user_id")
        raw_game_id = validate_id(raw_game_id, "raw_game_id")
        if not self._user_mapper.contains_raw(raw_user_id):
            raise UnknownUserError(f"User '{raw_user_id}' is not in user mapper.")
        if not self._item_mapper.contains_raw(raw_game_id):
            raise UnknownItemError(f"Game '{raw_game_id}' is not in item mapper.")
        u_idx = int(self._user_mapper.to_idx(raw_user_id))
        i_idx = int(self._item_mapper.to_idx(raw_game_id))
        self._extra_seen[u_idx].add(i_idx)
        if rating == 1.0:
            self._user_positive_items.setdefault(u_idx, set()).add(i_idx)

    def _get_combined_seen(self, user_idx: int) -> Set[int]:
        seen = set(self._user_seen_items.get(user_idx, ()))
        seen.update(self._extra_seen.get(user_idx, ()))
        return seen

    def recommend_games(
        self,
        raw_user_id: str,
        top_k: int = 10,
        exclude_played: bool = True,
        genre_filter: Any = None,
        max_price: Optional[float] = None,
    ) -> List[RecommendationResult]:
        raw_user_id = validate_id(raw_user_id, "raw_user_id")
        top_k = validate_top_k(top_k)
        if not isinstance(exclude_played, bool):
            raise InvalidParameterError("exclude_played must be boolean.")
        criteria = parse_filter_criteria(genre_filter, max_price)

        is_fallback = False
        mapped_user_idx: Optional[int] = None

        if self._user_mapper.contains_raw(raw_user_id):
            mapped_user_idx = int(self._user_mapper.to_idx(raw_user_id))
            user_pos = self._user_positive_items.get(mapped_user_idx, set())
            if not user_pos:
                if not self._allow_cold_start_fallback:
                    raise ColdStartError(
                        f"User '{raw_user_id}' has no positive train interactions "
                        "and cold-start fallback is disabled."
                    )
                is_fallback = True
        else:
            if not self._allow_cold_start_fallback:
                raise UnknownUserError(f"User '{raw_user_id}' is not in mapping.")
            is_fallback = True

        candidates = np.arange(self._item_mapper.size, dtype=np.int64)
        if exclude_played and mapped_user_idx is not None:
            seen_items = self._get_combined_seen(mapped_user_idx)
            if seen_items:
                seen_arr = checked_indices(list(seen_items), self._item_mapper.size)
                candidates = np.setdiff1d(candidates, seen_arr, assume_unique=True)

        if candidates.size == 0:
            return []

        active_model_name = self.model_name
        if is_fallback:
            if self._fallback_model is None:
                raise ConfigurationError("Cold-start fallback triggered but fallback_model is not configured.")
            raw_scores = self._fallback_model.score_global(candidates)
            scores = checked_scores(raw_scores, len(candidates))
            active_model_name = getattr(self._fallback_model, "name", "Popularity")
        else:
            assert mapped_user_idx is not None
            raw_scores = self._model.score_items(mapped_user_idx, candidates)
            scores = checked_scores(raw_scores, len(candidates))

        order = np.lexsort((candidates, -scores))

        results: List[RecommendationResult] = []
        for idx in order:
            item_idx = int(candidates[idx])
            raw_game_id = str(self._item_mapper.to_raw(item_idx))
            if raw_game_id not in self._catalog:
                continue
            game = self._catalog[raw_game_id]
            if not game_passes_filters(game, criteria):
                continue
            scored = ScoredItem(item_idx=item_idx, score=float(scores[idx]))
            results.append(to_recommendation_result(
                scored, game, len(results) + 1, active_model_name, is_fallback
            ))
            if len(results) == top_k:
                break

        return results

    def get_user_played_games(self, raw_user_id: str) -> List[GameItem]:
        raw_user_id = validate_id(raw_user_id, "raw_user_id")
        if not self._user_mapper.contains_raw(raw_user_id):
            raise UnknownUserError(f"User '{raw_user_id}' is not in mapping.")
        u_idx = int(self._user_mapper.to_idx(raw_user_id))
        seen_indices = sorted(self._get_combined_seen(u_idx))
        games: List[GameItem] = []
        for i_idx in seen_indices:
            raw_id = str(self._item_mapper.to_raw(i_idx))
            if raw_id not in self._catalog:
                raise UnknownItemError(f"Game '{raw_id}' is not in catalog.")
            games.append(self._catalog[raw_id])
        return games

    def get_game_details(self, raw_game_id: str) -> GameItem:
        raw_game_id = validate_id(raw_game_id, "raw_game_id")
        if raw_game_id not in self._catalog:
            raise UnknownItemError(f"Game '{raw_game_id}' is not in catalog.")
        return self._catalog[raw_game_id]

    @staticmethod
    def format_recommendations_table(results: Sequence[RecommendationResult]) -> str:
        return format_recommendations_table(results)


__all__ = [
    "PretrainedDatasetProxy",
    "PretrainedServiceFacade",
]
