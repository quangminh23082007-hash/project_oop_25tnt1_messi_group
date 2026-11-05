"""Application service for training models and serving game recommendations."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, List, Mapping, Optional, Set

from src.contracts.dto import EvaluationReport, RecommendationResult, ScoredItem
from src.contracts.exceptions import (
    EmptyDatasetError,
    InvalidParameterError,
    ModelNotFittedError,
    UnknownItemError,
    UnknownUserError,
)
from src.contracts.interfaces import BaseRecommender


class _DatasetView:
    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)

    @property
    def num_items(self) -> int:
        value = getattr(self._dataset, "num_items", None)
        if value is None:
            value = getattr(self._dataset, "num_games", None)
        if value is None:
            raise InvalidParameterError("dataset must expose num_items or num_games.")
        return int(value)

    def get_user_seen_items(self, user_idx: int) -> Set[int]:
        if hasattr(self._dataset, "get_user_seen_items"):
            return set(self._dataset.get_user_seen_items(user_idx))
        if hasattr(self._dataset, "get_interaction_matrix_data"):
            triples = self._dataset.get_interaction_matrix_data(mode="binary")
            return {int(item_idx) for u_idx, item_idx, *_ in triples if int(u_idx) == user_idx}
        return set()


class RecommendationService:
    def __init__(
        self,
        model: BaseRecommender,
        evaluator: Optional[Any] = None,
        allow_cold_start_fallback: bool = False,
        fallback_model: Optional[BaseRecommender] = None,
    ) -> None:
        if not isinstance(model, BaseRecommender):
            raise InvalidParameterError("model must be an instance of BaseRecommender.")
        if evaluator is not None and not callable(getattr(evaluator, "evaluate", None)):
            raise InvalidParameterError("evaluator must provide an evaluate() method.")
        if fallback_model is not None and not isinstance(fallback_model, BaseRecommender):
            raise InvalidParameterError("fallback_model must be a BaseRecommender instance.")
        if not isinstance(allow_cold_start_fallback, bool):
            raise InvalidParameterError("allow_cold_start_fallback must be a boolean.")

        self._model = model
        self._evaluator = evaluator
        self._allow_cold_start_fallback = allow_cold_start_fallback
        self._fallback_model = fallback_model
        self._fallback_dataset: Any = None
        self._dataset: Any = None

    @property
    def is_ready(self) -> bool:
        return self._dataset is not None and self._model.is_fitted

    @property
    def model_name(self) -> str:
        return self._model.name

    @property
    def allow_cold_start_fallback(self) -> bool:
        return self._allow_cold_start_fallback

    @allow_cold_start_fallback.setter
    def allow_cold_start_fallback(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise InvalidParameterError("allow_cold_start_fallback must be a boolean.")
        if value and self._dataset is not None:
            fallback = self._fallback_model or self._new_popularity_model()
            fallback.fit(self._dataset)
            self._fallback_model = fallback
            self._fallback_dataset = self._dataset
        self._allow_cold_start_fallback = value

    @property
    def dataset(self) -> Any:
        self._ensure_ready()
        return self._dataset

    def _ensure_ready(self) -> None:
        if self._dataset is None or not self._model.is_fitted:
            raise ModelNotFittedError("Service must be trained before calling operations.")

    @staticmethod
    def _validate_non_empty_id(raw_id: str, param_name: str = "raw_id") -> str:
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise InvalidParameterError(f"{param_name} must be a non-empty string.")
        return raw_id.strip()

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise InvalidParameterError(f"top_k must be a positive integer, got {top_k!r}.")
        return top_k

    @staticmethod
    def _dataset_is_valid(dataset: Any) -> bool:
        if dataset is None:
            return False
        if any(not hasattr(dataset, name) for name in ("num_interactions", "num_users")):
            return False
        return hasattr(dataset, "num_items") or hasattr(dataset, "num_games")

    @staticmethod
    def _mapper_contains(mapper: Any, raw_id: str) -> bool:
        if hasattr(mapper, "contains_raw"):
            return bool(mapper.contains_raw(raw_id))
        if hasattr(mapper, "contains"):
            return bool(mapper.contains(raw_id))
        try:
            mapper.to_idx(raw_id)
        except (KeyError, IndexError, ValueError, UnknownUserError, UnknownItemError):
            return False
        return True

    def _has_user(self, raw_user_id: str) -> bool:
        mapper = getattr(self._dataset, "user_mapper", None)
        if mapper is not None:
            return self._mapper_contains(mapper, raw_user_id)
        try:
            self._user_idx(raw_user_id)
        except (KeyError, IndexError, ValueError, UnknownUserError):
            return False
        return True

    def _user_idx(self, raw_user_id: str) -> int:
        mapper = getattr(self._dataset, "user_mapper", None)
        if mapper is not None:
            if hasattr(mapper, "to_idx"):
                return int(mapper.to_idx(raw_user_id))
            if hasattr(mapper, "index"):
                return int(mapper.index(raw_user_id))
        if hasattr(self._dataset, "user_index"):
            return int(self._dataset.user_index(raw_user_id))
        raise UnknownUserError(f"User '{raw_user_id}' is not available in the dataset.")

    def _item_idx(self, raw_item_id: str) -> int:
        mapper = getattr(self._dataset, "item_mapper", None)
        if mapper is not None:
            if hasattr(mapper, "to_idx"):
                return int(mapper.to_idx(raw_item_id))
            if hasattr(mapper, "index"):
                return int(mapper.index(raw_item_id))
        if hasattr(self._dataset, "game_index"):
            return int(self._dataset.game_index(raw_item_id))
        raise UnknownItemError(f"Item '{raw_item_id}' is not available in the dataset.")

    def _raw_item_id(self, item_idx: int) -> str:
        mapper = getattr(self._dataset, "item_mapper", None)
        if mapper is not None:
            if hasattr(mapper, "to_raw"):
                return str(mapper.to_raw(item_idx))
            if hasattr(mapper, "raw"):
                return str(mapper.raw(item_idx))
        if hasattr(self._dataset, "game_from_index"):
            return str(self._dataset.game_from_index(item_idx))
        raise UnknownItemError(f"Item index '{item_idx}' is not available in the dataset.")

    def _game(self, raw_item_id: str) -> Any:
        if hasattr(self._dataset, "get_game_by_raw_id"):
            return self._dataset.get_game_by_raw_id(raw_item_id)
        if hasattr(self._dataset, "get_game"):
            return self._dataset.get_game(raw_item_id)
        raise UnknownItemError(f"Item '{raw_item_id}' is not available in the dataset.")

    def _seen_items(self, user_idx: int) -> Set[int]:
        if hasattr(self._dataset, "get_user_seen_items"):
            return set(self._dataset.get_user_seen_items(user_idx))
        if hasattr(self._dataset, "get_interaction_matrix_data"):
            triples = self._dataset.get_interaction_matrix_data(mode="binary")
            return {int(item_idx) for u_idx, item_idx, *_ in triples if int(u_idx) == user_idx}
        return set()

    def _hydrate_results(
        self,
        scored_items: List[ScoredItem],
        top_k: int,
    ) -> List[RecommendationResult]:
        results: List[RecommendationResult] = []
        for scored in scored_items[:top_k]:
            raw_item_id = self._raw_item_id(scored.item_idx)
            game = self._game(raw_item_id)
            game_id = getattr(game, "item_id", None)
            if game_id is None:
                game_id = getattr(game, "game_id", raw_item_id)
            title = getattr(game, "title", None)
            if title is None:
                title = getattr(game, "name", raw_item_id)
            game_id = str(game_id)
            title = str(title)
            genres = getattr(game, "genres", ()) or ()
            if isinstance(genres, str):
                genres = (genres,)
            developer = str(getattr(game, "developer", "Unknown") or "Unknown")
            results.append(
                RecommendationResult(
                    rank=len(results) + 1,
                    game_id=game_id,
                    title=title,
                    genres=tuple(genres),
                    predicted_score=round(float(scored.score), 4),
                    developer=developer,
                )
            )
        return results

    @staticmethod
    def _new_popularity_model() -> BaseRecommender:
        for module_name in ("src.models.popularity", "popularity"):
            try:
                module = import_module(module_name)
                model_class = getattr(module, "PopularityRecommender")
                model = model_class()
                if isinstance(model, BaseRecommender):
                    return model
            except (ImportError, AttributeError, TypeError):
                continue
        raise InvalidParameterError("A popularity fallback model is not available.")

    def train(self, dataset: Any) -> "RecommendationService":
        if not self._dataset_is_valid(dataset):
            raise EmptyDatasetError("dataset must provide users, interactions and item dimensions.")
        if int(dataset.num_interactions) == 0:
            raise EmptyDatasetError("Cannot train on empty dataset.")

        dataset_view = _DatasetView(dataset)
        self._model.fit(dataset_view)
        new_fallback = None
        if self._allow_cold_start_fallback:
            new_fallback = self._fallback_model or self._new_popularity_model()
            new_fallback.fit(dataset_view)

        self._dataset = dataset_view
        self._fallback_model = new_fallback
        self._fallback_dataset = dataset_view if new_fallback is not None else None
        return self

    def swap_model(self, new_model: BaseRecommender, retrain_immediately: bool = True) -> None:
        if not isinstance(new_model, BaseRecommender):
            raise InvalidParameterError("new_model must be a BaseRecommender.")
        if not isinstance(retrain_immediately, bool):
            raise InvalidParameterError("retrain_immediately must be a boolean.")
        if retrain_immediately and self._dataset is not None:
            new_model.fit(self._dataset)
        self._model = new_model

    def recommend_games(
        self,
        raw_user_id: str,
        top_k: int = 10,
        exclude_played: bool = True,
    ) -> List[RecommendationResult]:
        clean_user_id = self._validate_non_empty_id(raw_user_id, "raw_user_id")
        valid_k = self._validate_top_k(top_k)
        if not isinstance(exclude_played, bool):
            raise InvalidParameterError("exclude_played must be a boolean.")
        self._ensure_ready()

        if self._has_user(clean_user_id):
            scored_items = self._model.recommend(
                user_idx=self._user_idx(clean_user_id),
                dataset=self._dataset,
                top_k=valid_k,
                exclude_seen=exclude_played,
            )
        else:
            if not self._allow_cold_start_fallback:
                raise UnknownUserError(f"User '{clean_user_id}' unknown and fallback disabled.")
            fallback = self._fallback_model or self._new_popularity_model()
            if not fallback.is_fitted or self._fallback_dataset is not self._dataset:
                fallback.fit(self._dataset)
                self._fallback_model = fallback
                self._fallback_dataset = self._dataset
            scored_items = fallback.recommend(
                user_idx=0,
                dataset=self._dataset,
                top_k=valid_k,
                exclude_seen=False,
            )

        return self._hydrate_results(scored_items, valid_k)

    def get_user_played_games(self, raw_user_id: str) -> List[Any]:
        clean_user_id = self._validate_non_empty_id(raw_user_id, "raw_user_id")
        self._ensure_ready()
        user_idx = self._user_idx(clean_user_id)
        return [self._game(self._raw_item_id(i)) for i in sorted(self._seen_items(user_idx))]

    def get_game_details(self, raw_game_id: str) -> Any:
        clean_game_id = self._validate_non_empty_id(raw_game_id, "raw_game_id")
        self._ensure_ready()
        return self._game(clean_game_id)

    def evaluate_model(self, test_ground_truth: Mapping[str, Set[str]]) -> EvaluationReport:
        self._ensure_ready()
        if self._evaluator is None:
            raise InvalidParameterError("Evaluator is not set.")
        if not isinstance(test_ground_truth, Mapping) or not test_ground_truth:
            raise EmptyDatasetError("test_ground_truth is empty.")

        internal_gt: Dict[int, Set[int]] = {}
        for raw_uid, raw_item_ids in test_ground_truth.items():
            clean_uid = self._validate_non_empty_id(raw_uid, "test_ground_truth user_id")
            if not self._has_user(clean_uid):
                continue
            user_idx = self._user_idx(clean_uid)
            item_indices: Set[int] = set()
            for raw_item_id in raw_item_ids:
                clean_item_id = self._validate_non_empty_id(
                    raw_item_id, "test_ground_truth item_id"
                )
                try:
                    item_indices.add(self._item_idx(clean_item_id))
                except (KeyError, IndexError, ValueError, UnknownItemError):
                    continue
            if item_indices:
                internal_gt[user_idx] = item_indices

        if not internal_gt:
            raise EmptyDatasetError("No valid evaluation pairs after mapping.")

        return self._evaluator.evaluate(
            model=self._model,
            train_dataset=self._dataset,
            test_ground_truth=internal_gt,
        )


__all__ = ["RecommendationService"]
