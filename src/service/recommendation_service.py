"""Application service for the public recommendation API."""

from __future__ import annotations

from importlib import import_module
from math import isfinite
from typing import Any, Iterable, List, Optional, Sequence, Set

import numpy as np

from src.contracts.dto import RecommendationResult, ScoredItem
from src.contracts.exceptions import (
    EmptyDatasetError,
    InvalidParameterError,
    ModelNotFittedError,
    UnknownItemError,
    UnknownUserError,
)
from src.contracts.interfaces import BaseRecommender


class RecommendationService:
    def __init__(
        self,
        model: BaseRecommender,
        evaluator: Any = None,
        allow_cold_start_fallback: bool = False,
        fallback_model: Optional[BaseRecommender] = None,
    ) -> None:
        if not isinstance(model, BaseRecommender):
            raise InvalidParameterError("model must be an instance of BaseRecommender.")
        if evaluator is not None and not callable(getattr(evaluator, "evaluate", None)):
            raise InvalidParameterError("evaluator must provide an evaluate() method.")
        if fallback_model is not None:
            self._validate_fallback_model(fallback_model)
        if not isinstance(allow_cold_start_fallback, bool):
            raise InvalidParameterError("allow_cold_start_fallback must be a boolean.")

        self._model = model
        self._evaluator = evaluator
        self._allow_cold_start_fallback = allow_cold_start_fallback
        self._fallback_model = fallback_model
        self._train_dataset: Any = None
        self._fallback_dataset: Any = None

    @property
    def is_ready(self) -> bool:
        return self._train_dataset is not None and self._model.is_fitted

    @property
    def model_name(self) -> str:
        return self._model.name

    @property
    def dataset(self) -> Any:
        self._ensure_ready()
        return self._train_dataset

    @property
    def allow_cold_start_fallback(self) -> bool:
        return self._allow_cold_start_fallback

    @allow_cold_start_fallback.setter
    def allow_cold_start_fallback(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise InvalidParameterError("allow_cold_start_fallback must be a boolean.")
        if value and self._train_dataset is not None:
            fallback = self._fallback_model or self._new_popularity_model()
            self._validate_fallback_model(fallback)
            fallback.fit(self._train_dataset)
            self._fallback_model = fallback
            self._fallback_dataset = self._train_dataset
        self._allow_cold_start_fallback = value

    def _ensure_ready(self) -> None:
        if self._train_dataset is None or not self._model.is_fitted:
            raise ModelNotFittedError("Service must be trained before calling operations.")

    @staticmethod
    def _validate_id(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InvalidParameterError(f"{name} must be a non-empty string.")
        return value.strip()

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise InvalidParameterError(f"top_k must be a positive integer, got {top_k!r}.")
        return top_k

    @staticmethod
    def _validate_dataset(dataset: Any, role: str) -> None:
        required = (
            "split_name",
            "interactions",
            "num_interactions",
            "num_users",
            "num_items",
            "user_mapper",
            "item_mapper",
            "mapping_fingerprint",
            "fingerprint",
            "review_fingerprint",
            "get_game_by_raw_id",
            "get_user_seen_items",
            "get_user_positive_items",
            "iter_reviews",
        )
        missing = [name for name in required if not hasattr(dataset, name)]
        if missing:
            raise InvalidParameterError(
                f"{role} dataset is missing required fields: {', '.join(missing)}."
            )
        if dataset.split_name not in {"train", "validation", "test"}:
            raise InvalidParameterError("dataset.split_name must be train, validation or test.")
        if int(dataset.num_interactions) < 0:
            raise InvalidParameterError("dataset.num_interactions must be non-negative.")

    @staticmethod
    def _validate_fallback_model(model: BaseRecommender) -> None:
        if not isinstance(model, BaseRecommender):
            raise InvalidParameterError("fallback_model must be a BaseRecommender instance.")
        if not callable(getattr(model, "score_global", None)):
            raise InvalidParameterError("fallback_model must provide score_global(item_ids).")

    @staticmethod
    def _mapper_contains(mapper: Any, raw_id: str) -> bool:
        return bool(mapper.contains_raw(raw_id))

    def _user_idx(self, raw_user_id: str) -> int:
        if not self._mapper_contains(self._train_dataset.user_mapper, raw_user_id):
            raise UnknownUserError(f"User '{raw_user_id}' is not in the train mapping.")
        return int(self._train_dataset.user_mapper.to_idx(raw_user_id))

    def _item_idx(self, raw_item_id: str) -> int:
        if not self._mapper_contains(self._train_dataset.item_mapper, raw_item_id):
            raise UnknownItemError(f"Game '{raw_item_id}' is not in the train mapping.")
        return int(self._train_dataset.item_mapper.to_idx(raw_item_id))

    def _raw_item_id(self, item_idx: int) -> str:
        return str(self._train_dataset.item_mapper.to_raw(int(item_idx)))

    def _seen_items(self, user_idx: int) -> frozenset[int]:
        return frozenset(self._train_dataset.get_user_seen_items(user_idx))

    def _positive_items(self, user_idx: int) -> frozenset[int]:
        return frozenset(self._train_dataset.get_user_positive_items(user_idx))

    @staticmethod
    def _normalise_genres(genre_filter: Any) -> Optional[Set[str]]:
        if genre_filter is None:
            return None
        if isinstance(genre_filter, str):
            values = [genre_filter]
        else:
            try:
                values = list(genre_filter)
            except TypeError as error:
                raise InvalidParameterError("genre_filter must be a string or iterable of strings.") from error
        if not values or any(not isinstance(value, str) or not value.strip() for value in values):
            raise InvalidParameterError("genre_filter must contain non-empty strings.")
        return {value.strip().casefold() for value in values}

    @staticmethod
    def _validate_price(max_price: Optional[float]) -> Optional[float]:
        if max_price is None:
            return None
        if isinstance(max_price, bool):
            raise InvalidParameterError("max_price must be a finite non-negative number.")
        try:
            value = float(max_price)
        except (TypeError, ValueError) as error:
            raise InvalidParameterError("max_price must be a finite non-negative number.") from error
        if not isfinite(value) or value < 0.0:
            raise InvalidParameterError("max_price must be a finite non-negative number.")
        return value

    def _passes_filter(
        self,
        game: Any,
        genres: Optional[Set[str]],
        max_price: Optional[float],
    ) -> bool:
        if genres is not None:
            game_genres = getattr(game, "genres", ()) or ()
            if isinstance(game_genres, str):
                game_genres = (game_genres,)
            if not genres.intersection({str(value).casefold() for value in game_genres}):
                return False
        if max_price is not None:
            price = getattr(game, "price", None)
            if price is None:
                return False
            try:
                price = float(price)
            except (TypeError, ValueError):
                return False
            if not isfinite(price) or price > max_price:
                return False
        return True

    def _to_result(self, scored: ScoredItem, rank: int, is_fallback: bool) -> RecommendationResult:
        raw_game_id = self._raw_item_id(scored.item_idx)
        game = self._train_dataset.get_game_by_raw_id(raw_game_id)
        genres = getattr(game, "genres", ()) or ()
        if isinstance(genres, str):
            genres = (genres,)
        price = getattr(game, "price", None)
        if price is not None:
            price = float(price)
        return RecommendationResult(
            rank=rank,
            game_id=str(getattr(game, "item_id", raw_game_id)),
            title=str(getattr(game, "title", "")),
            genres=tuple(genres),
            predicted_score=round(float(scored.score), 4),
            developer=str(getattr(game, "developer", "Unknown") or "Unknown"),
            price=price,
            model_name=self.model_name,
            is_fallback=is_fallback,
        )

    def _new_popularity_model(self) -> BaseRecommender:
        for module_name in ("src.models.popularity", "popularity"):
            try:
                module = import_module(module_name)
                model = getattr(module, "PopularityRecommender")()
                self._validate_fallback_model(model)
                return model
            except (ImportError, AttributeError, TypeError, InvalidParameterError):
                continue
        raise InvalidParameterError("A global popularity fallback model is not available.")

    def _fallback_scores(self, user_idx: Optional[int]) -> List[ScoredItem]:
        candidates = np.arange(self._train_dataset.num_items, dtype=np.int64)
        if user_idx is not None:
            seen = self._seen_items(user_idx)
            candidates = np.asarray([item for item in candidates if int(item) not in seen], dtype=np.int64)
        if candidates.size == 0:
            return []
        scores = np.asarray(self._fallback_model.score_global(candidates), dtype=float)
        if scores.ndim != 1 or scores.size != candidates.size or not np.all(np.isfinite(scores)):
            raise InvalidParameterError("fallback_model.score_global() returned invalid scores.")
        order = np.lexsort((candidates, -scores))
        return [ScoredItem(int(candidates[i]), float(scores[i])) for i in order]

    def train(
        self,
        train_dataset: Any,
        *,
        validation_score: Any = None,
        item_features: Any = None,
    ) -> "RecommendationService":
        self._validate_dataset(train_dataset, "train")
        if train_dataset.split_name != "train":
            raise InvalidParameterError("RecommendationService.train() requires a train dataset.")
        if train_dataset.num_interactions == 0:
            raise EmptyDatasetError("Cannot train on an empty dataset.")
        if validation_score is not None and not callable(validation_score):
            raise InvalidParameterError("validation_score must be callable or None.")

        fit_kwargs = {"validation_score": validation_score, "item_features": item_features}
        if validation_score is None:
            fit_kwargs.pop("validation_score")
        if item_features is None:
            fit_kwargs.pop("item_features")

        self._train_dataset = None
        self._fallback_dataset = None
        try:
            self._model.fit(train_dataset, **fit_kwargs)
            fallback = None
            if self._allow_cold_start_fallback:
                fallback = self._fallback_model or self._new_popularity_model()
                fallback.fit(train_dataset)
        except Exception:
            self._train_dataset = None
            self._fallback_dataset = None
            raise

        self._train_dataset = train_dataset
        self._fallback_model = fallback if self._allow_cold_start_fallback else None
        self._fallback_dataset = train_dataset if fallback is not None else None
        return self

    def swap_model(self, new_model: BaseRecommender, retrain_immediately: bool = True) -> None:
        if not isinstance(new_model, BaseRecommender):
            raise InvalidParameterError("new_model must be a BaseRecommender.")
        if not isinstance(retrain_immediately, bool):
            raise InvalidParameterError("retrain_immediately must be a boolean.")
        if retrain_immediately:
            self._ensure_ready()
            new_model.fit(self._train_dataset)
        self._model = new_model

    def recommend_games(
        self,
        raw_user_id: str,
        top_k: int = 10,
        exclude_played: bool = True,
        genre_filter: Any = None,
        max_price: Optional[float] = None,
    ) -> List[RecommendationResult]:
        raw_user_id = self._validate_id(raw_user_id, "raw_user_id")
        top_k = self._validate_top_k(top_k)
        if not isinstance(exclude_played, bool):
            raise InvalidParameterError("exclude_played must be a boolean.")
        genres = self._normalise_genres(genre_filter)
        max_price = self._validate_price(max_price)
        self._ensure_ready()

        is_fallback = False
        user_idx: Optional[int]
        try:
            user_idx = self._user_idx(raw_user_id)
            if not self._positive_items(user_idx):
                raise UnknownUserError(f"User '{raw_user_id}' has no positive train profile.")
        except UnknownUserError:
            if not self._allow_cold_start_fallback:
                raise
            if self._fallback_model is None or self._fallback_dataset is not self._train_dataset:
                self._fallback_model = self._fallback_model or self._new_popularity_model()
                self._fallback_model.fit(self._train_dataset)
                self._fallback_dataset = self._train_dataset
            user_idx = None
            is_fallback = True

        if is_fallback:
            scored_items = self._fallback_scores(None)
        else:
            scored_items = self._model.recommend(
                user_idx,
                self._train_dataset,
                max(top_k, self._train_dataset.num_items),
                exclude_played,
            )

        results: List[RecommendationResult] = []
        for scored in scored_items:
            game = self._train_dataset.get_game_by_raw_id(self._raw_item_id(scored.item_idx))
            if not self._passes_filter(game, genres, max_price):
                continue
            results.append(self._to_result(scored, len(results) + 1, is_fallback))
            if len(results) == top_k:
                break
        return results

    def get_user_played_games(self, raw_user_id: str) -> List[Any]:
        self._ensure_ready()
        user_idx = self._user_idx(self._validate_id(raw_user_id, "raw_user_id"))
        return [
            self._train_dataset.get_game_by_raw_id(self._raw_item_id(item_idx))
            for item_idx in sorted(self._seen_items(user_idx))
        ]

    def get_game_details(self, raw_game_id: str) -> Any:
        self._ensure_ready()
        raw_game_id = self._validate_id(raw_game_id, "raw_game_id")
        if not self._mapper_contains(self._train_dataset.item_mapper, raw_game_id):
            raise UnknownItemError(f"Game '{raw_game_id}' is not in the train mapping.")
        return self._train_dataset.get_game_by_raw_id(raw_game_id)

    def evaluate_model(
        self,
        evaluation_dataset: Any,
        *,
        additional_seen_dataset: Any = None,
    ) -> Any:
        self._ensure_ready()
        if self._evaluator is None:
            raise InvalidParameterError("Evaluator is not set.")
        self._validate_dataset(evaluation_dataset, "evaluation")
        if evaluation_dataset.split_name not in {"validation", "test"}:
            raise InvalidParameterError("evaluation_dataset must be validation or test.")
        if evaluation_dataset.mapping_fingerprint != self._train_dataset.mapping_fingerprint:
            raise InvalidParameterError("Dataset mappings do not match the train dataset.")
        if additional_seen_dataset is not None:
            self._validate_dataset(additional_seen_dataset, "additional_seen")
            if additional_seen_dataset.split_name != "validation":
                raise InvalidParameterError("additional_seen_dataset must be validation.")
            if additional_seen_dataset.mapping_fingerprint != self._train_dataset.mapping_fingerprint:
                raise InvalidParameterError("Additional seen mapping does not match train.")
        return self._evaluator.evaluate(
            self._model,
            self._train_dataset,
            evaluation_dataset,
            additional_seen_dataset=additional_seen_dataset,
        )


__all__ = ["RecommendationService"]
