from __future__ import annotations

from importlib import import_module
from typing import Any, List, Optional, Sequence

import numpy as np

from src.contracts.dto import (
    EvaluationResult,
    GameItem,
    RecommendationResult,
    ScoredItem,
)
from src.contracts.exceptions import (
    ColdStartError,
    ConfigurationError,
    DataValidationError,
    EmptyDatasetError,
    InvalidParameterError,
    ModelNotFittedError,
    RecSysError,
    TrainingError,
    UnknownItemError,
    UnknownUserError,
)
from src.contracts.interfaces import (
    BaseEvaluator,
    BaseRecommender,
    checked_indices,
    checked_scores,
)
from src.service.filters import FilterCriteria, game_passes_filters, parse_filter_criteria
from src.service.formatters import format_recommendations_table
from src.service.mappers import to_recommendation_result
from src.service.validators import (
    model_matches_dataset,
    validate_dataset,
    validate_evaluation_dataset,
    validate_fallback_model,
    validate_id,
    validate_top_k,
)


class RecommendationService:
    """Facade coordinating model inference, data access, filtering, and evaluation."""

    def __init__(
        self,
        model: BaseRecommender,
        evaluator: Optional[BaseEvaluator] = None,
        allow_cold_start_fallback: bool = False,
        fallback_model: Optional[BaseRecommender] = None,
        train_dataset: Optional[Any] = None,
    ) -> None:
        if not isinstance(model, BaseRecommender):
            raise InvalidParameterError("model must be an instance of BaseRecommender.")
        if evaluator is not None and not isinstance(evaluator, BaseEvaluator):
            raise InvalidParameterError("evaluator must be an instance of BaseEvaluator.")
        if not isinstance(allow_cold_start_fallback, bool):
            raise InvalidParameterError("allow_cold_start_fallback must be a boolean.")
        if fallback_model is not None:
            validate_fallback_model(fallback_model)

        self._model = model
        self._evaluator = evaluator
        self._allow_cold_start_fallback = allow_cold_start_fallback
        self._fallback_model: Optional[BaseRecommender] = fallback_model

        self._train_dataset: Any = None
        self._fallback_dataset: Any = None

        if train_dataset is not None:
            if hasattr(train_dataset, "train") and hasattr(train_dataset, "validation") and hasattr(train_dataset, "test"):
                train_dataset = train_dataset.train
            validate_dataset(train_dataset, "train")
            if train_dataset.split_name != "train":
                raise DataValidationError("train_dataset must have split_name='train'.")
            if not getattr(self._model, "is_fitted", False):
                raise ModelNotFittedError("model must be fitted if train_dataset is provided at initialization.")
            if not model_matches_dataset(self._model, train_dataset):
                raise DataValidationError("model dimensions/fingerprints do not match the provided train_dataset.")
            self._train_dataset = train_dataset
            if self._allow_cold_start_fallback and self._fallback_model is not None:
                if getattr(self._fallback_model, "is_fitted", False) and model_matches_dataset(self._fallback_model, train_dataset):
                    self._fallback_dataset = train_dataset

    @property
    def is_ready(self) -> bool:
        return (
            self._train_dataset is not None
            and bool(self._model.is_fitted)
            and model_matches_dataset(self._model, self._train_dataset)
        )

    @property
    def model(self) -> BaseRecommender:
        return self._model

    @property
    def model_name(self) -> str:
        return self._model.name

    @property
    def fallback_model(self) -> Optional[BaseRecommender]:
        return self._fallback_model

    @fallback_model.setter
    def fallback_model(self, fallback: Optional[BaseRecommender]) -> None:
        if fallback is not None:
            validate_fallback_model(fallback)
            if self._train_dataset is not None and self._allow_cold_start_fallback:
                self._fit_model(fallback, self._train_dataset, {}, "Fallback model")
                self._fallback_dataset = self._train_dataset
        else:
            self._fallback_dataset = None
        self._fallback_model = fallback

    @property
    def dataset(self) -> Any:
        self._ensure_ready()
        return self._train_dataset

    @property
    def train_dataset(self) -> Any:
        return self.dataset

    @property
    def evaluator(self) -> Optional[BaseEvaluator]:
        return self._evaluator

    @evaluator.setter
    def evaluator(self, evaluator: Optional[BaseEvaluator]) -> None:
        if evaluator is not None and not isinstance(evaluator, BaseEvaluator):
            raise InvalidParameterError("evaluator must be an instance of BaseEvaluator.")
        self._evaluator = evaluator

    @property
    def allow_cold_start_fallback(self) -> bool:
        return self._allow_cold_start_fallback

    @allow_cold_start_fallback.setter
    def allow_cold_start_fallback(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise InvalidParameterError("allow_cold_start_fallback must be a boolean.")
        if value and not self._allow_cold_start_fallback and self._train_dataset is not None:
            fallback = self._fallback_model or self._new_popularity_model()
            validate_fallback_model(fallback)
            self._fit_model(fallback, self._train_dataset, {}, "Fallback model")
            self._fallback_model = fallback
            self._fallback_dataset = self._train_dataset
        self._allow_cold_start_fallback = value

    def _ensure_ready(self) -> None:
        if not self.is_ready:
            raise ModelNotFittedError("Service must be trained before calling operations.")

    _validate_id = staticmethod(validate_id)
    _validate_top_k = staticmethod(validate_top_k)
    _validate_dataset = staticmethod(validate_dataset)
    _model_matches_dataset = staticmethod(model_matches_dataset)
    _validate_fallback_model = staticmethod(validate_fallback_model)

    @classmethod
    def _fit_model(cls, model: BaseRecommender, dataset: Any, fit_kwargs: dict, role: str) -> None:
        try:
            model.fit(dataset, **fit_kwargs)
            if not model_matches_dataset(model, dataset):
                raise TrainingError(f"{role} did not finish fitted with the train dimensions/fingerprints.")
        except RecSysError:
            raise
        except Exception as exc:
            raise TrainingError(f"{role} training failed: {exc}") from exc

    def _user_idx(self, raw_user_id: str) -> int:
        mapper = self._train_dataset.user_mapper
        if not mapper.contains_raw(raw_user_id):
            raise UnknownUserError(f"User '{raw_user_id}' is not in the train mapping.")
        return int(mapper.to_idx(raw_user_id))

    def _item_idx(self, raw_item_id: str) -> int:
        mapper = self._train_dataset.item_mapper
        if not mapper.contains_raw(raw_item_id):
            raise UnknownItemError(f"Game '{raw_item_id}' is not in the train mapping.")
        return int(mapper.to_idx(raw_item_id))

    def _raw_item_id(self, item_idx: int) -> str:
        return str(self._train_dataset.item_mapper.to_raw(int(item_idx)))

    def _seen_items(self, user_idx: int) -> frozenset[int]:
        return frozenset(self._train_dataset.get_user_seen_items(user_idx))

    def _positive_items(self, user_idx: int) -> frozenset[int]:
        return frozenset(self._train_dataset.get_user_positive_items(user_idx))

    def _new_popularity_model(self) -> BaseRecommender:
        for module_name in (
            "src.models",
            "src.models.popularity",
            "src.models.popularity_recommender",
            "src.model",
            "src.model.popularity",
            "models.popularity",
            "models",
            "popularity",
        ):
            try:
                module = import_module(module_name)
                model_cls = getattr(module, "PopularityRecommender", None)
                if model_cls is not None:
                    model = model_cls()
                    validate_fallback_model(model)
                    return model
            except (ImportError, AttributeError, TypeError, InvalidParameterError, ConfigurationError):
                continue
        raise ConfigurationError(
            "A global popularity fallback model is not available. "
            "Ensure src.models.PopularityRecommender is importable."
        )

    def _ensure_fallback_ready(self) -> None:
        if self._fallback_model is None:
            self._fallback_model = self._new_popularity_model()
        if self._fallback_dataset is not self._train_dataset or not model_matches_dataset(self._fallback_model, self._train_dataset):
            validate_fallback_model(self._fallback_model)
            self._fit_model(self._fallback_model, self._train_dataset, {}, "Fallback model")
            self._fallback_dataset = self._train_dataset

    def _fallback_scores(self, exclude_seen_for_user: Optional[int]) -> List[ScoredItem]:
        candidates = np.arange(self._train_dataset.num_items, dtype=np.int64)
        if exclude_seen_for_user is not None:
            seen_set = self._seen_items(exclude_seen_for_user)
            if seen_set:
                seen_arr = checked_indices(list(seen_set), self._train_dataset.num_items)
                candidates = np.setdiff1d(candidates, seen_arr, assume_unique=True)
        if candidates.size == 0:
            return []
        scores = checked_scores(
            self._fallback_model.score_global(candidates), len(candidates)
        )
        order = np.lexsort((candidates, -scores))
        return [ScoredItem(int(candidates[i]), float(scores[i])) for i in order]

    def train(
        self,
        train_dataset: Any,
        *,
        validation_score: Any = None,
        item_features: Any = None,
        verbose: bool = False,
    ) -> "RecommendationService":
        if train_dataset is None:
            raise EmptyDatasetError("Cannot train on None dataset.")
        if hasattr(train_dataset, "train") and hasattr(train_dataset, "validation") and hasattr(train_dataset, "test"):
            train_dataset = train_dataset.train

        validate_dataset(train_dataset, "train")
        if train_dataset.split_name != "train":
            raise DataValidationError(
                "RecommendationService.train() requires a dataset with split_name='train'."
            )
        if train_dataset.num_interactions == 0:
            raise EmptyDatasetError("Cannot train on an empty dataset.")
        if validation_score is not None and not callable(validation_score):
            raise InvalidParameterError("validation_score must be callable or None.")
        if not isinstance(verbose, bool):
            raise InvalidParameterError("verbose must be a boolean.")

        # Reset state so failure leaves the service unready
        self._train_dataset = None
        self._fallback_dataset = None

        fit_kwargs: dict = {}
        if validation_score is not None:
            fit_kwargs["validation_score"] = validation_score
        if item_features is not None:
            fit_kwargs["item_features"] = item_features
        if verbose:
            fit_kwargs["verbose"] = True

        try:
            self._fit_model(self._model, train_dataset, fit_kwargs, "Primary model")
            fallback = None
            if self._allow_cold_start_fallback:
                fallback = self._fallback_model or self._new_popularity_model()
                validate_fallback_model(fallback)
                self._fit_model(fallback, train_dataset, {}, "Fallback model")
        except Exception:
            self._train_dataset = None
            self._fallback_dataset = None
            raise

        self._train_dataset = train_dataset
        if self._allow_cold_start_fallback and fallback is not None:
            self._fallback_model = fallback
            self._fallback_dataset = train_dataset

        return self

    def swap_model(
        self,
        new_model: BaseRecommender,
        retrain_immediately: bool = True,
        *,
        validation_score: Any = None,
        item_features: Any = None,
        verbose: bool = False,
    ) -> None:
        if not isinstance(new_model, BaseRecommender):
            raise InvalidParameterError("new_model must be a BaseRecommender.")
        if not isinstance(retrain_immediately, bool):
            raise InvalidParameterError("retrain_immediately must be a boolean.")
        if retrain_immediately:
            self._ensure_ready()
            fit_kwargs: dict = {}
            if validation_score is not None:
                fit_kwargs["validation_score"] = validation_score
            if item_features is not None:
                fit_kwargs["item_features"] = item_features
            if verbose:
                fit_kwargs["verbose"] = True
            self._fit_model(new_model, self._train_dataset, fit_kwargs, "Replacement model")
        self._model = new_model

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
            raise InvalidParameterError("exclude_played must be a boolean.")
        criteria = parse_filter_criteria(genre_filter, max_price)
        self._ensure_ready()

        is_fallback = False
        mapped_user_idx: Optional[int] = None

        try:
            mapped_user_idx = self._user_idx(raw_user_id)
        except UnknownUserError:
            if not self._allow_cold_start_fallback:
                raise
            is_fallback = True

        if not is_fallback:
            assert mapped_user_idx is not None
            if not self._positive_items(mapped_user_idx):
                if not self._allow_cold_start_fallback:
                    raise ColdStartError(
                        f"User '{raw_user_id}' has no positive train interactions "
                        "and cold-start fallback is disabled."
                    )
                is_fallback = True

        if is_fallback:
            self._ensure_fallback_ready()
            # Fallback excludes seen items (both positive and negative) for known users
            scored_items = self._fallback_scores(
                exclude_seen_for_user=mapped_user_idx if exclude_played else None
            )
        else:
            assert mapped_user_idx is not None
            # Over-fetch when filters are active to satisfy top_k
            if criteria.genres is not None or criteria.max_price is not None:
                request_k = max(top_k, self._train_dataset.num_items)
            else:
                request_k = top_k
            scored_items = self._model.recommend(
                mapped_user_idx,
                self._train_dataset,
                top_k=request_k,
                exclude_seen=exclude_played,
            )

        results: List[RecommendationResult] = []
        for scored in scored_items:
            raw_game_id = self._raw_item_id(scored.item_idx)
            try:
                game: GameItem = self._train_dataset.get_game_by_raw_id(raw_game_id)
            except (KeyError, IndexError, UnknownItemError):
                continue
            if not isinstance(game, GameItem):
                raise DataValidationError("get_game_by_raw_id must return a GameItem instance.")
            if not game_passes_filters(game, criteria):
                continue

            active_model_name = (
                self._fallback_model.name
                if is_fallback and self._fallback_model is not None and getattr(self._fallback_model, "name", None)
                else self.model_name
            )
            results.append(to_recommendation_result(
                scored, game, len(results) + 1, active_model_name, is_fallback
            ))
            if len(results) == top_k:
                break

        if (
            not is_fallback
            and len(results) < top_k
            and request_k < self._train_dataset.num_items
            and len(scored_items) == request_k
        ):
            all_scored = self._model.recommend(
                mapped_user_idx,
                self._train_dataset,
                top_k=self._train_dataset.num_items,
                exclude_seen=exclude_played,
            )
            seen_result_ids = {r.game_id for r in results}
            for scored in all_scored:
                raw_game_id = self._raw_item_id(scored.item_idx)
                if raw_game_id in seen_result_ids:
                    continue
                try:
                    game = self._train_dataset.get_game_by_raw_id(raw_game_id)
                except (KeyError, IndexError, UnknownItemError):
                    continue
                if not isinstance(game, GameItem):
                    raise DataValidationError("get_game_by_raw_id must return a GameItem instance.")
                if not game_passes_filters(game, criteria):
                    continue
                results.append(to_recommendation_result(
                    scored, game, len(results) + 1, self.model_name, is_fallback=False
                ))
                seen_result_ids.add(raw_game_id)
                if len(results) == top_k:
                    break

        return results

    def get_user_played_games(self, raw_user_id: str) -> List[GameItem]:
        self._ensure_ready()
        user_idx = self._user_idx(validate_id(raw_user_id, "raw_user_id"))
        games: List[GameItem] = []
        for item_idx in sorted(self._seen_items(user_idx)):
            raw_item_id = self._raw_item_id(item_idx)
            try:
                game = self._train_dataset.get_game_by_raw_id(raw_item_id)
            except (KeyError, IndexError, UnknownItemError):
                raise UnknownItemError(f"Game '{raw_item_id}' is not in the train catalog.")
            if not isinstance(game, GameItem):
                raise DataValidationError("get_game_by_raw_id must return a GameItem instance.")
            games.append(game)
        return games

    def get_game_details(self, raw_game_id: str) -> GameItem:
        self._ensure_ready()
        raw_game_id = validate_id(raw_game_id, "raw_game_id")
        try:
            game = self._train_dataset.get_game_by_raw_id(raw_game_id)
        except (KeyError, IndexError, UnknownItemError):
            raise UnknownItemError(f"Game '{raw_game_id}' is not in the train catalog.")
        if not isinstance(game, GameItem):
            raise DataValidationError("get_game_by_raw_id must return a GameItem instance.")
        return game

    def get_available_user_ids(self, limit: Optional[int] = None) -> tuple[str, ...]:
        """Return raw user IDs, optionally sampled for a UI selector."""
        self._ensure_ready()
        mapper = self._train_dataset.user_mapper
        if limit is not None and limit > 0 and mapper.size > limit:
            step = max(1, mapper.size // limit)
            indices = list(range(0, mapper.size, step))[:limit]
            return tuple(str(mapper.to_raw(index)) for index in indices)
        return tuple(str(mapper.to_raw(index)) for index in range(mapper.size))

    def get_available_game_ids(self) -> tuple[str, ...]:
        """Return raw game IDs available in the fitted catalog mapping."""
        self._ensure_ready()
        mapper = self._train_dataset.item_mapper
        return tuple(str(mapper.to_raw(index)) for index in range(mapper.size))

    def get_available_genres(self) -> tuple[str, ...]:
        """Return sorted catalog genres for UI filters."""
        self._ensure_ready()
        genres: set[str] = set()
        for raw_game_id in self.get_available_game_ids():
            try:
                game = self._train_dataset.get_game_by_raw_id(raw_game_id)
            except (KeyError, IndexError, UnknownItemError):
                continue
            genres.update(str(genre).strip() for genre in game.genres if str(genre).strip())
        return tuple(sorted(genres, key=str.casefold))

    def recommend_baseline(
        self,
        raw_user_id: Optional[str] = None,
        top_k: int = 10,
        exclude_played: bool = True,
        genre_filter: Any = None,
        max_price: Optional[float] = None,
    ) -> List[RecommendationResult]:
        """Generate baseline recommendations from the popularity fallback model."""
        self._ensure_ready()
        self._ensure_fallback_ready()
        mapped_user_idx: Optional[int] = None
        if raw_user_id is not None and str(raw_user_id).strip():
            try:
                mapped_user_idx = self._user_idx(validate_id(str(raw_user_id).strip(), "raw_user_id"))
            except (UnknownUserError, InvalidParameterError):
                mapped_user_idx = None
        scored_items = self._fallback_scores(
            exclude_seen_for_user=mapped_user_idx if exclude_played else None
        )
        criteria = parse_filter_criteria(genre_filter, max_price)
        top_k = validate_top_k(top_k)
        results: List[RecommendationResult] = []
        model_name = getattr(self._fallback_model, "name", "Popularity baseline")
        for scored in scored_items:
            raw_game_id = self._train_dataset.item_mapper.to_raw(scored.item_idx)
            try:
                game = self._train_dataset.get_game_by_raw_id(raw_game_id)
            except (KeyError, IndexError, UnknownItemError):
                continue
            if not game_passes_filters(game, criteria):
                continue
            results.append(
                to_recommendation_result(
                    scored,
                    game,
                    len(results) + 1,
                    model_name,
                    True,
                )
            )
            if len(results) == top_k:
                break
        return results

    def evaluate_model(
        self,
        evaluation_dataset: Any,
        *,
        additional_seen_dataset: Any = None,
    ) -> EvaluationResult:
        self._ensure_ready()
        if self._evaluator is None:
            raise ConfigurationError("No evaluator was supplied to RecommendationService.")

        validate_evaluation_dataset(
            evaluation_dataset,
            self._train_dataset,
            additional_seen_dataset=additional_seen_dataset,
        )

        result = self._evaluator.evaluate(
            self._model,
            self._train_dataset,
            evaluation_dataset,
            additional_seen_dataset=additional_seen_dataset,
        )
        if not isinstance(result, EvaluationResult):
            raise DataValidationError("Evaluator must return an EvaluationResult.")
        return result

    @staticmethod
    def format_recommendations_table(results: Sequence[RecommendationResult]) -> str:
        return format_recommendations_table(results)



    @classmethod
    def load_pretrained(
        cls,
        folder: Any,
        *,
        allow_cold_start_fallback: bool = False,
        catalog_lookup: Any = None,
    ) -> Any:
        """Load a trained model bundle and return a PretrainedServiceFacade (Contracts v1.1 Section 6)."""
        from src.service.pretrained import PretrainedServiceFacade
        return PretrainedServiceFacade.load(
            folder,
            allow_cold_start_fallback=allow_cold_start_fallback,
            catalog_lookup=catalog_lookup,
        )


__all__ = ["RecommendationService"]
