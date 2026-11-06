"""Application service for the public recommendation API — Contracts v1 (2026-11-05).

Source of Truth:
  - Contracts_v1_Game_Recommender_2026-11-05.html
  - contracts_models_context.txt

Responsibilities (Section 7, Contracts v1):
  - Coordinate train / recommend / evaluate using raw string IDs.
  - Delegate scoring and recommendation to BaseRecommender.
  - Apply genre / price post-filters and global-fallback (Popularity).
  - Return typed DTOs from src.contracts (RecommendationResult, GameItem, EvaluationResult).

Design rules enforced here:
  - Service never impersonates an unknown user as user 0 or any other known user.
  - Fallback (global Popularity) excludes ALL seen items (positive AND negative) for
    users who exist in the mapping but have no positive train profile.
  - ColdStartError is raised (not UnknownUserError) when a mapped user has no positive
    history and fallback is disabled.
  - UnknownUserError is raised when a user is absent from the mapping and fallback is disabled.
  - UnknownItemError is raised when an item is absent from the train catalog mapping.
  - BPR / TextBPR models are rejected as fallback models (Section 7 rule).
  - Training failure leaves Service in a not-ready state; partial state is never retained.
  - evaluate_model() returns EvaluationResult; EvaluationReport is not used in v1.
"""

from __future__ import annotations

from importlib import import_module
from math import isfinite
from numbers import Integral, Real
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.contracts.dto import (
    DatasetSplits,
    EvaluationResult,
    GameItem,
    ItemFeatures,
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
    IdMapper,
    InteractionDataset,
    checked_indices,
    checked_scores,
)


class RecommendationService:
    """Facade that wires together a fitted BaseRecommender with the Data and Evaluation layers.

    Parameters
    ----------
    model:
        Any object that satisfies BaseRecommender (Contracts v1, Section 5).
    evaluator:
        Any object that satisfies BaseEvaluator (Contracts v1, Section 6), or None.
    allow_cold_start_fallback:
        When True the Service will serve global Popularity scores for users who are
        absent from the mapping OR present but have no positive train interactions.
        Default is False (strict mode: raise ColdStartError / UnknownUserError).
    fallback_model:
        Pre-supplied fallback model that must implement score_global(item_ids).
        When None the Service lazily creates a PopularityRecommender on first need.
    train_dataset:
        Optional InteractionDataset (or DatasetSplits) that model was already fitted with.
        Default is None (service starts not-ready until train() is executed).
    """

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
            self._validate_fallback_model(fallback_model)

        self._model = model
        self._evaluator = evaluator
        self._allow_cold_start_fallback = allow_cold_start_fallback
        self._fallback_model: Optional[BaseRecommender] = fallback_model

        # Train dataset state
        self._train_dataset: Any = None
        self._fallback_dataset: Any = None

        if train_dataset is not None:
            if hasattr(train_dataset, "train") and hasattr(train_dataset, "validation") and hasattr(train_dataset, "test"):
                train_dataset = train_dataset.train
            self._validate_dataset(train_dataset, "train")
            if not getattr(self._model, "is_fitted", False):
                raise ModelNotFittedError("model must be fitted if train_dataset is provided at initialization.")
            if not self._model_matches_dataset(self._model, train_dataset):
                raise DataValidationError("model dimensions/fingerprints do not match the provided train_dataset.")
            self._train_dataset = train_dataset
            if self._allow_cold_start_fallback and self._fallback_model is not None:
                if getattr(self._fallback_model, "is_fitted", False) and self._model_matches_dataset(self._fallback_model, train_dataset):
                    self._fallback_dataset = train_dataset

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True when the primary model and its exact train dataset are compatible.

        A fallback is deliberately not part of this readiness check.  It is only
        needed for an unknown/cold-start request, and ``_ensure_fallback_ready``
        fits (or refreshes) it against this very dataset before it is used.  This
        keeps a pre-fitted service usable for known users when fallback is enabled
        but was intentionally left to lazy creation.
        """
        return (
            self._train_dataset is not None
            and bool(self._model.is_fitted)
            and self._model_matches_dataset(self._model, self._train_dataset)
        )

    @property
    def model(self) -> BaseRecommender:
        """Return the active primary recommendation model."""
        return self._model

    @property
    def model_name(self) -> str:
        """Return the name of the active primary recommendation model."""
        return self._model.name

    @property
    def fallback_model(self) -> Optional[BaseRecommender]:
        """Return the active fallback recommendation model, or None."""
        return self._fallback_model

    @fallback_model.setter
    def fallback_model(self, fallback: Optional[BaseRecommender]) -> None:
        if fallback is not None:
            self._validate_fallback_model(fallback)
            if self._train_dataset is not None and self._allow_cold_start_fallback:
                self._fit_model(fallback, self._train_dataset, {}, "Fallback model")
                self._fallback_dataset = self._train_dataset
        else:
            # ``None`` means use the contract's lazy Popularity fallback on the
            # next cold-start request, rather than leaving the service unusable.
            self._fallback_dataset = None
        self._fallback_model = fallback

    @property
    def dataset(self) -> Any:
        """Return the train dataset associated with the current fitted service."""
        self._ensure_ready()
        return self._train_dataset

    @property
    def train_dataset(self) -> Any:
        """Alias for dataset property."""
        return self.dataset

    @property
    def evaluator(self) -> Optional[BaseEvaluator]:
        """Return the configured evaluation engine, or None."""
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
        # If enabling while already trained, eagerly fit the fallback so that the
        # first request doesn't incur a training delay.
        if value and not self._allow_cold_start_fallback and self._train_dataset is not None:
            fallback = self._fallback_model or self._new_popularity_model()
            self._validate_fallback_model(fallback)
            self._fit_model(fallback, self._train_dataset, {}, "Fallback model")
            self._fallback_model = fallback
            self._fallback_dataset = self._train_dataset
        self._allow_cold_start_fallback = value

    # ------------------------------------------------------------------
    # Private guards & validators
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        if not self.is_ready:
            raise ModelNotFittedError("Service must be trained before calling operations.")

    @staticmethod
    def _validate_id(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InvalidParameterError(f"{name} must be a non-empty string.")
        return value.strip()

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k <= 0:
            raise InvalidParameterError(f"top_k must be a positive integer, got {top_k!r}.")
        return int(top_k)

    @staticmethod
    def _validate_dataset(dataset: Any, role: str) -> None:
        """Duck-type check against the InteractionDataset Protocol (Contracts v1, Section 4)."""
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
        missing = [attr for attr in required if not hasattr(dataset, attr)]
        if missing:
            raise DataValidationError(
                f"{role} dataset is missing required attributes: {', '.join(missing)}."
            )
        if dataset.split_name not in {"train", "validation", "test"}:
            raise DataValidationError(
                "dataset.split_name must be 'train', 'validation' or 'test'."
            )
        for name in ("num_interactions", "num_users", "num_items"):
            value = getattr(dataset, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise DataValidationError(f"{role}.{name} must be a non-negative integer.")
        try:
            interaction_count = len(dataset.interactions)
        except TypeError as exc:
            raise DataValidationError(f"{role}.interactions must be a sequence.") from exc
        if interaction_count != dataset.num_interactions:
            raise DataValidationError(
                f"{role}.num_interactions must match the interactions sequence length."
            )
        for mapper_name in ("user_mapper", "item_mapper"):
            mapper = getattr(dataset, mapper_name)
            mapper_size = getattr(mapper, "size", None)
            expected_size = dataset.num_users if mapper_name == "user_mapper" else dataset.num_items
            if (
                isinstance(mapper_size, bool)
                or not isinstance(mapper_size, Integral)
                or mapper_size < 0
                or mapper_size != expected_size
                or not callable(getattr(mapper, "contains_raw", None))
                or not callable(getattr(mapper, "to_idx", None))
                or not callable(getattr(mapper, "to_raw", None))
            ):
                raise DataValidationError(f"{role}.{mapper_name} does not satisfy IdMapper.")
        for name in ("mapping_fingerprint", "fingerprint", "review_fingerprint"):
            if not isinstance(getattr(dataset, name), str) or not getattr(dataset, name).strip():
                raise DataValidationError(f"{role}.{name} must be a non-empty string.")
        for name in (
            "get_game_by_raw_id",
            "get_user_seen_items",
            "get_user_positive_items",
            "iter_reviews",
        ):
            if not callable(getattr(dataset, name)):
                raise DataValidationError(f"{role}.{name} must be callable.")

    @staticmethod
    def _model_matches_dataset(model: BaseRecommender, dataset: Any) -> bool:
        return (
            bool(getattr(model, "is_fitted", False))
            and getattr(model, "num_users", None) == dataset.num_users
            and getattr(model, "num_items", None) == dataset.num_items
            and getattr(model, "mapping_fingerprint", None) == dataset.mapping_fingerprint
            and getattr(model, "train_fingerprint", None) == dataset.fingerprint
        )

    @classmethod
    def _fit_model(cls, model: BaseRecommender, dataset: Any, fit_kwargs: dict, role: str) -> None:
        """Fit a model and verify the complete v1 provenance handshake."""
        try:
            model.fit(dataset, **fit_kwargs)
            if not cls._model_matches_dataset(model, dataset):
                raise TrainingError(
                    f"{role} did not finish fitted with the train dimensions/fingerprints."
                )
        except RecSysError:
            raise
        except Exception as exc:
            raise TrainingError(f"{role} training failed: {exc}") from exc

    @staticmethod
    def _validate_fallback_model(model: BaseRecommender) -> None:
        """Fallback must be a BaseRecommender that also exposes score_global().

        Section 7 explicitly requires: 'BPR/TextBPR không được cấu hình làm fallback toàn cục.'
        """
        if not isinstance(model, BaseRecommender):
            raise InvalidParameterError("fallback_model must be a BaseRecommender instance.")
        model_name = getattr(model, "name", "").lower()
        class_name = model.__class__.__name__.lower()
        if "bpr" in model_name or "bpr" in class_name:
            raise ConfigurationError(
                "BPR / TextBPR models cannot be used as global fallback models."
            )
        if not callable(getattr(model, "score_global", None)):
            raise InvalidParameterError("fallback_model must provide score_global(item_ids).")

    # ------------------------------------------------------------------
    # Private ID helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mapper_contains(mapper: Any, raw_id: str) -> bool:
        return bool(mapper.contains_raw(raw_id))

    def _user_idx(self, raw_user_id: str) -> int:
        """Return the internal user index; raise UnknownUserError if absent from mapping."""
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
        """All items the user has interacted with (positive AND negative)."""
        return frozenset(self._train_dataset.get_user_seen_items(user_idx))

    def _positive_items(self, user_idx: int) -> frozenset[int]:
        """Items the user has rated positively (rating == 1.0)."""
        return frozenset(self._train_dataset.get_user_positive_items(user_idx))

    # ------------------------------------------------------------------
    # Private filter & presentation helpers
    # ------------------------------------------------------------------

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
                raise InvalidParameterError(
                    "genre_filter must be a string or iterable of strings."
                ) from error
        if not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise InvalidParameterError("genre_filter must contain non-empty strings.")
        return {v.strip().casefold() for v in values}

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
        game: GameItem,
        genres: Optional[Set[str]],
        max_price: Optional[float],
    ) -> bool:
        """Return True when the game satisfies both the genre and price constraints.

        Contract rule: a game with unknown price (None) is excluded when
        max_price is set (Section 3, Contracts v1).
        """
        if genres is not None:
            raw_genres = getattr(game, "genres", ()) or ()
            if isinstance(raw_genres, str):
                raw_genres = (raw_genres,)
            game_genres = {str(g).strip().casefold() for g in raw_genres if isinstance(g, str) and g.strip()}
            if not genres.intersection(game_genres):
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

    def _to_result(
        self,
        scored: ScoredItem,
        rank: int,
        is_fallback: bool,
    ) -> RecommendationResult:
        """Convert an internal ScoredItem to the public RecommendationResult DTO."""
        raw_game_id = self._raw_item_id(scored.item_idx)
        game: GameItem = self._train_dataset.get_game_by_raw_id(raw_game_id)
        raw_genres = getattr(game, "genres", ()) or ()
        if isinstance(raw_genres, str):
            raw_genres = (raw_genres,)
        genres = tuple(str(g).strip() for g in raw_genres if isinstance(g, str) and g.strip())
        price = getattr(game, "price", None)
        if price is not None:
            price = float(price)
        model_name = (
            self._fallback_model.name
            if (is_fallback and self._fallback_model is not None and getattr(self._fallback_model, "name", None))
            else self.model_name
        )
        return RecommendationResult(
            rank=rank,
            game_id=str(getattr(game, "item_id", raw_game_id)).strip(),
            title=str(getattr(game, "title", "") or "").strip(),
            genres=genres,
            predicted_score=float(scored.score),
            price=price,
            developer=str(getattr(game, "developer", "Unknown") or "Unknown").strip() or "Unknown",
            model_name=model_name,
            is_fallback=is_fallback,
        )

    # ------------------------------------------------------------------
    # Private fallback helpers
    # ------------------------------------------------------------------

    def _new_popularity_model(self) -> BaseRecommender:
        """Lazily import PopularityRecommender from the Model layer."""
        for module_name in ("src.models", "src.models.popularity", "popularity"):
            try:
                module = import_module(module_name)
                model_cls = getattr(module, "PopularityRecommender", None)
                if model_cls is not None:
                    model = model_cls()
                    self._validate_fallback_model(model)
                    return model
            except (ImportError, AttributeError, TypeError, InvalidParameterError, ConfigurationError):
                continue
        raise ConfigurationError(
            "A global popularity fallback model is not available. "
            "Ensure src.models.PopularityRecommender is importable."
        )

    def _ensure_fallback_ready(self) -> None:
        """Fit or refit the fallback model against the current train dataset."""
        if self._fallback_model is None:
            self._fallback_model = self._new_popularity_model()
        if self._fallback_dataset is not self._train_dataset or not self._model_matches_dataset(self._fallback_model, self._train_dataset):
            self._validate_fallback_model(self._fallback_model)
            self._fit_model(self._fallback_model, self._train_dataset, {}, "Fallback model")
            self._fallback_dataset = self._train_dataset

    def _fallback_scores(
        self,
        exclude_seen_for_user: Optional[int],
    ) -> List[ScoredItem]:
        """Return globally scored items, optionally excluding a user's full history.

        Parameters
        ----------
        exclude_seen_for_user:
            When an internal user index is supplied the method removes ALL items the
            user has interacted with (rating 1.0 AND 0.0) from the candidate list,
            matching the Contract v1 rule: 'loại game đã review positive lẫn negative.'
            Pass None only for users who are entirely absent from the mapping.
        """
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

    # ------------------------------------------------------------------
    # Public API — Section 7, Contracts v1
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: Any,
        *,
        validation_score: Any = None,
        item_features: Any = None,
        verbose: bool = False,
    ) -> "RecommendationService":
        """Train the primary model (and fallback if enabled) on *train_dataset*.

        On any failure the Service is left in a not-ready state; callers must
        call train() again before using the Service.

        Parameters
        ----------
        train_dataset:
            An InteractionDataset or DatasetSplits instance.
        validation_score:
            Optional callback candidate -> float (e.g. NDCG@10) to maximize.
        item_features:
            Optional ItemFeatures DTO (required for TextBPR).
        verbose:
            Whether to log training progress.

        Returns self for chaining.
        """
        if train_dataset is None:
            raise EmptyDatasetError("Cannot train on None dataset.")
        if hasattr(train_dataset, "train") and hasattr(train_dataset, "validation") and hasattr(train_dataset, "test"):
            train_dataset = train_dataset.train

        self._validate_dataset(train_dataset, "train")
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

        # Invalidate state before training so a partial failure leaves Service not-ready.
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
                self._validate_fallback_model(fallback)
                self._fit_model(fallback, train_dataset, {}, "Fallback model")
        except Exception:
            # Guarantee that Service stays not-ready on any error.
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
        """Replace the primary model at runtime.

        Parameters
        ----------
        new_model:
            Must satisfy BaseRecommender.
        retrain_immediately:
            When True (default) the new model is immediately fitted on the stored
            train dataset using the supplied *validation_score* and *item_features*.
            TextBPRMatrixFactorization requires *item_features* to be an ItemFeatures DTO.
        validation_score, item_features, verbose:
            Forwarded to new_model.fit() when retrain_immediately is True.
        """
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
        """Return up to *top_k* RecommendationResult items for *raw_user_id*.

        Contract rules applied:
        - exclude_played=True (default) removes ALL seen items (positive AND negative).
        - Users absent from mapping → UnknownUserError (or fallback if enabled).
        - Users in mapping but no positive train profile → ColdStartError (or fallback).
        - Fallback always uses global Popularity; it never impersonates another user.
        - Genre / price filters are applied post-ranking; [] is returned when nothing passes.
        """
        raw_user_id = self._validate_id(raw_user_id, "raw_user_id")
        top_k = self._validate_top_k(top_k)
        if not isinstance(exclude_played, bool):
            raise InvalidParameterError("exclude_played must be a boolean.")
        genres = self._normalise_genres(genre_filter)
        max_price = self._validate_price(max_price)
        self._ensure_ready()

        is_fallback = False
        mapped_user_idx: Optional[int] = None

        # --- Step 1: resolve user identity ---
        try:
            mapped_user_idx = self._user_idx(raw_user_id)
        except UnknownUserError:
            # User is not in the mapping at all.
            if not self._allow_cold_start_fallback:
                raise
            is_fallback = True

        if not is_fallback:
            # User is in the mapping; check for a positive train profile.
            assert mapped_user_idx is not None
            if not self._positive_items(mapped_user_idx):
                # Has interactions (all negative) but no positive history.
                if not self._allow_cold_start_fallback:
                    raise ColdStartError(
                        f"User '{raw_user_id}' has no positive train interactions "
                        "and cold-start fallback is disabled."
                    )
                is_fallback = True
                # Keep mapped_user_idx so fallback can exclude seen items (pos + neg).

        # --- Step 2: gather scored items ---
        if is_fallback:
            self._ensure_fallback_ready()
            # Pass mapped_user_idx (may be None for fully unknown users) so that
            # seen items are excluded for users who exist in the mapping.
            scored_items = self._fallback_scores(
                exclude_seen_for_user=mapped_user_idx if exclude_played else None
            )
        else:
            assert mapped_user_idx is not None
            # Request more items than top_k only when post-filters are active so we
            # can fill up to top_k after filtering without an unnecessary full scan.
            if genres is not None or max_price is not None:
                request_k = max(top_k, self._train_dataset.num_items)
            else:
                request_k = top_k
            scored_items = self._model.recommend(
                mapped_user_idx,
                self._train_dataset,
                top_k=request_k,
                exclude_seen=exclude_played,
            )

        # --- Step 3: apply genre / price filters and build results ---
        results: List[RecommendationResult] = []
        for scored in scored_items:
            raw_game_id = self._raw_item_id(scored.item_idx)
            try:
                game: GameItem = self._train_dataset.get_game_by_raw_id(raw_game_id)
            except (KeyError, IndexError, UnknownItemError):
                continue
            if not isinstance(game, GameItem):
                raise DataValidationError("get_game_by_raw_id must return a GameItem instance.")
            if not self._passes_filter(game, genres, max_price):
                continue
            results.append(self._to_result(scored, len(results) + 1, is_fallback))
            if len(results) == top_k:
                break
        return results

    def get_user_played_games(self, raw_user_id: str) -> List[GameItem]:
        """Return catalog entries for every item the user has seen (pos + neg), sorted by idx."""
        self._ensure_ready()
        user_idx = self._user_idx(self._validate_id(raw_user_id, "raw_user_id"))
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
        """Return the GameItem for the given raw game ID."""
        self._ensure_ready()
        raw_game_id = self._validate_id(raw_game_id, "raw_game_id")
        try:
            game = self._train_dataset.get_game_by_raw_id(raw_game_id)
        except (KeyError, IndexError, UnknownItemError):
            raise UnknownItemError(f"Game '{raw_game_id}' is not in the train catalog.")
        if not isinstance(game, GameItem):
            raise DataValidationError("get_game_by_raw_id must return a GameItem instance.")
        return game

    def evaluate_model(
        self,
        evaluation_dataset: Any,
        *,
        additional_seen_dataset: Any = None,
    ) -> EvaluationResult:
        """Evaluate the fitted primary model and return an EvaluationResult (Contracts v1, Sec 6).

        Parameters
        ----------
        evaluation_dataset:
            Must have split_name 'validation' or 'test', and the same mapping_fingerprint
            and dimensions as the train dataset.
        additional_seen_dataset:
            Optional extra history to exclude during evaluation (must be validation split
            when scoring the test split per Section 6). Forwarded to BaseEvaluator.evaluate().
        """
        self._ensure_ready()
        if self._evaluator is None:
            raise ConfigurationError(
                "No evaluator was supplied to RecommendationService."
            )
        self._validate_dataset(evaluation_dataset, "evaluation")
        if evaluation_dataset.split_name not in {"validation", "test"}:
            raise DataValidationError(
                "evaluation_dataset must have split_name 'validation' or 'test'."
            )
        if evaluation_dataset.mapping_fingerprint != self._train_dataset.mapping_fingerprint:
            raise DataValidationError(
                "evaluation_dataset mapping_fingerprint does not match the train dataset."
            )
        if (
            evaluation_dataset.num_users != self._train_dataset.num_users
            or evaluation_dataset.num_items != self._train_dataset.num_items
        ):
            raise DataValidationError(
                "evaluation_dataset dimensions do not match the train dataset."
            )
        if evaluation_dataset.num_interactions == 0:
            raise EmptyDatasetError("evaluation_dataset contains zero interactions.")

        if additional_seen_dataset is not None:
            if evaluation_dataset.split_name != "test":
                raise DataValidationError(
                    "additional_seen_dataset can only be provided when evaluating on test split."
                )
            self._validate_dataset(additional_seen_dataset, "additional_seen")
            if additional_seen_dataset.split_name != "validation":
                raise DataValidationError(
                    "additional_seen_dataset must have split_name 'validation'."
                )
            if additional_seen_dataset.mapping_fingerprint != self._train_dataset.mapping_fingerprint:
                raise DataValidationError(
                    "additional_seen_dataset mapping_fingerprint does not match train."
                )
            if (
                additional_seen_dataset.num_users != self._train_dataset.num_users
                or additional_seen_dataset.num_items != self._train_dataset.num_items
            ):
                raise DataValidationError(
                    "additional_seen_dataset dimensions do not match the train dataset."
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
    def format_recommendations_table(results: List[RecommendationResult]) -> str:
        """Format a list of RecommendationResult DTOs as a readable ASCII table."""
        if not results:
            return "No matching game recommendations found."

        header = (
            f"{'Rank':<6} | {'Game ID':<10} | {'Title':<24} | "
            f"{'Genres':<22} | {'Price':<8} | {'Developer':<16} | {'Score':>8} | {'Fallback':<8}"
        )
        separator = "-" * len(header)
        rows = [header, separator]
        for r in results:
            genre_str = ", ".join(r.genres)
            price_str = f"${r.price:.2f}" if r.price is not None else "N/A"
            rows.append(
                f"#{r.rank:<5} | {r.game_id:<10} | {r.title[:24]:<24} | "
                f"{genre_str[:22]:<22} | {price_str:<8} | {r.developer[:16]:<16} | "
                f"{r.predicted_score:>8.4f} | {str(r.is_fallback):<8}"
            )
        return "\n".join(rows)


__all__ = ["RecommendationService"]
