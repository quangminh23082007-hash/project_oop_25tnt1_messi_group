"""
src/service/recommendation_service.py

"""

import math
from typing import Dict, List, Optional, Set

from src.contracts.dto import EvaluationReport, RecommendationResult, ScoredItem
from src.contracts.exceptions import (
    EmptyDatasetError,
    InvalidParameterError,
    ModelNotFittedError,
    UnknownItemError,
    UnknownUserError,
)
from src.contracts.interfaces import BaseRecommender
from src.data.dataset import InteractionDataset
from src.data.entities import GameItem
from src.evaluation.evaluator import Evaluator
from src.models.popularity import PopularityRecommender


class RecommendationService:
    """
    Facade service coordinating game recommendation training, top-K inference,
    cold-start fallback, catalog lookup, and ranking evaluation.
    """

    def __init__(
        self,
        model: BaseRecommender,
        evaluator: Optional[Evaluator] = None,
        allow_cold_start_fallback: bool = False,
        fallback_model: Optional[BaseRecommender] = None,
    ) -> None:
        """
        Initialize the RecommendationService with injected dependencies.

        Args:
            model: Primary recommendation algorithm inheriting from BaseRecommender.
            evaluator: Optional Evaluator instance for computing ranking metrics.
            allow_cold_start_fallback: If True, unseen users receive popular game
                recommendations instead of raising UnknownUserError.
            fallback_model: Optional custom fallback recommender for unseen users.
                Defaults to PopularityRecommender when allow_cold_start_fallback is True.

        Raises:
            InvalidParameterError: If model is not a valid BaseRecommender instance.
        """
        if model is None or not isinstance(model, BaseRecommender):
            raise InvalidParameterError("model must be a valid instance of BaseRecommender.")
        if evaluator is not None and not isinstance(evaluator, Evaluator):
            raise InvalidParameterError("evaluator must be an instance of Evaluator or None.")
        if not isinstance(allow_cold_start_fallback, bool):
            raise InvalidParameterError("allow_cold_start_fallback must be a boolean value.")
        if fallback_model is not None and not isinstance(fallback_model, BaseRecommender):
            raise InvalidParameterError("fallback_model must be a BaseRecommender instance or None.")

        self._model: BaseRecommender = model
        self._evaluator: Optional[Evaluator] = evaluator
        self._allow_cold_start_fallback: bool = allow_cold_start_fallback
        self._fallback_model: Optional[BaseRecommender] = fallback_model
        self._fallback_dataset: Optional[InteractionDataset] = None
        self._dataset: Optional[InteractionDataset] = None

    # =========================================================================
    # ENCAPSULATED PROPERTIES (GETTERS / SETTERS)
    # =========================================================================

    @property
    def is_ready(self) -> bool:
        """Return True if the service has a dataset loaded and a fitted primary model."""
        return self._dataset is not None and self._model.is_fitted

    @property
    def model_name(self) -> str:
        """Return the name of the currently active recommendation model."""
        return self._model.name

    @property
    def allow_cold_start_fallback(self) -> bool:
        """Return whether cold-start fallback is enabled for unknown users."""
        return self._allow_cold_start_fallback

    @allow_cold_start_fallback.setter
    def allow_cold_start_fallback(self, value: bool) -> None:
        """Enable or disable cold-start fallback at runtime, fitting fallback if needed."""
        if not isinstance(value, bool):
            raise InvalidParameterError("allow_cold_start_fallback must be a boolean.")
        if value and self._dataset is not None:
            fallback_model = self._fallback_model or PopularityRecommender()
            if not fallback_model.is_fitted or self._fallback_dataset is not self._dataset:
                fallback_model.fit(self._dataset)
            self._fallback_model = fallback_model
            self._fallback_dataset = self._dataset
        self._allow_cold_start_fallback = value

    @property
    def dataset(self) -> InteractionDataset:
        """Return the active InteractionDataset or raise ModelNotFittedError if untrained."""
        self._ensure_ready()
        assert self._dataset is not None
        return self._dataset

    # =========================================================================
    # INTERNAL GUARD & HELPER METHODS (ENCAPSULATION)
    # =========================================================================

    def _ensure_ready(self) -> None:
        """Verify that train() has been executed and the primary model is fitted."""
        if self._dataset is None or not self._model.is_fitted:
            raise ModelNotFittedError(
                f"RecommendationService (model='{self._model.name}') is not trained yet. "
                "Call service.train(dataset) before requesting recommendations or evaluation."
            )

    @staticmethod
    def _validate_non_empty_id(raw_id: str, param_name: str = "raw_user_id") -> str:
        """Validate that an external ID is a non-empty string."""
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise InvalidParameterError(
                f"{param_name} must be a non-empty string, got {raw_id!r}."
            )
        return raw_id.strip()

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        """Validate that top_k is a strictly positive integer."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise InvalidParameterError(
                f"top_k must be a positive integer (> 0), got {top_k!r}."
            )
        return top_k

    def _hydrate_results(
        self,
        scored_items: List[ScoredItem],
        top_k: int,
        genre_filter: Optional[str] = None,
        max_price: Optional[float] = None,
    ) -> List[RecommendationResult]:
        """
        Convert internal ScoredItem objects (item_idx, score) into user-facing
        RecommendationResult DTOs, applying optional genre and price filters.
        Hides internal item indices from external clients.
        """
        assert self._dataset is not None
        normalized_genre = genre_filter.strip().lower() if genre_filter else None

        results: List[RecommendationResult] = []
        for scored in scored_items:
            raw_game_id = self._dataset.item_mapper.to_raw(scored.item_idx)
            game = self._dataset.get_game_by_raw_id(raw_game_id)

            # Optional filter 1: Filter by game genre (case-insensitive)
            if normalized_genre is not None:
                game_genres_lower = [g.strip().lower() for g in game.genres]
                if normalized_genre not in game_genres_lower:
                    continue

            # Optional filter 2: Filter by maximum game price
            if max_price is not None and game.price > max_price:
                continue

            results.append(
                RecommendationResult(
                    rank=len(results) + 1,
                    game_id=game.item_id,
                    title=game.title,
                    genres=game.genres,
                    predicted_score=round(float(scored.score), 4),
                    developer=game.developer,
                )
            )
            if len(results) >= top_k:
                break

        return results

    # =========================================================================
    # CORE PUBLIC API: TRAINING & MODEL MANAGEMENT
    # =========================================================================

    def train(self, dataset: InteractionDataset) -> "RecommendationService":
        """
        Train the primary recommendation model (and cold-start fallback if enabled)
        on the provided InteractionDataset.

        Args:
            dataset: Non-empty InteractionDataset instance.

        Returns:
            Self (RecommendationService) to support method chaining.

        Raises:
            EmptyDatasetError: If dataset is None or contains 0 interactions.
        """
        if dataset is None or not isinstance(dataset, InteractionDataset):
            raise EmptyDatasetError(
                "dataset must be a valid, non-empty InteractionDataset instance."
            )
        if dataset.num_interactions == 0:
            raise EmptyDatasetError("Cannot train service on an empty InteractionDataset.")

        # Fit all components before committing the dataset reference, so a failed
        # training operation does not expose a partly-updated service state.
        self._model.fit(dataset)

        # Build the new fallback model only when cold-start is enabled.
        # Always start fresh (None) so that a previously fitted fallback on an
        # old dataset is never silently reused when fallback is disabled or when
        # the dataset changes.
        new_fallback: Optional[BaseRecommender] = None
        if self._allow_cold_start_fallback:
            new_fallback = self._fallback_model or PopularityRecommender()
            new_fallback.fit(dataset)

        # Commit state atomically — only after all fit() calls succeeded.
        self._dataset = dataset
        self._fallback_model = new_fallback
        self._fallback_dataset = dataset if new_fallback is not None else None
        return self

    def swap_model(
        self, new_model: BaseRecommender, retrain_immediately: bool = True
    ) -> None:
        """
        Hot-swap the underlying recommendation algorithm (Strategy Pattern).

        Args:
            new_model: A new BaseRecommender instance.
            retrain_immediately: If True and a dataset is already loaded, automatically
                calls fit(self._dataset) on the new model.
        """
        if new_model is None or not isinstance(new_model, BaseRecommender):
            raise InvalidParameterError(
                "new_model must be a valid instance of BaseRecommender."
            )
        if retrain_immediately and self._dataset is not None:
            new_model.fit(self._dataset)
        self._model = new_model

    # =========================================================================
    # CORE PUBLIC API: TOP-K GAME RECOMMENDATIONS & CATALOG QUERIES
    # =========================================================================

    def recommend_games(
        self,
        raw_user_id: str,
        top_k: int = 10,
        exclude_played: bool = True,
        genre_filter: Optional[str] = None,
        max_price: Optional[float] = None,
    ) -> List[RecommendationResult]:
        """
        Generate top-K game recommendations for a user by their external raw_user_id.

        Args:
            raw_user_id: External user identifier string (e.g., 'u1', 'steam_user_42').
            top_k: Maximum number of games to return (must be > 0).
            exclude_played: If True, games already in the user's training history are excluded.
            genre_filter: Optional genre name (e.g., 'RPG', 'Action') to filter results.
            max_price: Optional maximum price ceiling (must be >= 0.0 if specified).

        Returns:
            List of RecommendationResult DTOs ordered by rank (1..K).

        Raises:
            InvalidParameterError: If raw_user_id is empty, top_k <= 0, or max_price < 0.
            ModelNotFittedError: If called before train() has been executed.
            UnknownUserError: If raw_user_id is unseen and allow_cold_start_fallback is False.
        """
        clean_user_id = self._validate_non_empty_id(raw_user_id, "raw_user_id")
        valid_k = self._validate_top_k(top_k)
        if not isinstance(exclude_played, bool):
            raise InvalidParameterError("exclude_played must be a boolean value.")

        if genre_filter is not None and (
            not isinstance(genre_filter, str) or not genre_filter.strip()
        ):
            raise InvalidParameterError("genre_filter must be a non-empty string or None.")
        if max_price is not None and (
            isinstance(max_price, bool)
            or not isinstance(max_price, (int, float))
            or not math.isfinite(float(max_price))
            or float(max_price) < 0.0
        ):
            raise InvalidParameterError(
                f"max_price must be a non-negative number, got {max_price!r}."
            )

        self._ensure_ready()
        assert self._dataset is not None

        # When post-filtering by genre or price, request all candidate items from the model
        # first so we can still fill up to top_k matching games after filtering.
        has_post_filters = genre_filter is not None or max_price is not None
        candidate_pool_size = (
            self._dataset.num_items if has_post_filters else valid_k
        )

        # Case 1: Known user in training dataset -> Personalized recommendation
        if self._dataset.user_mapper.contains_raw(clean_user_id):
            user_idx = self._dataset.user_mapper.to_idx(clean_user_id)
            scored_items = self._model.recommend(
                user_idx=user_idx,
                dataset=self._dataset,
                top_k=candidate_pool_size,
                exclude_seen=exclude_played,
            )
        # Case 2: Unseen / Cold-start user
        else:
            if not self._allow_cold_start_fallback:
                raise UnknownUserError(
                    f"User '{clean_user_id}' does not exist in the training dataset "
                    "and cold-start fallback is disabled.",
                    user_id=clean_user_id,
                )
            if self._fallback_model is None:
                self._fallback_model = PopularityRecommender()
            if (
                not self._fallback_model.is_fitted
                or self._fallback_dataset is not self._dataset
            ):
                self._fallback_model.fit(self._dataset)
                self._fallback_dataset = self._dataset

            # Cold-start fallback uses a non-personalized recommender (PopularityRecommender)
            # that ranks items by global popularity, so user_idx=0 is a dummy value —
            # scores are identical regardless of which user_idx is passed.
            scored_items = self._fallback_model.recommend(
                user_idx=0,
                dataset=self._dataset,
                top_k=candidate_pool_size,
                exclude_seen=False,
            )

        return self._hydrate_results(
            scored_items=scored_items,
            top_k=valid_k,
            genre_filter=genre_filter,
            max_price=float(max_price) if max_price is not None else None,
        )

    def get_user_played_games(self, raw_user_id: str) -> List[GameItem]:
        """
        Retrieve the list of GameItem objects that a user has already played in the training set.

        Args:
            raw_user_id: External user identifier string.

        Returns:
            List of GameItem instances sorted by item_id.

        Raises:
            InvalidParameterError: If raw_user_id is empty.
            ModelNotFittedError: If service is not trained yet.
            UnknownUserError: If raw_user_id is not found in the dataset.
        """
        clean_user_id = self._validate_non_empty_id(raw_user_id, "raw_user_id")
        self._ensure_ready()
        assert self._dataset is not None

        user_idx = self._dataset.user_mapper.to_idx(clean_user_id)
        seen_indices = sorted(self._dataset.get_user_seen_items(user_idx))
        return [
            self._dataset.get_game_by_raw_id(self._dataset.item_mapper.to_raw(i_idx))
            for i_idx in seen_indices
        ]

    def get_game_details(self, raw_game_id: str) -> GameItem:
        """
        Look up metadata for a specific game by its external raw_game_id.

        Raises:
            InvalidParameterError: If raw_game_id is empty.
            ModelNotFittedError: If service has no dataset loaded.
            UnknownItemError: If raw_game_id does not exist in the catalog.
        """
        clean_game_id = self._validate_non_empty_id(raw_game_id, "raw_game_id")
        self._ensure_ready()
        assert self._dataset is not None
        return self._dataset.get_game_by_raw_id(clean_game_id)

    # =========================================================================
    # CORE PUBLIC API: EVALUATION & FORMATTING
    # =========================================================================

    def evaluate_model(
        self, test_ground_truth: Dict[str, Set[str]]
    ) -> EvaluationReport:
        """
        Evaluate the active model against held-out test ground truth using the injected Evaluator.

        The Facade translates external raw IDs into internal indices so that
        clients never need to interact with integer user_idx / item_idx.

        Args:
            test_ground_truth: Mapping from raw_user_id to set of raw_item_ids
                representing the items each user actually interacted with in the
                held-out test split.

        Returns:
            EvaluationReport containing averaged metric scores.

        Raises:
            ModelNotFittedError: If called before train().
            InvalidParameterError: If no Evaluator was configured on this service.
            EmptyDatasetError: If test_ground_truth is empty or yields no valid users.
        """
        self._ensure_ready()
        assert self._dataset is not None

        if self._evaluator is None:
            raise InvalidParameterError(
                "Cannot call evaluate_model() because no Evaluator was configured on this service."
            )
        if not isinstance(test_ground_truth, dict):
            raise InvalidParameterError("test_ground_truth must be a dictionary of user IDs to item-ID sets.")
        if not test_ground_truth:
            raise EmptyDatasetError("test_ground_truth cannot be empty during evaluation.")

        # Translate raw string IDs → internal integer indices inside the Facade
        internal_gt: Dict[int, Set[int]] = {}
        for raw_uid, raw_iids in test_ground_truth.items():
            clean_uid = self._validate_non_empty_id(raw_uid, "test_ground_truth user_id")
            if not isinstance(raw_iids, (set, frozenset)):
                raise InvalidParameterError(
                    "Each test_ground_truth value must be a set of non-empty item IDs."
                )
            if not self._dataset.user_mapper.contains_raw(clean_uid):
                continue
            u_idx = self._dataset.user_mapper.to_idx(clean_uid)
            i_idxs: Set[int] = set()
            for raw_iid in raw_iids:
                clean_iid = self._validate_non_empty_id(raw_iid, "test_ground_truth item_id")
                if self._dataset.item_mapper.contains_raw(clean_iid):
                    i_idxs.add(self._dataset.item_mapper.to_idx(clean_iid))
            if i_idxs:
                internal_gt[u_idx] = i_idxs

        if not internal_gt:
            raise EmptyDatasetError(
                "No valid (user, items) pairs found in test_ground_truth after ID mapping."
            )

        return self._evaluator.evaluate(
            model=self._model,
            train_dataset=self._dataset,
            test_ground_truth=internal_gt,
        )

    @staticmethod
    def format_recommendations_table(results: List[RecommendationResult]) -> str:
        """Format a list of RecommendationResult DTOs as a readable ASCII table for CLI display."""
        if not results:
            return "No matching game recommendations found."

        header = (
            f"{'Rank':<6} | {'Game ID':<10} | {'Title':<24} | "
            f"{'Genres':<22} | {'Developer':<16} | {'Score':>8}"
        )
        separator = "-" * len(header)
        rows = [header, separator]
        for r in results:
            genre_str = ", ".join(r.genres)
            rows.append(
                f"#{r.rank:<5} | {r.game_id:<10} | {r.title[:24]:<24} | "
                f"{genre_str[:22]:<22} | {r.developer[:16]:<16} | {r.predicted_score:>8.4f}"
            )
        return "\n".join(rows)


__all__ = ["RecommendationService"]
