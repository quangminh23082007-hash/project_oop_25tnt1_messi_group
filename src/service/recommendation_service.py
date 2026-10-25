"""
src/service/recommendation_service.py

"""

from typing import Dict, List, Optional, Set
from src.contracts.dto import EvaluationReport, RecommendationResult, ScoredItem
from src.contracts.exceptions import (
    EmptyDatasetError,
    InvalidParameterError,
    ModelNotFittedError,
    UnknownUserError,
)
from src.contracts.interfaces import BaseRecommender
from src.data.dataset import InteractionDataset
from src.data.entities import GameItem
from src.evaluation.evaluator import Evaluator
from src.models.popularity import PopularityRecommender


class RecommendationService:
    def __init__(
        self,
        model: BaseRecommender,
        evaluator: Optional[Evaluator] = None,
        allow_cold_start_fallback: bool = False,
        fallback_model: Optional[BaseRecommender] = None,
    ) -> None:
        if model is None or not isinstance(model, BaseRecommender):
            raise InvalidParameterError("model must be an instance of BaseRecommender.")
        if evaluator is not None and not isinstance(evaluator, Evaluator):
            raise InvalidParameterError("evaluator must be an Evaluator or None.")
        if fallback_model is not None and not isinstance(fallback_model, BaseRecommender):
            raise InvalidParameterError("fallback_model must be a BaseRecommender instance.")

        self._model = model
        self._evaluator = evaluator
        self._allow_cold_start_fallback = allow_cold_start_fallback
        self._fallback_model = fallback_model
        self._fallback_dataset: Optional[InteractionDataset] = None
        self._dataset: Optional[InteractionDataset] = None

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
            fallback = self._fallback_model or PopularityRecommender()
            if not fallback.is_fitted or self._fallback_dataset is not self._dataset:
                fallback.fit(self._dataset)
            self._fallback_model = fallback
            self._fallback_dataset = self._dataset
        self._allow_cold_start_fallback = value

    @property
    def dataset(self) -> InteractionDataset:
        self._ensure_ready()
        assert self._dataset is not None
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

    def _hydrate_results(self, scored_items: List[ScoredItem], top_k: int) -> List[RecommendationResult]:
        assert self._dataset is not None
        results = []
        for item in scored_items[:top_k]:
            raw_game_id = self._dataset.item_mapper.to_raw(item.item_idx)
            game = self._dataset.get_game_by_raw_id(raw_game_id)
            results.append(
                RecommendationResult(
                    rank=len(results) + 1,
                    game_id=game.item_id,
                    title=game.title,
                    genres=game.genres,
                    predicted_score=round(float(item.score), 4),
                    developer=game.developer,
                )
            )
        return results

    def train(self, dataset: InteractionDataset) -> "RecommendationService":
        if dataset is None or not isinstance(dataset, InteractionDataset):
            raise EmptyDatasetError("Invalid dataset.")
        if dataset.num_interactions == 0:
            raise EmptyDatasetError("Cannot train on empty dataset.")

        # Atomic commit
        self._model.fit(dataset)
        new_fallback: Optional[BaseRecommender] = None
        if self._allow_cold_start_fallback:
            new_fallback = self._fallback_model or PopularityRecommender()
            new_fallback.fit(dataset)

        self._dataset = dataset
        self._fallback_model = new_fallback
        self._fallback_dataset = dataset if new_fallback is not None else None
        return self

    def swap_model(self, new_model: BaseRecommender, retrain_immediately: bool = True) -> None:
        if new_model is None or not isinstance(new_model, BaseRecommender):
            raise InvalidParameterError("new_model must be a BaseRecommender.")
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
        self._ensure_ready()
        assert self._dataset is not None

        if self._dataset.user_mapper.contains_raw(clean_user_id):
            user_idx = self._dataset.user_mapper.to_idx(clean_user_id)
            scored_items = self._model.recommend(
                user_idx=user_idx,
                dataset=self._dataset,
                top_k=valid_k,
                exclude_seen=exclude_played,
            )
        else:
            if not self._allow_cold_start_fallback:
                raise UnknownUserError(f"User '{clean_user_id}' unknown and fallback disabled.")
            if self._fallback_model is None:
                self._fallback_model = PopularityRecommender()
            if not self._fallback_model.is_fitted or self._fallback_dataset is not self._dataset:
                self._fallback_model.fit(self._dataset)
                self._fallback_dataset = self._dataset

            scored_items = self._fallback_model.recommend(
                user_idx=0,
                dataset=self._dataset,
                top_k=valid_k,
                exclude_seen=False,
            )

        return self._hydrate_results(scored_items, valid_k)

    def get_user_played_games(self, raw_user_id: str) -> List[GameItem]:
        clean_user_id = self._validate_non_empty_id(raw_user_id, "raw_user_id")
        self._ensure_ready()
        assert self._dataset is not None
        user_idx = self._dataset.user_mapper.to_idx(clean_user_id)
        seen_indices = sorted(self._dataset.get_user_seen_items(user_idx))
        return [
            self._dataset.get_game_by_raw_id(self._dataset.item_mapper.to_raw(i))
            for i in seen_indices
        ]

    def get_game_details(self, raw_game_id: str) -> GameItem:
        clean_game_id = self._validate_non_empty_id(raw_game_id, "raw_game_id")
        self._ensure_ready()
        assert self._dataset is not None
        return self._dataset.get_game_by_raw_id(clean_game_id)

    def evaluate_model(self, test_ground_truth: Dict[str, Set[str]]) -> EvaluationReport:
        self._ensure_ready()
        assert self._dataset is not None
        if self._evaluator is None:
            raise InvalidParameterError("Evaluator is not set.")
        if not test_ground_truth:
            raise EmptyDatasetError("test_ground_truth is empty.")

        internal_gt: Dict[int, Set[int]] = {}
        for raw_uid, raw_iids in test_ground_truth.items():
            clean_uid = self._validate_non_empty_id(raw_uid, "test_ground_truth user_id")
            if not self._dataset.user_mapper.contains_raw(clean_uid):
                continue
            u_idx = self._dataset.user_mapper.to_idx(clean_uid)
            i_idxs = {
                self._dataset.item_mapper.to_idx(iid)
                for iid in raw_iids
                if self._dataset.item_mapper.contains_raw(iid)
            }
            if i_idxs:
                internal_gt[u_idx] = i_idxs

        if not internal_gt:
            raise EmptyDatasetError("No valid evaluation pairs after mapping.")

        return self._evaluator.evaluate(
            model=self._model,
            train_dataset=self._dataset,
            test_ground_truth=internal_gt,
        )