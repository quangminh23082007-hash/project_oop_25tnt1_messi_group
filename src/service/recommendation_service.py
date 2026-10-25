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
    ) -> None:
        if model is None or not isinstance(model, BaseRecommender):
            raise InvalidParameterError("model must be an instance of BaseRecommender.")
        if evaluator is not None and not isinstance(evaluator, Evaluator):
            raise InvalidParameterError("evaluator must be an Evaluator or None.")

        self._model = model
        self._evaluator = evaluator
        self._allow_cold_start_fallback = allow_cold_start_fallback
        self._dataset: Optional[InteractionDataset] = None
        self._fallback_model: Optional[PopularityRecommender] = None

    @property
    def is_ready(self) -> bool:
        return self._dataset is not None and self._model.is_fitted

    @property
    def model_name(self) -> str:
        return self._model.name

    def _ensure_ready(self) -> None:
        if self._dataset is None or not self._model.is_fitted:
            raise ModelNotFittedError("RecommendationService is not trained yet.")

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

    def _hydrate_results(self, scored_items: List[ScoredItem]) -> List[RecommendationResult]:
        assert self._dataset is not None
        results = []
        for rank, item in enumerate(scored_items, start=1):
            raw_game_id = self._dataset.item_mapper.to_raw(item.item_idx)
            game = self._dataset.get_game_by_raw_id(raw_game_id)
            results.append(
                RecommendationResult(
                    rank=rank,
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

        self._model.fit(dataset)
        if self._allow_cold_start_fallback:
            self._fallback_model = PopularityRecommender()
            self._fallback_model.fit(dataset)

        self._dataset = dataset
        return self

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
                self._fallback_model.fit(self._dataset)
            scored_items = self._fallback_model.recommend(
                user_idx=0,
                dataset=self._dataset,
                top_k=valid_k,
                exclude_seen=False,
            )

        return self._hydrate_results(scored_items)

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
            if not self._dataset.user_mapper.contains_raw(raw_uid):
                continue
            u_idx = self._dataset.user_mapper.to_idx(raw_uid)
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