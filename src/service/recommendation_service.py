"""
src/service/recommendation_service.py

"""

from typing import Dict, List, Optional, Set
from src.contracts.dto import EvaluationReport, RecommendationResult
from src.contracts.exceptions import (
    EmptyDatasetError,
    InvalidParameterError,
    ModelNotFittedError,
    UnknownUserError,
)
from src.contracts.interfaces import BaseRecommender
from src.data.dataset import InteractionDataset
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

        self._model = model
        self._evaluator = evaluator
        self._allow_cold_start_fallback = allow_cold_start_fallback
        self._dataset: Optional[InteractionDataset] = None
        self._fallback_model: Optional[PopularityRecommender] = None

    def train(self, dataset: InteractionDataset) -> None:
        if dataset is None or dataset.num_interactions == 0:
            raise EmptyDatasetError("Cannot train on an empty dataset.")
        
        self._dataset = dataset
        self._model.fit(dataset)
        if self._allow_cold_start_fallback:
            self._fallback_model = PopularityRecommender()
            self._fallback_model.fit(dataset)

    def recommend_games(
        self,
        raw_user_id: str,
        top_k: int = 10,
        exclude_played: bool = True,
    ) -> List[RecommendationResult]:
        if not raw_user_id or not isinstance(raw_user_id, str):
            raise InvalidParameterError("raw_user_id must be a non-empty string.")
        if top_k <= 0:
            raise InvalidParameterError(f"top_k must be > 0, got {top_k}.")
        if self._dataset is None or not self._model.is_fitted:
            raise ModelNotFittedError("Service must be trained before calling recommend.")

        # Check unknown user and fallback
        if not self._dataset.user_mapper.contains_raw(raw_user_id):
            if not self._allow_cold_start_fallback or self._fallback_model is None:
                raise UnknownUserError(f"User '{raw_user_id}' not found and fallback disabled.")
            scored_items = self._fallback_model.recommend(
                user_idx=0,
                dataset=self._dataset,
                top_k=top_k,
                exclude_seen=False,
            )
        else:
            user_idx = self._dataset.user_mapper.to_idx(raw_user_id)
            scored_items = self._model.recommend(
                user_idx=user_idx,
                dataset=self._dataset,
                top_k=top_k,
                exclude_seen=exclude_played,
            )

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

    def evaluate_model(self, test_ground_truth: Dict[int, Set[int]]) -> EvaluationReport:
        if self._dataset is None or not self._model.is_fitted:
            raise ModelNotFittedError("Cannot evaluate untrained service.")
        if self._evaluator is None:
            raise InvalidParameterError("No evaluator configured.")
        return self._evaluator.evaluate(
            model=self._model,
            train_dataset=self._dataset,
            test_ground_truth=test_ground_truth,
        )