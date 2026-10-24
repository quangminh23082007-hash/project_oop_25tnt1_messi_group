"""
src/service/recommendation_service.py

"""

from typing import Dict, List, Optional, Set
from src.contracts.dto import EvaluationReport, RecommendationResult
from src.contracts.interfaces import BaseRecommender
from src.data.dataset import InteractionDataset
from src.evaluation.evaluator import Evaluator


class RecommendationService:
    def __init__(
        self,
        model: BaseRecommender,
        evaluator: Optional[Evaluator] = None,
        allow_cold_start_fallback: bool = False,
    ) -> None:
        self.model = model
        self.evaluator = evaluator
        self.allow_cold_start_fallback = allow_cold_start_fallback
        self.dataset: Optional[InteractionDataset] = None

    def train(self, dataset: InteractionDataset) -> None:
        self.dataset = dataset
        self.model.fit(dataset)

    def recommend_games(
        self,
        raw_user_id: str,
        top_k: int = 10,
        exclude_played: bool = True,
    ) -> List[RecommendationResult]:
        assert self.dataset is not None
        user_idx = self.dataset.user_mapper.to_idx(raw_user_id)
        
        scored_items = self.model.recommend(
            user_idx=user_idx,
            dataset=self.dataset,
            top_k=top_k,
            exclude_seen=exclude_played,
        )

        results = []
        for rank, item in enumerate(scored_items, start=1):
            raw_game_id = self.dataset.item_mapper.to_raw(item.item_idx)
            game = self.dataset.get_game_by_raw_id(raw_game_id)
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
        assert self.dataset is not None
        assert self.evaluator is not None
        return self.evaluator.evaluate(
            model=self.model,
            train_dataset=self.dataset,
            test_ground_truth=test_ground_truth,
        )