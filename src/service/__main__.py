"""Command line entry point for the Service layer.

Usage:
    python -m src.service
"""

from typing import Dict, Set
import numpy as np

from src.contracts.dto import GameItem
from src.contracts.interfaces import BaseRecommender
from src.service.pretrained import PretrainedDatasetProxy, PretrainedServiceFacade, SimpleIdMapper
from src.service.recommendation_service import RecommendationService


def _create_fallback_demo_service() -> RecommendationService:
    class DemoRecommender(BaseRecommender):
        def __init__(self, name: str = "DemoBPRRecommender") -> None:
            super().__init__(name)
            self._is_fitted = True
            self._num_users = 5
            self._num_items = 6
            self._mapping_fingerprint = "demo-mfp"
            self._train_fingerprint = "demo-train-fp"

        def fit(self, train_dataset, *, validation_score=None, item_features=None, verbose=False):
            self._is_fitted = True
            return self

        def score_items(self, user_id: int, item_ids) -> np.ndarray:
            ids = np.asarray(item_ids, dtype=np.int64)
            scores = np.array([4.2, 3.8, 3.5, 2.9, 2.1, 1.5], dtype=float)
            return scores[ids]

        def config(self) -> dict:
            return {"model": "DemoBPR", "factors": 16}

    class DemoPopularityRecommender(BaseRecommender):
        def __init__(self, name: str = "DemoPopularityRecommender") -> None:
            super().__init__(name)
            self._is_fitted = True
            self._num_users = 5
            self._num_items = 6
            self._mapping_fingerprint = "demo-mfp"
            self._train_fingerprint = "demo-train-fp"
            self._counts = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5], dtype=float)

        def fit(self, train_dataset, *, validation_score=None, item_features=None, verbose=False):
            self._is_fitted = True
            return self

        def score_global(self, item_ids) -> np.ndarray:
            ids = np.asarray(item_ids, dtype=np.int64)
            return self._counts[ids]

        def score_items(self, user_id: int, item_ids) -> np.ndarray:
            return self.score_global(item_ids)

        def config(self) -> dict:
            return {"model": "DemoPopularity"}

    raw_user_ids = ["steam_user_101", "steam_user_102", "steam_user_103", "steam_user_104", "steam_user_cold"]
    raw_game_ids = ["570", "730", "1091500", "292030", "367520", "105600"]
    catalog = {
        "570": GameItem(item_id="570", title="Dota 2", genres=("Action", "MOBA", "Free to Play"), price=0.0, developer="Valve"),
        "730": GameItem(item_id="730", title="Counter-Strike 2", genres=("Action", "FPS", "Shooter"), price=0.0, developer="Valve"),
        "1091500": GameItem(item_id="1091500", title="Cyberpunk 2077", genres=("Action", "RPG", "Open World"), price=59.99, developer="CD PROJEKT RED"),
        "292030": GameItem(item_id="292030", title="The Witcher 3: Wild Hunt", genres=("Action", "RPG", "Open World"), price=29.99, developer="CD PROJEKT RED"),
        "367520": GameItem(item_id="367520", title="Hollow Knight", genres=("Metroidvania", "Action", "Indie"), price=14.99, developer="Team Cherry"),
        "105600": GameItem(item_id="105600", title="Terraria", genres=("Sandbox", "Adventure", "Survival"), price=9.99, developer="Re-Logic"),
    }
    user_seen = {0: {0, 1}, 1: {1, 2}, 2: {3}, 3: {0, 2, 3}, 4: {0}}
    user_pos = {0: {0}, 1: {1, 2}, 2: {3}, 3: {2, 3}, 4: set()}

    dataset = PretrainedDatasetProxy(
        user_mapper=SimpleIdMapper(raw_user_ids),
        item_mapper=SimpleIdMapper(raw_game_ids),
        catalog=catalog,
        user_seen_items=user_seen,
        user_positive_items=user_pos,
        mapping_fingerprint="demo-mfp",
        fingerprint="demo-train-fp",
    )

    model = DemoRecommender()
    fallback_model = DemoPopularityRecommender()
    return RecommendationService(
        model=model,
        fallback_model=fallback_model,
        allow_cold_start_fallback=True,
        train_dataset=dataset,
    )


def main() -> None:
    print("=" * 60)
    print("STEAM GAME RECOMMENDATION SERVICE - DEMO EXECUTION")
    print("=" * 60)
    try:
        from src.ui.bootstrap import create_demo_service
        service = create_demo_service()
    except (ImportError, ModuleNotFoundError):
        service = _create_fallback_demo_service()

    print(f"Service loaded successfully. Primary model: {service.model_name}")
    print(f"Fallback model: {getattr(service.fallback_model, 'name', 'None')}")
    print(f"Is ready: {service.is_ready}")
    print("-" * 60)

    user_id = "steam_user_101"
    print(f"Generating recommendations for: {user_id}")
    results = service.recommend_games(user_id, top_k=3)
    print(service.format_recommendations_table(results))

    print("\n" + "=" * 60)
    cold_user = "steam_user_cold"
    print(f"Generating cold-start recommendations for: {cold_user}")
    cold_results = service.recommend_games(cold_user, top_k=3)
    print(service.format_recommendations_table(cold_results))
    print("=" * 60)


if __name__ == "__main__":
    main()
