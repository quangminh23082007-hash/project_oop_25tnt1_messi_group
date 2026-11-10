from __future__ import annotations

import importlib
import os
from typing import Optional
import numpy as np
import streamlit as st

from src.contracts.dto import GameItem, Interaction
from src.contracts.interfaces import BaseRecommender
from src.service.pretrained import PretrainedDatasetProxy, PretrainedServiceFacade
from src.service.recommendation_service import RecommendationService


class DemoMapper:
    def __init__(self, raw_ids: list[str]) -> None:
        self._raw_ids = list(raw_ids)
        self.size = len(raw_ids)
        self._to_idx = {r: i for i, r in enumerate(raw_ids)}

    def contains_raw(self, raw_id: str) -> bool:
        return raw_id in self._to_idx

    def to_idx(self, raw_id: str) -> int:
        return self._to_idx[raw_id]

    def to_raw(self, index: int) -> str:
        return self._raw_ids[index]


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


def create_demo_service() -> RecommendationService:
    raw_user_ids = ["steam_user_101", "steam_user_102", "steam_user_103", "steam_user_104", "steam_user_cold"]
    raw_game_ids = ["570", "730", "1091500", "292030", "367520", "105600"]

    user_mapper = DemoMapper(raw_user_ids)
    item_mapper = DemoMapper(raw_game_ids)

    catalog = {
        "570": GameItem(item_id="570", title="Dota 2", genres=("Action", "MOBA", "Free to Play"), price=0.0, developer="Valve"),
        "730": GameItem(item_id="730", title="Counter-Strike 2", genres=("Action", "FPS", "Shooter"), price=0.0, developer="Valve"),
        "1091500": GameItem(item_id="1091500", title="Cyberpunk 2077", genres=("Action", "RPG", "Open World"), price=59.99, developer="CD PROJEKT RED"),
        "292030": GameItem(item_id="292030", title="The Witcher 3: Wild Hunt", genres=("Action", "RPG", "Open World"), price=29.99, developer="CD PROJEKT RED"),
        "367520": GameItem(item_id="367520", title="Hollow Knight", genres=("Metroidvania", "Action", "Indie"), price=14.99, developer="Team Cherry"),
        "105600": GameItem(item_id="105600", title="Terraria", genres=("Sandbox", "Adventure", "Survival"), price=9.99, developer="Re-Logic"),
    }

    user_seen = {
        0: {0, 1},       # user 101: Dota 2, CS2
        1: {1, 2},       # user 102: CS2, Cyberpunk
        2: {3},          # user 103: Witcher 3
        3: {0, 2, 3},    # user 104: Dota, Cyberpunk, Witcher
        4: {0},          # cold-start user (chỉ có negative review)
    }
    user_pos = {
        0: {0},
        1: {1, 2},
        2: {3},
        3: {2, 3},
        4: set(),        # cold-start: không có tương tác tích cực
    }

    interactions = [
        Interaction(
            user_id=user_id,
            item_id=item_id,
            rating=1.0 if item_id in user_pos.get(user_id, set()) else 0.0,
        )
        for user_id, seen_items in user_seen.items()
        for item_id in sorted(seen_items)
    ]

    dataset = PretrainedDatasetProxy(
        user_mapper=user_mapper,
        item_mapper=item_mapper,
        catalog=catalog,
        user_seen_items=user_seen,
        user_positive_items=user_pos,
        mapping_fingerprint="demo-mfp",
        fingerprint="demo-train-fp",
        interactions=interactions,
    )

    model = DemoRecommender()
    fallback_model = DemoPopularityRecommender()

    svc = RecommendationService(
        model=model,
        fallback_model=fallback_model,
        allow_cold_start_fallback=True,
        train_dataset=dataset,
    )
    return svc


@st.cache_resource(show_spinner="Đang khởi tạo hệ thống đề xuất...")
def get_service() -> RecommendationService:
    """Return the configured service, with the demo only as a local fallback.

    Set ``RECSYS_SERVICE_FACTORY=package.module:function`` to connect the real
    Data, Model and Evaluation layers without changing the UI.
    """
    factory_path = os.getenv("RECSYS_SERVICE_FACTORY", "").strip()
    if not factory_path:
        for candidate in (
            "src.integration:create_service",
            "src.app_factory:create_service",
            "src.data.integration:create_service",
        ):
            try:
                module_name, function_name = candidate.rsplit(":", 1)
                getattr(importlib.import_module(module_name), function_name)
                factory_path = candidate
                break
            except (ImportError, AttributeError):
                continue
    if factory_path:
        try:
            module_name, function_name = factory_path.rsplit(":", 1)
            factory = getattr(importlib.import_module(module_name), function_name)
            service = factory()
            if not isinstance(service, RecommendationService):
                raise TypeError("Configured factory must return RecommendationService.")
            return service
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot load RECSYS_SERVICE_FACTORY={factory_path!r}: {exc}"
            ) from exc

    return create_demo_service()


def get_sample_user_ids(service: RecommendationService, max_samples: int = 10) -> list[str]:
    try:
        dataset = service.dataset
        user_mapper = getattr(dataset, "user_mapper", None)
        if user_mapper is not None and getattr(user_mapper, "size", 0) > 0:
            limit = min(max_samples, user_mapper.size)
            samples = [str(user_mapper.to_raw(i)) for i in range(limit)]
            if samples:
                return samples
    except Exception:
        pass
    return ["steam_user_101", "steam_user_102", "steam_user_103", "steam_user_104", "steam_user_cold"]


def get_available_genres(service: RecommendationService) -> list[str]:
    genres_set: set[str] = set()
    try:
        dataset = service.dataset
        catalog = getattr(dataset, "_catalog", None) or getattr(dataset, "catalog", None)
        if isinstance(catalog, dict):
            for item in catalog.values():
                if hasattr(item, "genres") and item.genres:
                    for g in item.genres:
                        if isinstance(g, str) and g.strip():
                            genres_set.add(g.strip())
    except Exception:
        pass

    if not genres_set:
        genres_set = {"Action", "RPG", "Open World", "Strategy", "Metroidvania", "Indie", "Sandbox", "Adventure", "Simulation"}

    return ["Tất cả"] + sorted(genres_set)


def get_sample_games(service: RecommendationService, max_samples: int = 15) -> list[tuple[str, str]]:
    games: list[tuple[str, str]] = []
    try:
        dataset = service.dataset
        catalog = getattr(dataset, "_catalog", None) or getattr(dataset, "catalog", None)
        if isinstance(catalog, dict):
            for raw_id, item in catalog.items():
                title = getattr(item, "title", raw_id)
                games.append((str(raw_id), str(title)))
                if len(games) >= max_samples:
                    break
        elif hasattr(dataset, "item_mapper"):
            item_mapper = dataset.item_mapper
            limit = min(max_samples, getattr(item_mapper, "size", 0))
            for i in range(limit):
                raw_id = str(item_mapper.to_raw(i))
                try:
                    game = service.get_game_details(raw_id)
                    games.append((raw_id, game.title))
                except Exception:
                    games.append((raw_id, f"Game {raw_id}"))
    except Exception:
        pass

    if not games:
        games = [
            ("570", "Dota 2"),
            ("730", "Counter-Strike 2"),
            ("1091500", "Cyberpunk 2077"),
            ("292030", "The Witcher 3: Wild Hunt"),
            ("367520", "Hollow Knight"),
            ("105600", "Terraria"),
        ]
    return games
