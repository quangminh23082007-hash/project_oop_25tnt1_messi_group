from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

from src.contracts.dto import GameItem, RecommendationResult


class UIRecommendationService(Protocol):
    """Small service surface consumed by the Streamlit application."""

    @property
    def model_name(self) -> str: ...

    @property
    def is_ready(self) -> bool: ...

    @property
    def allow_cold_start_fallback(self) -> bool: ...

    @allow_cold_start_fallback.setter
    def allow_cold_start_fallback(self, value: bool) -> None: ...

    @property
    def supports_catalog_filters(self) -> bool: ...

    def recommend_games(
        self,
        raw_user_id: str,
        top_k: int = 10,
        exclude_played: bool = True,
        genre_filter: Any = None,
        max_price: Optional[float] = None,
    ) -> Sequence[RecommendationResult]: ...

    def recommend_baseline(
        self,
        raw_user_id: Optional[str] = None,
        top_k: int = 10,
        exclude_played: bool = True,
        genre_filter: Any = None,
        max_price: Optional[float] = None,
    ) -> Sequence[RecommendationResult]: ...

    def get_user_played_games(self, raw_user_id: str) -> Sequence[GameItem]: ...

    def get_game_details(self, raw_game_id: str) -> GameItem: ...

    def get_available_user_ids(self, limit: Optional[int] = None) -> Sequence[str]: ...

    def get_available_game_ids(self) -> Sequence[str]: ...

    def get_available_genres(self) -> Sequence[str]: ...


__all__ = ["UIRecommendationService"]
