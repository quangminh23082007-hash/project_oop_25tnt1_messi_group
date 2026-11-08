"""Pure DTO assembly helpers."""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real

from src.contracts.dto import GameItem, RecommendationResult, ScoredItem
from src.contracts.exceptions import DataValidationError


def to_recommendation_result(
    scored: ScoredItem,
    game: GameItem,
    rank: int,
    model_name: str,
    is_fallback: bool,
) -> RecommendationResult:
    if not isinstance(scored, ScoredItem):
        raise DataValidationError("scored must be a ScoredItem instance.")
    if not isinstance(game, GameItem):
        raise DataValidationError("game must be a GameItem instance.")
    if isinstance(scored.item_idx, bool) or not isinstance(scored.item_idx, Integral) or scored.item_idx < 0:
        raise DataValidationError("recommendation item_idx must be a non-negative integer.")
    if isinstance(rank, bool) or not isinstance(rank, Integral) or rank <= 0:
        raise DataValidationError("recommendation rank must be a positive integer.")
    if isinstance(scored.score, bool) or not isinstance(scored.score, Real) or not isfinite(scored.score):
        raise DataValidationError("recommendation score must be finite.")
    if not isinstance(model_name, str) or not model_name.strip():
        raise DataValidationError("recommendation model_name must be a non-empty string.")
    if not isinstance(is_fallback, bool):
        raise DataValidationError("recommendation is_fallback must be boolean.")
    raw_genres = getattr(game, "genres", ()) or ()
    if isinstance(raw_genres, str):
        raw_genres = (raw_genres,)
    genres = tuple(str(g).strip() for g in raw_genres if isinstance(g, str) and g.strip())
    price = getattr(game, "price", None)
    if price is not None:
        price = float(price)
    return RecommendationResult(
        rank=int(rank),
        game_id=str(getattr(game, "item_id", "")).strip(),
        title=str(getattr(game, "title", "") or "").strip(),
        genres=genres,
        predicted_score=float(scored.score),
        price=price,
        developer=str(getattr(game, "developer", "Unknown") or "Unknown").strip() or "Unknown",
        model_name=str(model_name),
        is_fallback=bool(is_fallback),
    )
