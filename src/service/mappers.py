"""Pure DTO assembly helpers."""

from __future__ import annotations

from src.contracts.dto import GameItem, RecommendationResult, ScoredItem


def to_recommendation_result(
    scored: ScoredItem,
    game: GameItem,
    rank: int,
    model_name: str,
    is_fallback: bool,
) -> RecommendationResult:
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
