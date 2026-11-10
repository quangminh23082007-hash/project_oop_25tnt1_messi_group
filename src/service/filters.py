"""Pure recommendation filter helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Optional

from src.contracts.dto import GameItem
from src.contracts.exceptions import InvalidParameterError


@dataclass(frozen=True)
class FilterCriteria:
    genres: Optional[frozenset[str]] = None
    max_price: Optional[float] = None


def parse_filter_criteria(genre_filter: Any, max_price: Any) -> FilterCriteria:
    if genre_filter is None:
        genres = None
    elif isinstance(genre_filter, str):
        values = [genre_filter]
    else:
        try:
            values = list(genre_filter)
        except TypeError as error:
            raise InvalidParameterError("genre_filter must be a string or iterable of strings.") from error
    if genre_filter is not None:
        if not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise InvalidParameterError("genre_filter must contain non-empty strings.")
        genres = frozenset(v.strip().casefold() for v in values)
    if max_price is None:
        price = None
    else:
        if isinstance(max_price, bool):
            raise InvalidParameterError("max_price must be a finite non-negative number.")
        try:
            price = float(max_price)
        except (TypeError, ValueError) as error:
            raise InvalidParameterError("max_price must be a finite non-negative number.") from error
        if not isfinite(price) or price < 0.0:
            raise InvalidParameterError("max_price must be a finite non-negative number.")
    return FilterCriteria(genres=genres, max_price=price)


def game_passes_filters(game: GameItem, criteria: FilterCriteria) -> bool:
    if criteria.genres is not None:
        raw_genres = getattr(game, "genres", ()) or ()
        if isinstance(raw_genres, str):
            raw_genres = (raw_genres,)
        game_genres = {str(g).strip().casefold() for g in raw_genres if isinstance(g, str) and g.strip()}
        if not criteria.genres.intersection(game_genres):
            return False
    if criteria.max_price is not None:
        price = getattr(game, "price", None)
        if price is None:
            return False
        try:
            price = float(price)
        except (TypeError, ValueError):
            return False
        if not isfinite(price) or price > criteria.max_price:
            return False
    return True
