from __future__ import annotations

from typing import Any


def _call(service: Any, name: str, limit: int = 12) -> list[str]:
    provider = getattr(service, name, None)
    if not callable(provider):
        return []
    try:
        values = provider(limit=limit)
    except TypeError:
        try:
            values = provider()
        except (AttributeError, TypeError, ValueError):
            return []
    except (AttributeError, TypeError, ValueError):
        return []
    values = [str(value) for value in values if str(value).strip()]
    if len(values) <= limit:
        return values
    step = max(1, len(values) // limit)
    return values[::step][:limit]


def user_id_options(service: Any) -> list[str]:
    return _call(service, "get_available_user_ids")


def genre_options(service: Any) -> list[str]:
    return _call(service, "get_available_genres")


def game_id_options(service: Any) -> list[str]:
    return _call(service, "get_available_game_ids")
