"""
Catalog game.

GameItem được định nghĩa duy nhất trong src.contracts.dto và chỉ re-export ở đây.
Thông tin ngoài contract (positive/negative/tags, điểm Wilson) nằm ở GameStats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from src.contracts.dto import GameItem
from src.contracts.exceptions import DataValidationError
from src.data.parsing import clean_id, is_missing, parse_list, to_float, to_price, wilson_score


def _developer_name(value: object) -> str:
    if is_missing(value):
        return "Unknown"
    if isinstance(value, (list, tuple, set, dict)) or hasattr(value, "tolist"):
        items = parse_list(value)
        return items[0] if items else "Unknown"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "unknown"}:
        return "Unknown"
    return text


@dataclass(frozen=True)
class GameStats:
    """Số liệu bổ sung của game, không thuộc contract GameItem."""

    item_id: str
    positive: int = 0
    negative: int = 0
    tags: Tuple[str, ...] = ()

    @property
    def wilson_rating(self) -> float:
        """Điểm chất lượng game trên thang 1.0–5.0, tách khỏi nhãn tương tác của từng user."""
        return round(1.0 + 4.0 * wilson_score(self.positive, self.negative), 2)


def make_game(
    item_id: object,
    title: object = "",
    price: object = None,
    genres: object = (),
    positive: object = 0,
    negative: object = 0,
    tags: object = (),
    developer: object = "Unknown",
) -> Tuple[GameItem, GameStats]:
    """Chuẩn hóa một dòng game thô thành (GameItem của contract, GameStats)."""
    key = clean_id(item_id)
    if not key:
        raise DataValidationError("item_id must be a non-empty string.")
    name = "" if is_missing(title) else str(title).strip()
    game = GameItem(
        item_id=key,
        title=name or key,  # contract: title không được rỗng
        genres=tuple(parse_list(genres)),
        price=to_price(price),
        developer=_developer_name(developer),
    )
    stats = GameStats(
        item_id=key,
        positive=max(0, int(to_float(positive))),
        negative=max(0, int(to_float(negative))),
        tags=tuple(parse_list(tags)),
    )
    return game, stats


__all__ = ["GameItem", "GameStats", "make_game"]
