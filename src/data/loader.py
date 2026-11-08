"""BaseDataLoader: đọc games và interactions thành đúng DTO của contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow.parquet as pq

from src.contracts.dto import InteractionRecord
from src.contracts.exceptions import DataValidationError, EmptyDatasetError
from src.contracts.interfaces import BaseDataLoader
from src.data.entities import GameItem, GameStats, make_game
from src.data.parsing import (
    as_recommended,
    clean_id,
    is_missing,
    label_from_recommended,
    label_from_rating,
    text_or_none,
    to_timestamp,
)

Catalog = Tuple[Dict[str, GameItem], Dict[str, GameStats]]  # (GameItem, GameStats)


def _require_source_path(source_path: str) -> Path:
    if not isinstance(source_path, str) or not source_path.strip():
        raise DataValidationError("source_path must be a non-empty string.")
    return Path(source_path)


def _pick_column(names: Sequence[str], options: Sequence[str]) -> Optional[str]:
    present = set(names)
    for option in options:
        if option in present:
            return option
    return None


def read_game_catalog(path: Path) -> Catalog:
    """
    games.json: {"<appid>": {"name", "price", "genres", "tags", "positive", "negative", ...}}.
    Trả về (GameItem của contract, GameStats phần ngoài contract).
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"Không thể đọc {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise DataValidationError(
            f"games.json phải là dict khóa theo appid, nhưng đang là {type(raw).__name__}."
        )

    games: Dict[str, GameItem] = {}
    stats: Dict[str, GameStats] = {}
    for key, record in raw.items():
        item_id = clean_id(key)
        if not item_id or not isinstance(record, dict):
            continue
        game, game_stats = make_game(
            item_id=item_id,
            title=record.get("name"),
            price=record.get("price"),
            genres=record.get("genres") or (),
            positive=record.get("positive"),
            negative=record.get("negative"),
            tags=record.get("tags") or (),
            developer=record.get("developer", record.get("developers", "Unknown")),
        )
        games[item_id], stats[item_id] = game, game_stats
    if not games:
        raise DataValidationError(f"Không đọc được game nào từ {path}.")
    return games, stats


def read_games_json(path: Path) -> Dict[str, GameItem]:
    """Chỉ phần GameItem của contract (xem read_game_catalog để lấy cả GameStats)."""
    return read_game_catalog(path)[0]


class ParquetDataLoader(BaseDataLoader):
    """
    Đọc catalog và tương tác đã xử lý thành DTO của contract.

    load_games: games.parquet hoặc games.json -> dict[str, GameItem] (dto của contract).
    load_interactions: interactions.parquet, hoặc CSV phẳng có user_id và game_id/item_id.
      Nhãn rating nhị phân: cột rating (đã 0/1) nếu có, không thì Recommended=1.0, Not Recommended=0.0.
      Timestamp (cột timestamp/post_date) về Unix mili giây, thiếu thì 0.
    Sau mỗi lần load_interactions, `load_stats` ghi số dòng đọc và số dòng bị loại theo từng lý do.
    """

    def __init__(self) -> None:
        self.load_stats: Dict[str, int] = {}

    def load_games(self, source_path: str) -> Dict[str, GameItem]:
        return self._load_catalog(source_path)[0]

    def load_game_stats(self, source_path: str) -> Dict[str, GameStats]:
        return self._load_catalog(source_path)[1]

    def load_catalog(self, source_path: str) -> Catalog:
        """(GameItem của contract, GameStats) trong một lần đọc file."""
        return self._load_catalog(source_path)

    def load_interactions(self, source_path: str) -> List[InteractionRecord]:
        path = _require_source_path(source_path)
        if not path.is_file():
            raise DataValidationError(f"Không tìm thấy file tương tác: {source_path}")
        self.load_stats = {
            "rows_read": 0,
            "rows_skipped_missing_id": 0,
            "rows_skipped_bad_label": 0,
        }
        if path.suffix.lower() == ".csv":
            records = self._load_interactions_csv(path)
        else:
            records = self._load_interactions_parquet(path)
        self.load_stats["rows_kept"] = len(records)
        if not records:
            raise EmptyDatasetError(f"Không có tương tác hợp lệ trong {source_path}.")
        return records

    def load_review_texts(
        self,
        reviews_path: str,
        user_col: str = "user_id",
        game_col: str = "game_id",
        review_col: str = "review",
        chunksize: int = 200_000,
    ) -> Tuple[Dict[Tuple[str, str], str], Dict[str, int]]:
        """
        Đọc file review rời (CSV: user, game, review) thành {(user_id, game_id): text}.
        Dùng khi cần nạp text tách riêng khỏi InteractionRecord. Cặp lặp thì giữ dòng đầu tiên.
        Nội dung text giữ nguyên; dòng trống bị bỏ và đếm trong stats.
        """
        path = _require_source_path(reviews_path)
        try:
            header = list(pd.read_csv(path, nrows=0).columns)
        except Exception as exc:
            raise DataValidationError(f"Không thể đọc file CSV review: {exc}") from exc

        if user_col not in header:
            user_col = next((c for c in ("user_id", "user") if c in header), user_col)
        if game_col not in header:
            game_col = next((c for c in ("game_id", "item_id") if c in header), game_col)
        if review_col not in header:
            review_col = next((c for c in ("review", "text") if c in header), review_col)
        wanted = [user_col, game_col, review_col]
        missing = [column for column in wanted if column not in header]
        if missing:
            raise DataValidationError(f"File review thiếu cột {missing}. Các cột hiện có: {header}")

        texts: Dict[Tuple[str, str], str] = {}
        stats = {"rows_read": 0, "rows_blank_text": 0, "rows_duplicate_pair": 0}
        try:
            reader = pd.read_csv(
                path,
                usecols=wanted,
                dtype=str,
                na_filter=False,
                chunksize=chunksize,
                encoding="utf-8",
                encoding_errors="replace",
            )
            for chunk in reader:
                for user, game, review in zip(
                    chunk[user_col].tolist(), chunk[game_col].tolist(), chunk[review_col].tolist()
                ):
                    stats["rows_read"] += 1
                    text = text_or_none(review)
                    if text is None:
                        stats["rows_blank_text"] += 1
                        continue
                    key = (clean_id(user), clean_id(game))
                    if key in texts:
                        stats["rows_duplicate_pair"] += 1
                        continue
                    texts[key] = text
        except DataValidationError:
            raise
        except Exception as exc:
            raise DataValidationError(f"Không thể tải review từ CSV: {exc}") from exc
        return texts, stats

    # ----------------------------------------------------------------- catalog

    def _load_catalog(self, source_path: str) -> Catalog:
        path = _require_source_path(source_path)
        if not path.is_file():
            raise DataValidationError(f"Không tìm thấy file game: {source_path}")
        if path.suffix.lower() == ".json":
            return read_game_catalog(path)
        return self._load_catalog_parquet(path)

    def _load_catalog_parquet(self, path: Path) -> Catalog:
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise DataValidationError(f"Không thể đọc {path}: {exc}") from exc

        id_col = _pick_column(frame.columns, ("item_id", "game_id"))
        if id_col is None:
            raise DataValidationError(
                f"File game thiếu cột item_id/game_id. Các cột hiện có: {list(frame.columns)}"
            )
        title_col = _pick_column(frame.columns, ("title", "name"))
        developer_col = _pick_column(frame.columns, ("developer", "developers"))

        def column(name: Optional[str], default: object) -> List[object]:
            if name is None or name not in frame.columns:
                return [default] * len(frame)
            return frame[name].tolist()

        rows: Iterable[tuple] = zip(
            column(id_col, ""),
            column(title_col, ""),
            column("price", None),
            column("genres", ()),
            column("positive", 0),
            column("negative", 0),
            column("tags", ()),
            column(developer_col, "Unknown"),
        )
        games: Dict[str, GameItem] = {}
        stats: Dict[str, GameStats] = {}
        for item_id, title, price, genres, positive, negative, tags, developer in rows:
            key = clean_id(item_id)
            if not key:
                continue
            games[key], stats[key] = make_game(
                key, title, price, genres, positive, negative, tags, developer
            )
        if not games:
            raise DataValidationError(f"Không đọc được game nào từ {path}.")
        return games, stats

    # ------------------------------------------------------------ interactions

    def _load_interactions_parquet(self, path: Path) -> List[InteractionRecord]:
        try:
            parquet_file = pq.ParquetFile(path)
        except Exception as exc:
            raise DataValidationError(f"Không thể đọc {path}: {exc}") from exc
        names = list(parquet_file.schema.names)
        columns = self._interaction_columns(names)
        records: List[InteractionRecord] = []
        for batch in parquet_file.iter_batches(batch_size=500_000, columns=list(columns.values())):
            payload = {name: batch[name].to_pylist() for name in columns.values()}
            records.extend(self._records_from_columns(payload, columns))
        return records

    def _load_interactions_csv(self, path: Path) -> List[InteractionRecord]:
        try:
            header = list(pd.read_csv(path, nrows=0).columns)
        except Exception as exc:
            raise DataValidationError(f"Không thể đọc {path}: {exc}") from exc
        columns = self._interaction_columns(header)
        records: List[InteractionRecord] = []
        try:
            reader = pd.read_csv(
                path,
                usecols=list(columns.values()),
                dtype=str,
                na_filter=False,
                chunksize=200_000,
                encoding="utf-8",
                encoding_errors="replace",
            )
            for chunk in reader:
                payload = {name: chunk[name].tolist() for name in columns.values()}
                records.extend(self._records_from_columns(payload, columns))
        except DataValidationError:
            raise
        except Exception as exc:
            raise DataValidationError(f"Không thể đọc {path}: {exc}") from exc
        return records

    @staticmethod
    def _interaction_columns(names: Sequence[str]) -> Dict[str, str]:
        user_col = _pick_column(names, ("user_id", "user"))
        item_col = _pick_column(names, ("item_id", "game_id"))
        if user_col is None or item_col is None:
            raise DataValidationError(
                "File tương tác thiếu user_id/user hoặc item_id/game_id. "
                f"Các cột hiện có: {list(names)}"
            )
        rating_col = _pick_column(names, ("rating",))
        recommended_col = _pick_column(names, ("recommended", "recommend"))
        if rating_col is None and recommended_col is None:
            raise DataValidationError(
                "File tương tác cần cột 'rating' (0/1) hoặc 'recommended'/'recommend'. "
                f"Các cột hiện có: {list(names)}"
            )
        mapping = {"user_id": user_col, "item_id": item_col}
        if rating_col:
            mapping["rating"] = rating_col
        if recommended_col:
            mapping["recommended"] = recommended_col
        timestamp_col = _pick_column(names, ("timestamp", "post_date"))
        if timestamp_col:
            mapping["timestamp"] = timestamp_col
        return mapping

    def _records_from_columns(
        self,
        payload: Dict[str, List[object]],
        columns: Dict[str, str],
    ) -> List[InteractionRecord]:
        users = payload[columns["user_id"]]
        items = payload[columns["item_id"]]
        ratings = payload.get(columns["rating"]) if "rating" in columns else None
        recommended = payload.get(columns["recommended"]) if "recommended" in columns else None
        timestamps = payload.get(columns["timestamp"]) if "timestamp" in columns else None
        stats = self.load_stats

        records: List[InteractionRecord] = []
        for index, (user_id, item_id) in enumerate(zip(users, items)):
            stats["rows_read"] += 1
            user_key = clean_id(user_id)
            item_key = clean_id(item_id)
            if not user_key or not item_key:
                stats["rows_skipped_missing_id"] += 1
                continue

            label: Optional[float] = None
            if ratings is not None and not is_missing(ratings[index]) and str(ratings[index]).strip() != "":
                label = label_from_rating(ratings[index])
            if label is None and recommended is not None:
                flag = as_recommended(recommended[index])
                if flag is not None:
                    label = label_from_recommended(flag)
            if label is None:
                stats["rows_skipped_bad_label"] += 1
                continue

            timestamp = to_timestamp(timestamps[index]) if timestamps is not None else 0
            records.append(
                InteractionRecord(
                    user_id=user_key,
                    item_id=item_key,
                    rating=label,
                    timestamp=timestamp,
                )
            )
        return records


__all__ = ["ParquetDataLoader", "read_game_catalog", "read_games_json"]
