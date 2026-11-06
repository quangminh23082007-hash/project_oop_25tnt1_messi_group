"""Tạo games.parquet, interactions.parquet và reviews.csv từ games.json + Games_Reviews."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.contracts.exceptions import DataValidationError, EmptyDatasetError
from src.data.entities import GameItem, GameStats
from src.data.loader import read_game_catalog, read_games_json

csv.field_size_limit(min(2**31 - 1, 2_147_483_647))

# <game_id>_<số lượng đánh giá>.csv, ví dụ 730_1523.csv
_REVIEW_FILE_RE = re.compile(r"^(\d+)_(\d+)\.csv$", re.IGNORECASE)

# Cột CSV review mẫu: user, playtime, post_date, helpfulness, review, recommend, early_access_review
_REVIEW_COLUMNS: Dict[str, str] = {
    "user_id": "user",
    "recommended": "recommend",
    "playtime": "playtime",
    "helpfulness": "helpfulness",
    "review": "review",
    "timestamp": "post_date",
}
_REQUIRED_REVIEW_COLUMNS = ("user_id", "recommended")
_RECOMMENDED = "recommended"
_NOT_RECOMMENDED = "not recommended"
_SKIP_DIRS = {"__pycache__", "node_modules", "site-packages", "venv", "env"}


def games_dataframe(games: Iterable[GameItem], stats: Mapping[str, GameStats]) -> pd.DataFrame:
    """Schema parquet giữ game_id/name để đọc được file đã xuất trước đây, thêm developer."""
    games = list(games)
    extra = [stats.get(game.item_id) or GameStats(item_id=game.item_id) for game in games]
    return pd.DataFrame(
        {
            "game_id": [game.item_id for game in games],
            "name": [game.title for game in games],
            "price": pd.Series([game.price for game in games], dtype="float64"),  # None -> NaN
            "genres": [json.dumps(list(game.genres), ensure_ascii=False) for game in games],
            "positive": [item.positive for item in extra],
            "negative": [item.negative for item in extra],
            "tags": [json.dumps(list(item.tags), ensure_ascii=False) for item in extra],
            "developer": [game.developer for game in games],
        }
    )


def _resolve_review_columns(header: List[str]) -> Dict[str, str]:
    present = {name.strip(): name for name in header}
    missing = [
        _REVIEW_COLUMNS[key]
        for key in _REQUIRED_REVIEW_COLUMNS
        if _REVIEW_COLUMNS[key] not in present
    ]
    if missing:
        raise DataValidationError(f"thiếu cột bắt buộc {missing} (các cột hiện có: {header})")
    return {key: present[value] for key, value in _REVIEW_COLUMNS.items() if value in present}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)


def _timestamps(series: pd.Series) -> pd.Series:
    """
    post_date -> epoch giây UTC (int64, >= 0). Số thì giữ nguyên; ngày thì parse; lỗi thì 0.
    Thử parse nhanh theo định dạng suy ra từ dòng đầu, dòng lệch định dạng được parse lại riêng.
    """
    text = series.astype(str).str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    candidates = text.where(numeric.isna() & (text != ""), None)

    parsed = pd.to_datetime(candidates, errors="coerce", utc=True)
    failed = parsed.isna() & candidates.notna()
    if failed.any():
        try:
            retry = pd.to_datetime(candidates[failed], errors="coerce", utc=True, format="mixed")
        except (TypeError, ValueError):
            retry = None
        if retry is not None:
            parsed = parsed.mask(failed, retry)

    seconds = (parsed - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    seconds = seconds.fillna(numeric).fillna(0.0).clip(lower=0.0)
    return seconds.astype("int64")


class _InteractionWriter:
    """Ghi interactions.parquet theo từng lô, không giữ toàn bộ dữ liệu trong RAM."""

    SCHEMA = pa.schema(
        [
            ("user_id", pa.string()),
            ("game_id", pa.string()),
            ("recommended", pa.bool_()),
            ("playtime", pa.float64()),
            ("helpfulness", pa.float64()),
            ("timestamp", pa.int64()),
        ]
    )

    def __init__(self, path: Path, flush_rows: int = 500_000) -> None:
        self._writer = pq.ParquetWriter(str(path), self.SCHEMA)
        self._flush_rows = flush_rows
        self._buf: List[pd.DataFrame] = []
        self._buf_rows = 0
        self.rows = 0
        self.game_ids: Set[str] = set()

    def add(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.game_ids.update(frame["game_id"].unique().tolist())
        self._buf.append(frame)
        self._buf_rows += len(frame)
        self.rows += len(frame)
        if self._buf_rows >= self._flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        frame = pd.concat(self._buf, ignore_index=True)
        self._writer.write_table(pa.Table.from_pandas(frame, schema=self.SCHEMA, preserve_index=False))
        self._buf, self._buf_rows = [], 0

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._writer.close()


def _scan_raw_data(roots: Iterable[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    games_json: Optional[Path] = None
    best_dir: Optional[Path] = None
    best_count = 0

    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if not name.startswith(".") and name not in _SKIP_DIRS]
            if games_json is None:
                for filename in filenames:
                    if filename.lower() == "games.json":
                        games_json = Path(dirpath) / filename
                        break
            count = sum(1 for filename in filenames if _REVIEW_FILE_RE.match(filename))
            if count > best_count:
                best_count, best_dir = count, Path(dirpath)
    return games_json, best_dir


def find_raw_data(raw_dir: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path]]:
    """Tìm (games.json, thư mục review). raw_dir bỏ trống thì quét /kaggle/input rồi thư mục hiện tại."""
    if raw_dir:
        root = Path(raw_dir)
        if not root.exists():
            raise FileNotFoundError(f"Thư mục dữ liệu gốc không tồn tại: {root}")
        roots = [root]
    else:
        roots = [Path("/kaggle/input"), Path.cwd()]
        try:
            roots.append(Path(__file__).resolve().parent)
        except NameError:
            pass
        roots = list(dict.fromkeys(path.resolve() if path.exists() else path for path in roots))
    return _scan_raw_data(roots)


def build_parquet_from_raw(
    output_dir: Path,
    raw_dir: Optional[Path] = None,
    games_json: Optional[Path] = None,
    reviews_dir: Optional[Path] = None,
    min_reviews_per_game: int = 5,
    min_interactions_per_user: int = 1,
    only_games_with_reviews: bool = True,
    include_reviews: bool = True,
    chunksize: int = 200_000,
    verbose: bool = True,
) -> Tuple[Path, Path]:
    """
    Lọc dữ liệu gốc thành games.parquet, interactions.parquet và reviews.csv.

    Nhãn: recommended (bool) giữ nguyên, loader đổi thành rating 1.0/0.0; post_date đổi thành timestamp epoch.
    Bỏ game không có trong games.json, file có ít đánh giá hơn ngưỡng, dòng thiếu user,
    và bản ghi trùng (user, game) trong cùng một file. Ghi file tạm rồi đổi tên khi xong.
    """
    log = print if verbose else (lambda *args, **kwargs: None)

    if games_json is None or reviews_dir is None:
        found_games, found_reviews = find_raw_data(raw_dir)
        games_json = games_json or found_games
        reviews_dir = reviews_dir or found_reviews

    if not games_json or not Path(games_json).is_file():
        raise FileNotFoundError(
            "Không tìm thấy games.json. Hãy truyền raw_dir (thư mục chứa games.json) "
            "hoặc thêm dataset gốc vào Kaggle Input."
        )
    if not reviews_dir or not Path(reviews_dir).is_dir():
        raise FileNotFoundError(
            "Không tìm thấy thư mục Games_Reviews (các file dạng <gameid>_<số lượng>.csv)."
        )
    games_json, reviews_dir = Path(games_json), Path(reviews_dir)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    games_path, interactions_path, reviews_path = (
        out / "games.parquet",
        out / "interactions.parquet",
        out / "reviews.csv",
    )

    def temporary(path: Path) -> Path:
        return path.with_name(path.name + ".tmp")

    stage_path = out / "interactions.stage1.tmp"
    temp_files = [temporary(games_path), temporary(interactions_path), temporary(reviews_path), stage_path]

    log(f"[build] games.json     : {games_json}")
    log(f"[build] Games_Reviews  : {reviews_dir}")
    log(f"[build] Thư mục xuất   : {out}")

    games, game_stats = read_game_catalog(games_json)
    log(f"[build] Đọc {len(games):,} game từ games.json")

    files: List[Tuple[str, Path]] = []
    for path in sorted(reviews_dir.iterdir()):
        match = _REVIEW_FILE_RE.match(path.name)
        if not match or not path.is_file():
            continue
        item_id, declared_count = match.group(1), int(match.group(2))
        if item_id in games and declared_count >= min_reviews_per_game:
            files.append((item_id, path))
    log(f"[build] {len(files):,} file review hợp lệ cần xử lý")
    if not files:
        raise DataValidationError(
            "Không có file review nào khớp <gameid>_<số lượng>.csv và game_id có trong games.json "
            f"(thư mục: {reviews_dir})"
        )

    second_pass = min_interactions_per_user > 1
    user_counts: Dict[str, int] = {}
    skipped = 0
    bad_recommend = 0
    review_count = 0
    column_cache: Dict[Tuple[str, ...], Dict[str, str]] = {}

    pass1_writer: Optional[_InteractionWriter] = None
    final_writer: Optional[_InteractionWriter] = None
    review_file = None
    try:
        pass1_writer = _InteractionWriter(stage_path if second_pass else temporary(interactions_path))
        if include_reviews:
            review_file = open(temporary(reviews_path), "w", encoding="utf-8", newline="")
            review_writer = csv.writer(review_file)
            review_writer.writerow(["user_id", "game_id", "review"])

        for index, (item_id, path) in enumerate(files, 1):
            try:
                header = list(pd.read_csv(path, nrows=0, encoding="utf-8", encoding_errors="replace").columns)
                key = tuple(header)
                if key not in column_cache:
                    column_cache[key] = _resolve_review_columns(header)
                columns = column_cache[key]
                usecols = list(dict.fromkeys(columns.values()))
                seen_users: Set[str] = set()

                reader = pd.read_csv(
                    path,
                    usecols=usecols,
                    dtype=str,
                    na_filter=False,
                    chunksize=chunksize,
                    encoding="utf-8",
                    encoding_errors="replace",
                )
                for chunk in reader:
                    users = chunk[columns["user_id"]].str.strip()
                    recommend_text = chunk[columns["recommended"]].str.strip().str.lower()
                    valid_recommend = recommend_text.isin((_RECOMMENDED, _NOT_RECOMMENDED))
                    bad_recommend += int((~valid_recommend & (users != "")).sum())

                    keep = (users != "") & valid_recommend & ~users.duplicated(keep="first")
                    if seen_users:
                        keep &= ~users.isin(seen_users)
                    chunk, users, recommend_text = chunk[keep], users[keep], recommend_text[keep]
                    if chunk.empty:
                        continue
                    seen_users.update(users.tolist())

                    frame = pd.DataFrame(
                        {
                            "user_id": users,
                            "game_id": item_id,
                            "recommended": recommend_text == _RECOMMENDED,
                            "playtime": _numeric(chunk[columns["playtime"]]) if "playtime" in columns else 0.0,
                            "helpfulness": (
                                _numeric(chunk[columns["helpfulness"]]) if "helpfulness" in columns else 0.0
                            ),
                            "timestamp": (
                                _timestamps(chunk[columns["timestamp"]]) if "timestamp" in columns else 0
                            ),
                        }
                    )
                    pass1_writer.add(frame)

                    if second_pass:
                        for user in users.tolist():
                            user_counts[user] = user_counts.get(user, 0) + 1

                    if include_reviews and "review" in columns:
                        texts = chunk[columns["review"]].str.strip()
                        has_text = texts != ""
                        kept = int(has_text.sum())
                        if kept:
                            review_writer.writerows(
                                zip(users[has_text].tolist(), [item_id] * kept, texts[has_text].tolist())
                            )
                            review_count += kept
            except Exception as exc:
                skipped += 1
                log(f"[build] Bỏ qua {path.name}: {exc}")

            if verbose and (index % 500 == 0 or index == len(files)):
                log(f"[build] {index:,}/{len(files):,} file | {pass1_writer.rows:,} tương tác")

        pass1_writer.close()
        if review_file:
            review_file.close()
            review_file = None

        if pass1_writer.rows == 0:
            raise EmptyDatasetError(
                f"Không đọc được dòng tương tác nào (bỏ qua {skipped} file). "
                "Hãy kiểm tra tên cột CSV."
            )

        kept_games = pass1_writer.game_ids
        total_rows = pass1_writer.rows

        if second_pass:
            valid_users = {user for user, count in user_counts.items() if count >= min_interactions_per_user}
            user_counts.clear()
            log(f"[build] Lọc user (>= {min_interactions_per_user} tương tác): giữ {len(valid_users):,} user")
            final_writer = _InteractionWriter(temporary(interactions_path))
            for batch in pq.ParquetFile(str(stage_path)).iter_batches(batch_size=500_000):
                frame = batch.to_pandas()
                final_writer.add(frame[frame["user_id"].isin(valid_users)])
            final_writer.close()
            if final_writer.rows == 0:
                raise EmptyDatasetError("Không còn tương tác nào sau khi lọc user.")
            kept_games = final_writer.game_ids
            total_rows = final_writer.rows

        selected = [
            game for item_id, game in games.items() if not only_games_with_reviews or item_id in kept_games
        ]
        games_dataframe(selected, game_stats).to_parquet(temporary(games_path), index=False)

        os.replace(temporary(interactions_path), interactions_path)
        if include_reviews:
            os.replace(temporary(reviews_path), reviews_path)
        os.replace(temporary(games_path), games_path)
    except BaseException:
        if pass1_writer is not None:
            try:
                pass1_writer.close()
            except Exception:
                pass
        if final_writer is not None:
            try:
                final_writer.close()
            except Exception:
                pass
        if review_file:
            review_file.close()
        for path in temp_files:
            path.unlink(missing_ok=True)
        raise
    finally:
        stage_path.unlink(missing_ok=True)

    log(
        f"[build] Xong: {len(selected):,} game, {total_rows:,} tương tác, "
        f"{review_count:,} review chữ, bỏ qua {skipped} file lỗi"
    )
    if bad_recommend:
        log(f"[build] Cảnh báo: bỏ {bad_recommend:,} dòng có recommend không phải Recommended/Not Recommended")
    return games_path, interactions_path


def find_dataset_dir() -> Path:
    """Tìm thư mục đã có games.parquet và interactions.parquet."""
    candidates = [
        Path("/kaggle/input/datasets/synnix/database"),
        Path("/kaggle/input/database"),
        Path("/kaggle/input/Database"),
        Path("/kaggle/working/processed"),
    ]
    for candidate in candidates:
        if (candidate / "games.parquet").is_file() and (candidate / "interactions.parquet").is_file():
            return candidate

    for base in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if not base.exists():
            continue
        for root, dirnames, filenames in os.walk(base):
            dirnames[:] = [name for name in dirnames if not name.startswith(".") and name not in _SKIP_DIRS]
            if "games.parquet" in filenames and "interactions.parquet" in filenames:
                return Path(root)
    return Path.cwd()


def default_output_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/processed")
    return Path.cwd() / "processed"


def pick_output_dir(data_dir: Optional[Path], output_dir: Optional[Path]) -> Path:
    if output_dir:
        return Path(output_dir)
    if data_dir and not str(data_dir).replace("\\", "/").startswith("/kaggle/input"):
        return Path(data_dir)
    return default_output_dir()


__all__ = [
    "build_parquet_from_raw",
    "default_output_dir",
    "find_dataset_dir",
    "find_raw_data",
    "games_dataframe",
    "pick_output_dir",
    "read_games_json",
]
