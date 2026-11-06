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


_HAS_YEAR = r"\d{4}|[-/ ]\d{2}$"  # có năm 4 chữ số, hoặc năm 2 chữ số ở cuối (20-Jul-15)
_MS_THRESHOLD = 100_000_000_000  # số < 1e11 coi là epoch giây, ngược lại là mili giây


def _timestamps(series: pd.Series) -> pd.Series:
    """
    post_date -> Unix mili giây UTC (int64, >= 0) theo contract.

    Số thì giữ nguyên (nhỏ hơn 1e11 coi là giây và nhân 1000); chuỗi ngày thì parse.
    Ngày không có năm (ví dụ "May 14") hoặc không đọc được thì 0, không đoán năm.
    Parse nhanh theo định dạng suy ra từ dòng đầu, dòng lệch định dạng được parse lại riêng.
    """
    text = series.astype(str).str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    is_date_text = numeric.isna() & (text != "") & text.str.contains(_HAS_YEAR, regex=True)
    candidates = text.where(is_date_text, None)

    parsed = pd.to_datetime(candidates, errors="coerce", utc=True)
    failed = parsed.isna() & candidates.notna()
    if failed.any():
        try:
            retry = pd.to_datetime(candidates[failed], errors="coerce", utc=True, format="mixed")
        except (TypeError, ValueError):
            retry = None
        if retry is not None:
            parsed = parsed.mask(failed, retry)

    from_dates = (parsed - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds() * 1000.0
    from_numbers = numeric.where(numeric >= _MS_THRESHOLD, numeric * 1000.0)
    millis = from_dates.fillna(from_numbers).fillna(0.0).clip(lower=0.0)
    return millis.round().astype("int64")


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
            ("text", pa.string()),
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
    Lọc dữ liệu gốc thành games.parquet, interactions.parquet và ingest_report.json.

    Nhãn: recommended (bool) giữ nguyên, loader đổi thành rating 1.0/0.0.
    post_date đổi thành timestamp Unix mili giây UTC (0 nếu không đáng tin cậy).
    Review text ghi nguyên văn vào cột `text` của interactions.parquet (include_reviews=False thì để trống),
    nên text đi cùng đúng dòng tương tác. Dedup (user, game) do DataPipeline làm (giữ dòng đầu theo thứ
    tự file/dòng, có đếm), ingest giữ nguyên thứ tự file rồi dòng.
    Bỏ game không có trong games.json, file có ít đánh giá hơn ngưỡng, dòng thiếu user hoặc nhãn lạ.
    Ghi file tạm rồi đổi tên khi xong.
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
    games_path, interactions_path = out / "games.parquet", out / "interactions.parquet"
    report_path = out / "ingest_report.json"

    def temporary(path: Path) -> Path:
        return path.with_name(path.name + ".tmp")

    stage_path = out / "interactions.stage1.tmp"
    temp_files = [temporary(games_path), temporary(interactions_path), stage_path]

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
    rows_read = 0
    rows_missing_user = 0
    column_cache: Dict[Tuple[str, ...], Dict[str, str]] = {}

    pass1_writer: Optional[_InteractionWriter] = None
    final_writer: Optional[_InteractionWriter] = None
    try:
        pass1_writer = _InteractionWriter(stage_path if second_pass else temporary(interactions_path))

        for index, (item_id, path) in enumerate(files, 1):
            try:
                header = list(pd.read_csv(path, nrows=0, encoding="utf-8", encoding_errors="replace").columns)
                key = tuple(header)
                if key not in column_cache:
                    column_cache[key] = _resolve_review_columns(header)
                columns = column_cache[key]
                usecols = list(dict.fromkeys(columns.values()))

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
                    rows_read += len(chunk)
                    rows_missing_user += int((users == "").sum())
                    bad_recommend += int((~valid_recommend & (users != "")).sum())

                    keep = (users != "") & valid_recommend
                    chunk, users, recommend_text = chunk[keep], users[keep], recommend_text[keep]
                    if chunk.empty:
                        continue

                    if include_reviews and "review" in columns:
                        raw_texts = chunk[columns["review"]].tolist()
                        # Giữ nguyên nội dung; thiếu hoặc chỉ khoảng trắng thì null.
                        texts: List[Optional[str]] = [t if t.strip() else None for t in raw_texts]
                    else:
                        texts = [None] * len(chunk)
                    review_count += sum(1 for t in texts if t is not None)

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
                            "text": texts,
                        }
                    )
                    pass1_writer.add(frame)

                    if second_pass:
                        for user in users.tolist():
                            user_counts[user] = user_counts.get(user, 0) + 1
            except Exception as exc:
                skipped += 1
                log(f"[build] Bỏ qua {path.name}: {exc}")

            if verbose and (index % 500 == 0 or index == len(files)):
                log(f"[build] {index:,}/{len(files):,} file | {pass1_writer.rows:,} tương tác")

        pass1_writer.close()

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
        os.replace(temporary(games_path), games_path)
        report_path.write_text(
            json.dumps(
                {
                    "rows_read": rows_read,
                    "rows_skipped_missing_user": rows_missing_user,
                    "rows_skipped_bad_recommend": bad_recommend,
                    "rows_written": total_rows,
                    "rows_with_text": review_count,
                    "files_skipped_on_error": skipped,
                    "files_processed": len(files),
                    "min_reviews_per_game": min_reviews_per_game,
                    "min_interactions_per_user": min_interactions_per_user,
                    "timestamp_unit": "milliseconds_utc",
                    "user_id_source": "display_name",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
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
        for path in temp_files:
            path.unlink(missing_ok=True)
        raise
    finally:
        stage_path.unlink(missing_ok=True)

    log(
        f"[build] Xong: {len(selected):,} game, {total_rows:,} tương tác, "
        f"{review_count:,} dòng có review chữ (trước khi lọc theo user), bỏ qua {skipped} file lỗi"
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
