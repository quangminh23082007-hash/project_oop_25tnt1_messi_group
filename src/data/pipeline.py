"""
DataPipeline: nguồn dữ liệu -> DatasetSplits theo Contracts v1.

Luồng:
  1. Đọc games và interactions (dòng sai bị loại và được đếm ở loader).
  2. Gắn review text (từ cột text của nguồn, hoặc từ reviews_path nếu nguồn chưa có).
  3. Dedup theo (user, game): giữ dòng hợp lệ đầu tiên theo thứ tự file/dòng; đếm trùng và nhãn mâu thuẫn.
  4. Chia theo từng user trên ID thô: train / validation / test.
  5. FIT MAPPING TỪ TRAIN rồi đóng băng. Validation/test chỉ được ánh xạ bằng mapping này.
  6. ID ngoài mapping: user có positive ngoài mapping bị loại khỏi split đó (ghi nhận);
     dòng negative ngoài mapping bị bỏ và đếm riêng.
  7. Dựng ba InteractionDataset chỉ đọc và report.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.contracts.dto import DatasetSplits
from src.contracts.exceptions import (
    EmptyDatasetError,
    InvalidParameterError,
    UnknownItemError,
    UnknownUserError,
)
from src.data.dataset import Catalog, IdMapper, InteractionDataset
from src.data.loader import ParquetDataLoader

MIN_INTERACTIONS_TO_SPLIT = 3
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitConfig:
    """
    Mặc định đã thống nhất: per-user random, seed 42, 70% / 10% / 20%.
    Số dòng mỗi tập của user có n >= 3 tương tác:
      validation = max(1, floor(val_ratio * n)), test = max(1, floor(test_ratio * n)), train = phần còn lại.
    strategy="temporal" là tùy chọn ngoài v1 (cũ nhất vào train, mới nhất vào test) và cần timestamp thật.
    """

    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    strategy: str = "random"

    def __post_init__(self) -> None:
        for name in ("val_ratio", "test_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < float(value) < 1.0:
                raise InvalidParameterError(f"{name} phải nằm trong (0, 1), nhận {value!r}.")
        if float(self.val_ratio) + float(self.test_ratio) >= 1.0:
            raise InvalidParameterError("val_ratio + test_ratio phải < 1.")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise InvalidParameterError(f"random_state phải là int, nhận {self.random_state!r}.")
        if self.strategy not in ("random", "temporal"):
            raise InvalidParameterError(f"strategy phải là 'random' hoặc 'temporal', nhận {self.strategy!r}.")


def _split_sizes(size: int, config: SplitConfig) -> Tuple[int, int]:
    """(n_val, n_test) cho user có `size` >= 3 tương tác; train luôn còn ít nhất 1."""
    n_val = max(1, int(math.floor(size * float(config.val_ratio) + 1e-9)))
    n_test = max(1, int(math.floor(size * float(config.test_ratio) + 1e-9)))
    while size - n_val - n_test < 1:
        if n_val > 1:
            n_val -= 1
        else:
            n_test -= 1
    return n_val, n_test


class DataPipeline:
    """Cổng chuẩn bị dữ liệu của Data layer."""

    def __init__(
        self,
        loader: Optional[ParquetDataLoader] = None,
        *,
        price_currency: str = "unspecified",
        user_id_source: str = "display_name",
    ) -> None:
        self._loader = loader if loader is not None else ParquetDataLoader()
        self._price_currency = price_currency
        self._user_id_source = user_id_source

    # ------------------------------------------------------------------ prepare

    def prepare(
        self,
        games_path: str,
        interactions_path: str,
        *,
        reviews_path: Optional[str] = None,
        split_config: Optional[SplitConfig] = None,
    ) -> DatasetSplits:
        config = split_config if split_config is not None else SplitConfig()
        games, game_stats = self._loader.load_catalog(str(games_path))
        records = self._loader.load_interactions(str(interactions_path))
        load_stats = dict(self._loader.load_stats)

        external: Dict[Tuple[str, str], str] = {}
        external_stats: Dict[str, int] = {}
        if reviews_path:
            external, external_stats = self._loader.load_review_texts(str(reviews_path))

        # --- mảng theo mã thô (mã tăng dần = id thô tăng dần, nên thứ tự xác định)
        user_raw = sorted({r.user_id for r in records})
        item_raw = sorted({r.item_id for r in records})
        user_code = {raw: i for i, raw in enumerate(user_raw)}
        item_code = {raw: i for i, raw in enumerate(item_raw)}
        n_rows = len(records)
        u_arr = np.fromiter((user_code[r.user_id] for r in records), dtype=np.int64, count=n_rows)
        i_arr = np.fromiter((item_code[r.item_id] for r in records), dtype=np.int64, count=n_rows)
        rating = np.fromiter((r.rating for r in records), dtype=np.float64, count=n_rows)
        stamp = np.fromiter((r.timestamp for r in records), dtype=np.int64, count=n_rows)
        texts: List[Optional[str]] = [r.text for r in records]
        text_from_file = 0
        if external:
            for pos, record in enumerate(records):
                if texts[pos] is None:
                    found = external.get((record.user_id, record.item_id))
                    if found is not None:
                        texts[pos] = found
                        text_from_file += 1
        rows_with_text_before_dedup = sum(1 for t in texts if t is not None)

        # --- dedup: giữ dòng đầu tiên theo thứ tự đầu vào (argsort ổn định)
        stride = max(len(item_raw), 1)
        keys = u_arr * stride + i_arr
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        is_first = np.ones(n_rows, dtype=bool)
        is_first[1:] = sorted_keys[1:] != sorted_keys[:-1]
        starts = np.flatnonzero(is_first)
        sorted_rating = rating[order]
        conflicting_pairs = int(
            (np.minimum.reduceat(sorted_rating, starts) != np.maximum.reduceat(sorted_rating, starts)).sum()
        )
        keep = order[is_first]  # đã sắp tăng theo (user, item)
        duplicate_rows = n_rows - len(keep)
        u_arr, i_arr, rating, stamp = u_arr[keep], i_arr[keep], rating[keep], stamp[keep]
        texts = [texts[p] for p in keep.tolist()]
        n_pairs = len(keep)
        rows_with_text = sum(1 for t in texts if t is not None)

        # --- chia theo từng user
        assign = self._assign_splits(u_arr, stamp, len(user_raw), config)
        users_below_min = int((np.bincount(u_arr, minlength=len(user_raw)) < MIN_INTERACTIONS_TO_SPLIT).sum())

        # --- mapping fit từ train, đóng băng
        train_rows = assign == 0
        train_users = np.unique(u_arr[train_rows])
        train_items = np.unique(i_arr[train_rows])
        if len(train_users) == 0:
            raise EmptyDatasetError("Không có tương tác train sau khi chia.")
        user_mapper = IdMapper([user_raw[c] for c in train_users], missing_error=UnknownUserError, kind="user")
        item_mapper = IdMapper([item_raw[c] for c in train_items], missing_error=UnknownItemError, kind="item")
        user_map = np.full(len(user_raw), -1, dtype=np.int64)
        user_map[train_users] = np.arange(len(train_users))
        item_map = np.full(len(item_raw), -1, dtype=np.int64)
        item_map[train_items] = np.arange(len(train_items))
        catalog = Catalog(games, game_stats, extra_item_ids=item_raw)
        new_stride = max(item_mapper.size, 1)
        text_array = np.empty(n_pairs, dtype=object)
        text_array[:] = texts

        datasets: Dict[str, InteractionDataset] = {}
        exclusions: Dict[str, Dict[str, int]] = {}
        text_lost_to_mapping: Dict[str, int] = {}
        for code, name in enumerate(SPLIT_NAMES):
            rows = np.flatnonzero(assign == code)
            mu, mi = user_map[u_arr[rows]], item_map[i_arr[rows]]
            unmapped = (mu < 0) | (mi < 0)
            positive = rating[rows] == 1.0
            bad_users = np.unique(u_arr[rows][unmapped & positive])
            in_excluded_user = np.isin(u_arr[rows], bad_users)
            keep_mask = ~(in_excluded_user | unmapped)
            exclusions[name] = {
                "rows_before_filter": int(len(rows)),
                "users_before_filter": int(len(np.unique(u_arr[rows]))),
                "positive_rows_outside_mapping": int((unmapped & positive).sum()),
                "users_excluded_for_positive_outside_mapping": int(len(bad_users)),
                "rows_removed_with_excluded_users": int(in_excluded_user.sum()),
                "negative_rows_outside_mapping_dropped": int((unmapped & ~in_excluded_user).sum()),
                "rows_kept": int(keep_mask.sum()),
            }
            kept_rows = rows[keep_mask]
            ku, ki = mu[keep_mask], mi[keep_mask]
            dataset_texts: Dict[int, str] = {}
            for pos, user_idx, item_idx in zip(kept_rows.tolist(), ku.tolist(), ki.tolist()):
                text = text_array[pos]
                if text is not None:
                    dataset_texts[user_idx * new_stride + item_idx] = text
            text_lost_to_mapping[name] = int(
                sum(1 for pos in rows[~keep_mask].tolist() if text_array[pos] is not None)
            )
            datasets[name] = InteractionDataset.from_arrays(
                split_name=name,
                catalog=catalog,
                user_mapper=user_mapper,
                item_mapper=item_mapper,
                users=ku,
                items=ki,
                ratings=rating[kept_rows],
                timestamps=stamp[kept_rows],
                texts=dataset_texts,
            )

        for name in ("validation", "test"):
            if datasets[name].num_interactions == 0:
                raise EmptyDatasetError(
                    f"Split {name} không còn tương tác nào (cần user có >= {MIN_INTERACTIONS_TO_SPLIT} "
                    "tương tác và item nằm trong mapping train)."
                )

        report = self._build_report(
            config=config,
            load_stats=load_stats,
            external_stats=external_stats,
            text_from_file=text_from_file,
            n_rows_pairs=(n_pairs, duplicate_rows, conflicting_pairs),
            text_counts=(rows_with_text_before_dedup, rows_with_text, text_lost_to_mapping),
            users_total=len(user_raw),
            items_total=len(item_raw),
            users_below_min=users_below_min,
            assign=assign,
            stamp=stamp,
            exclusions=exclusions,
            datasets=datasets,
            catalog=catalog,
            games_in_catalog=len(games),
            item_has_text_train=int(
                len({int(i) for i, t in zip(i_arr[train_rows].tolist(), text_array[train_rows].tolist()) if t is not None})
            ),
            interactions_path=Path(interactions_path),
        )
        return DatasetSplits(
            train=datasets["train"], validation=datasets["validation"], test=datasets["test"], report=report
        )

    # ----------------------------------------------------------------- nội bộ

    @staticmethod
    def _assign_splits(user_codes: np.ndarray, stamps: np.ndarray, num_users: int, config: SplitConfig) -> np.ndarray:
        """0 = train, 1 = validation, 2 = test. user_codes đã sắp tăng theo (user, item)."""
        assign = np.zeros(len(user_codes), dtype=np.int8)
        counts = np.bincount(user_codes, minlength=num_users)
        indptr = np.zeros(num_users + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        if config.strategy == "temporal" and not (stamps > 0).any():
            raise InvalidParameterError(
                "strategy='temporal' cần timestamp thật (> 0). Hãy tạo lại Parquet để có cột timestamp."
            )
        rng = np.random.default_rng(config.random_state)
        for user in range(num_users):
            lo, hi = int(indptr[user]), int(indptr[user + 1])
            size = hi - lo
            if size < MIN_INTERACTIONS_TO_SPLIT:
                continue
            rows = np.arange(lo, hi, dtype=np.int64)
            if config.strategy == "random":
                rows = rng.permutation(rows)
            else:
                rows = rows[np.argsort(stamps[rows], kind="stable")]
            n_val, n_test = _split_sizes(size, config)
            train_end, val_end = size - n_val - n_test, size - n_test
            assign[rows[train_end:val_end]] = 1
            assign[rows[val_end:]] = 2
        return assign

    def _build_report(
        self,
        *,
        config: SplitConfig,
        load_stats: Dict[str, int],
        external_stats: Dict[str, int],
        text_from_file: int,
        n_rows_pairs: Tuple[int, int, int],
        text_counts: Tuple[int, int, Dict[str, int]],
        users_total: int,
        items_total: int,
        users_below_min: int,
        assign: np.ndarray,
        stamp: np.ndarray,
        exclusions: Dict[str, Dict[str, int]],
        datasets: Dict[str, InteractionDataset],
        catalog: Catalog,
        games_in_catalog: int,
        item_has_text_train: int,
        interactions_path: Path,
    ) -> Dict[str, Any]:
        n_pairs, duplicate_rows, conflicting_pairs = n_rows_pairs
        text_before_dedup, text_after_dedup, lost_to_mapping = text_counts
        train = datasets["train"]
        sidecar = interactions_path.with_name("ingest_report.json")
        ingest_report: Optional[Dict[str, Any]] = None
        if sidecar.is_file():
            try:
                ingest_report = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                ingest_report = None

        splits_info = {
            name: {
                "interactions": ds.num_interactions,
                "positive": ds.num_positive,
                "negative": ds.num_negative,
                "users_with_interactions": ds.num_active_users,
                "reviews_with_text": ds.num_reviews,
                "fingerprint": ds.fingerprint,
                "review_fingerprint": ds.review_fingerprint,
            }
            for name, ds in datasets.items()
        }
        text_available = text_after_dedup > 0
        return {
            "protocol": {
                "name": "per_user_random" if config.strategy == "random" else "per_user_temporal_extension",
                "evaluation_kind": (
                    "held-out preference evaluation" if config.strategy == "random" else "temporal (ngoài v1)"
                ),
                "random_state": config.random_state,
                "val_ratio": float(config.val_ratio),
                "test_ratio": float(config.test_ratio),
                "min_interactions_to_split": MIN_INTERACTIONS_TO_SPLIT,
                "split_size_rule": "val=max(1,floor(val_ratio*n)), test=max(1,floor(test_ratio*n)), train=rest",
            },
            "rows": {
                "read_from_source": load_stats.get("rows_read", 0),
                "skipped_missing_id": load_stats.get("rows_skipped_missing_id", 0),
                "skipped_bad_label": load_stats.get("rows_skipped_bad_label", 0),
                "kept_after_cleaning": load_stats.get("rows_kept", 0),
                "duplicate_rows_dropped": duplicate_rows,
                "duplicate_pairs_with_conflicting_label": conflicting_pairs,
                "unique_user_item_pairs": n_pairs,
                "ingest_report": ingest_report,
            },
            "entities_before_split": {
                "users": users_total,
                "items": items_total,
                "users_below_min_interactions_all_in_train": users_below_min,
            },
            "mapping": {
                "fitted_from": "train",
                "num_users": train.num_users,
                "num_items": train.num_items,
                "mapping_fingerprint": train.mapping_fingerprint,
                "games_in_catalog": games_in_catalog,
                "catalog_games_outside_mapping": len(set(catalog.games) - set(train.item_mapper.raw_ids)),
                "items_without_catalog_entry": len(catalog.synthetic_ids),
            },
            "splits": splits_info,
            "split_rows_before_mapping_filter": {
                name: int((assign == code).sum()) for code, name in enumerate(SPLIT_NAMES)
            },
            "exclusions_by_mapping": exclusions,
            "text": {
                "available": text_available,
                "warning": None if text_available else (
                    "Nguồn không có review text: chưa đáp ứng bộ bàn giao để chạy TextBPR."
                ),
                "text_column_in_source": bool(load_stats.get("text_column_present", 0)),
                "rows_with_text_in_source": load_stats.get("rows_with_text", 0),
                "rows_with_text_from_reviews_path": text_from_file,
                "reviews_path_stats": external_stats or None,
                "pairs_with_text_after_dedup": text_after_dedup,
                "pairs_missing_or_blank_text": n_pairs - text_after_dedup,
                "text_dropped_by_dedup": text_before_dedup - text_after_dedup,
                "text_dropped_by_mapping_or_exclusion": lost_to_mapping,
                "reviews_per_split": {name: ds.num_reviews for name, ds in datasets.items()},
                "train_items_with_review": item_has_text_train,
            },
            "timestamp": {
                "unit": "milliseconds_utc",
                "zero_ratio": float((stamp == 0).mean()) if len(stamp) else 1.0,
            },
            "catalog": {"price_currency": self._price_currency},
            "user_id_source": {
                "kind": self._user_id_source,
                "note": (
                    "Nguồn chỉ có tên user, không có Steam ID: các tài khoản trùng tên bị gộp thành một user. "
                    "Đây là giới hạn đã được nhóm chấp nhận."
                ),
            },
            "fingerprints": {
                "mapping": train.mapping_fingerprint,
                **{name: ds.fingerprint for name, ds in datasets.items()},
                **{f"{name}_review": ds.review_fingerprint for name, ds in datasets.items()},
            },
            "artifacts": None,
        }

    # ------------------------------------------------------------------ export

    def export_artifacts(self, splits: DatasetSplits, output_dir: Path) -> Dict[str, Any]:
        """
        Ghi bộ bàn giao review text của contract vào output_dir:
          reviews.csv (user_id,item_id,rating,timestamp,text,split; ID gốc; một dòng mỗi interaction),
          train_reviews.jsonl / validation_reviews.jsonl / test_reviews.jsonl (ID nội bộ, chỉ text không rỗng),
          mapping.json (hai chiều, fingerprints), data_report.json (report kèm checksum từng artifact).
        Trả về manifest {tên file: {path, sha256, records}} và gắn vào splits.report["artifacts"].
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest: Dict[str, Any] = {}

        def finish(name: str, temp: Path, records: int) -> None:
            final = out / name
            os.replace(temp, final)
            digest = hashlib.sha256()
            with open(final, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            manifest[name] = {"path": str(final), "sha256": digest.hexdigest(), "records": records}

        datasets = {"train": splits.train, "validation": splits.validation, "test": splits.test}

        temp = out / "reviews.csv.tmp"
        rows = 0
        with open(temp, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["user_id", "item_id", "rating", "timestamp", "text", "split"])
            for name, ds in datasets.items():
                for user, item, rating, stamp, text in ds.iter_raw_rows():
                    writer.writerow([user, item, int(rating), stamp, text or "", name])
                    rows += 1
        finish("reviews.csv", temp, rows)

        for name, ds in datasets.items():
            temp = out / f"{name}_reviews.jsonl.tmp"
            count = 0
            with open(temp, "w", encoding="utf-8", newline="\n") as handle:
                for review in ds.iter_reviews():
                    handle.write(
                        json.dumps(
                            {
                                "user_id": review.user_id,
                                "item_id": review.item_id,
                                "rating": int(review.rating),
                                "timestamp": review.timestamp,
                                "text": review.text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    count += 1
            finish(f"{name}_reviews.jsonl", temp, count)

        temp = out / "mapping.json.tmp"
        temp.write_text(
            json.dumps(
                {
                    "note": "Chỉ số = vị trí trong danh sách. user_raw_ids[i] là ID gốc của user nội bộ i.",
                    "mapping_fingerprint": splits.train.mapping_fingerprint,
                    "user_raw_ids": list(splits.train.user_mapper.raw_ids),
                    "item_raw_ids": list(splits.train.item_mapper.raw_ids),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        finish("mapping.json", temp, splits.train.num_users + splits.train.num_items)

        splits.report["artifacts"] = manifest
        (out / "data_report.json").write_text(
            json.dumps(splits.report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        return manifest


__all__ = ["DataPipeline", "SplitConfig"]
