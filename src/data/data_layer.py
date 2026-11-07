"""
Điểm vào của Data layer: nạp Parquet (tự tạo từ dữ liệu gốc nếu chưa có) rồi chuẩn bị
DatasetSplits (train / validation / test) theo Contracts v1.

    python -m src.data.data_layer --data-dir ./data/processed
    python -m src.data.data_layer --raw-dir ./data/raw --output-dir ./data/processed --rebuild \
        --min-reviews-per-game 5 --export-dir ./data/artifacts
    Kaggle: from src.data.data_layer import main; splits = main(...)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parents[2]  # thư mục chứa "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.contracts.dto import DatasetSplits  # noqa: E402
from src.contracts.exceptions import (  # noqa: E402
    DataValidationError,
    EmptyDatasetError,
    UnknownItemError,
    UnknownUserError,
)
from src.data.dataset import InteractionDataset  # noqa: E402
from src.data.entities import GameItem  # noqa: E402
from src.data.ingest import build_parquet_from_raw, find_dataset_dir, pick_output_dir  # noqa: E402
from src.data.loader import ParquetDataLoader  # noqa: E402
from src.data.pipeline import DataPipeline, SplitConfig  # noqa: E402


def main(
    data_dir: Optional[Path] = None,
    reviews_path: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    rebuild: bool = False,
    *,
    split_config: Optional[SplitConfig] = None,
    export_dir: Optional[Path] = None,
    price_currency: str = "unspecified",
    **build_options: object,
) -> DatasetSplits:
    """
    Chuẩn bị dữ liệu và trả về DatasetSplits(train, validation, test, report).

    Chưa có Parquet, hoặc rebuild=True, thì tạo từ dữ liệu gốc trước.
    build_options truyền thẳng cho build_parquet_from_raw (min_reviews_per_game,
    min_interactions_per_user, ...).
    reviews_path là nguồn text tách riêng, chỉ được đọc khi được truyền vào.
    export_dir: nếu có thì ghi bộ artifact review của contract (reviews.csv, *_reviews.jsonl,
    mapping.json, data_report.json) vào đó.
    """
    target_dir = Path(data_dir) if data_dir else find_dataset_dir()
    games_path = target_dir / "games.parquet"
    interactions_path = target_dir / "interactions.parquet"

    if rebuild or not (games_path.is_file() and interactions_path.is_file()):
        out_dir = pick_output_dir(Path(data_dir) if data_dir else None, output_dir)
        if rebuild:
            print("Rebuild: tạo lại Parquet từ dữ liệu gốc...")
        else:
            print(f"Chưa có file Parquet tại {target_dir} -> tạo từ dữ liệu gốc...")
        build_parquet_from_raw(out_dir, raw_dir=raw_dir, **build_options)  # type: ignore[arg-type]
        target_dir = out_dir
        games_path = target_dir / "games.parquet"
        interactions_path = target_dir / "interactions.parquet"

    print(f"Đang nạp dữ liệu từ: {target_dir}")
    if not games_path.is_file() or not interactions_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file parquet tại {target_dir}")

    pipeline = DataPipeline(price_currency=price_currency)
    splits = pipeline.prepare(
        str(games_path),
        str(interactions_path),
        reviews_path=str(reviews_path) if reviews_path is not None else None,
        split_config=split_config,
    )
    if export_dir:
        manifest = pipeline.export_artifacts(splits, Path(export_dir))
        print(f"\nĐã ghi {len(manifest)} artifact vào {export_dir}")

    _print_summary(splits)
    return splits


def _print_summary(splits: DatasetSplits) -> None:
    report = splits.report
    rows, mapping, text = report["rows"], report["mapping"], report["text"]
    print("\n--- NGUỒN DỮ LIỆU ---")
    print(f"Dòng đọc từ nguồn: {rows['read_from_source']:,}")
    print(f"Dòng bị loại (thiếu id / nhãn lạ): "
          f"{rows['skipped_missing_id']:,} / {rows['skipped_bad_label']:,}")
    print(f"Dòng trùng đã loại: {rows['duplicate_rows_dropped']:,} "
          f"(cặp nhãn mâu thuẫn: {rows['duplicate_pairs_with_conflicting_label']:,})")
    print(f"Cặp (user, game) duy nhất: {rows['unique_user_item_pairs']:,}")

    print("\n--- MAPPING (fit từ train) ---")
    print(f"User: {mapping['num_users']:,} | Game: {mapping['num_items']:,}")
    print(f"Game trong catalog nhưng ngoài mapping: {mapping['catalog_games_outside_mapping']:,}")

    print("\n--- SPLIT ---")
    for name, info in report["splits"].items():
        print(f"{name:<11}: {info['interactions']:>12,} tương tác "
              f"({info['positive']:,} dương, {info['negative']:,} âm) | "
              f"{info['users_with_interactions']:,} user | {info['reviews_with_text']:,} review")
    for name in ("validation", "test"):
        ex = report["exclusions_by_mapping"][name]
        print(f"  {name}: loại {ex['users_excluded_for_positive_outside_mapping']:,} user "
              f"(positive ngoài mapping), bỏ {ex['negative_rows_outside_mapping_dropped']:,} dòng negative ngoài mapping")

    print("\n--- REVIEW TEXT ---")
    if text["warning"]:
        print("CẢNH BÁO:", text["warning"])
    print(f"Cặp có text: {text['pairs_with_text_after_dedup']:,} | "
          f"thiếu/trống: {text['pairs_missing_or_blank_text']:,} | "
          f"item train có review: {text['train_items_with_review']:,}")
    print(f"Timestamp = 0: {report['timestamp']['zero_ratio']:.1%}")
    print(f"Timestamp lỗi đã thay bằng 0 ở ingest: "
          f"{report['timestamp']['invalid_rows_from_ingest']:,} dòng")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chuẩn bị DatasetSplits (tự tạo Parquet nếu chưa có).")
    parser.add_argument("--data-dir", type=Path, help="Thư mục chứa file Parquet")
    parser.add_argument("--raw-dir", type=Path, help="Thư mục chứa games.json và Games_Reviews")
    parser.add_argument("--output-dir", type=Path, help="Nơi lưu Parquet mới tạo")
    parser.add_argument("--reviews-path", type=Path, help="CSV review rời; chỉ nạp text khi được truyền")
    parser.add_argument("--rebuild", action="store_true", help="Tạo lại Parquet từ dữ liệu gốc")
    parser.add_argument("--min-reviews-per-game", type=int, default=5)
    parser.add_argument("--min-interactions-per-user", type=int, default=1)
    parser.add_argument("--strategy", choices=("random", "temporal"), default="random")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-dir", type=Path, help="Ghi bộ artifact review của contract vào thư mục này")
    parser.add_argument("--price-currency", default="unspecified", help="Đơn vị tiền tệ của price, ghi vào report")
    args, _ = parser.parse_known_args(argv)
    return args


if __name__ == "__main__":
    arguments = _parse_args()
    main(
        data_dir=arguments.data_dir,
        reviews_path=arguments.reviews_path,
        raw_dir=arguments.raw_dir,
        output_dir=arguments.output_dir,
        rebuild=arguments.rebuild,
        split_config=SplitConfig(
            val_ratio=arguments.val_ratio,
            test_ratio=arguments.test_ratio,
            random_state=arguments.seed,
            strategy=arguments.strategy,
        ),
        export_dir=arguments.export_dir,
        price_currency=arguments.price_currency,
        min_reviews_per_game=arguments.min_reviews_per_game,
        min_interactions_per_user=arguments.min_interactions_per_user,
    )


__all__ = [
    "DataPipeline",
    "DataValidationError",
    "DatasetSplits",
    "EmptyDatasetError",
    "GameItem",
    "InteractionDataset",
    "ParquetDataLoader",
    "SplitConfig",
    "UnknownItemError",
    "UnknownUserError",
    "main",
]
