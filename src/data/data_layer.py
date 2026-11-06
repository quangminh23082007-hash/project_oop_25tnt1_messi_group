"""
Nạp InteractionDataset theo contract (src.contracts).

Nếu chưa có Parquet, tự tạo từ games.json và Games_Reviews/*.csv.

    python -m src.data_layer --raw-dir ./data
    python -m src.data_layer --raw-dir ./data --split --strategy temporal
    Kaggle: from src.data_layer import main; dataset = main()
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parents[1]
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


def main(
    data_dir: Optional[Path] = None,
    reviews_path: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    rebuild: bool = False,
    **build_options: object,
) -> InteractionDataset:
    """
    Nạp dataset. Chưa có Parquet, hoặc rebuild=True, thì tạo từ dữ liệu gốc rồi mới nạp.

    build_options truyền thẳng cho build_parquet_from_raw
    (min_reviews_per_game, min_interactions_per_user, ...).
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

    review_file = Path(reviews_path) if reviews_path else target_dir / "reviews.csv"
    print(f"Đang nạp dữ liệu từ: {target_dir}")
    if not games_path.is_file() or not interactions_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file parquet tại {target_dir}")

    dataset = ParquetDataLoader().load_dataset(
        str(games_path),
        str(interactions_path),
        reviews_path=str(review_file) if review_file.is_file() else None,
    )

    print("\n--- THỐNG KÊ DỮ LIỆU ---")
    print(f"Tổng số Game: {dataset.num_items:,}")
    print(f"Tổng số User: {dataset.num_users:,}")
    print(f"Tổng số Tương tác: {dataset.num_interactions:,}")
    print(f"Tổng số Review: {dataset.num_reviews:,}")
    print("\n--- KIỂM TRA MAPPING ---")
    for key, value in dataset.mapping_report().items():
        print(f"{key}: {value:,}")
    return dataset


def make_splits(
    dataset: InteractionDataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42,
    strategy: str = "random",
) -> DatasetSplits:
    """Chia train/validation/test theo contract (dùng chung mapper). Gọi sau main()."""
    splits = dataset.split(
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        strategy=strategy,
    )
    print("\n--- THỐNG KÊ SPLIT ---")
    for key, value in splits.report.items():
        if not key.endswith("fingerprint"):
            print(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")
    return splits


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nạp InteractionDataset (tự tạo Parquet nếu chưa có).")
    parser.add_argument("--data-dir", type=Path, help="Thư mục chứa file Parquet")
    parser.add_argument("--raw-dir", type=Path, help="Thư mục chứa games.json và Games_Reviews")
    parser.add_argument("--output-dir", type=Path, help="Nơi lưu Parquet mới tạo")
    parser.add_argument("--reviews-path", type=Path, help="File reviews.csv có sẵn (tùy chọn)")
    parser.add_argument("--rebuild", action="store_true", help="Tạo lại Parquet từ dữ liệu gốc")
    parser.add_argument("--min-reviews-per-game", type=int, default=1)
    parser.add_argument("--min-interactions-per-user", type=int, default=1)
    parser.add_argument("--split", action="store_true", help="Chia train/validation/test sau khi nạp")
    parser.add_argument("--strategy", choices=("random", "temporal"), default="random")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args(argv)
    return args


if __name__ == "__main__":
    arguments = _parse_args()
    loaded = main(
        data_dir=arguments.data_dir,
        reviews_path=arguments.reviews_path,
        raw_dir=arguments.raw_dir,
        output_dir=arguments.output_dir,
        rebuild=arguments.rebuild,
        min_reviews_per_game=arguments.min_reviews_per_game,
        min_interactions_per_user=arguments.min_interactions_per_user,
    )
    if arguments.split:
        make_splits(
            loaded,
            val_ratio=arguments.val_ratio,
            test_ratio=arguments.test_ratio,
            random_state=arguments.seed,
            strategy=arguments.strategy,
        )


__all__ = [
    "DataValidationError",
    "EmptyDatasetError",
    "DatasetSplits",
    "GameItem",
    "InteractionDataset",
    "ParquetDataLoader",
    "UnknownItemError",
    "UnknownUserError",
    "main",
    "make_splits",
]
