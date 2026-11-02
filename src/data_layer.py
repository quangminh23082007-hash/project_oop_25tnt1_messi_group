"""
Đọc trực tiếp từ file Parquet
và cung cấp đầy đủ dữ liệu cho thuật toán gợi ý

Nội dung review (văn bản) được nạp riêng từ file CSV.

Nếu CHƯA có file Parquet, hệ thống tự động tạo chúng từ dữ liệu gốc:
    - games.json
    - Games_Reviews/<game_id>_<số lượng đánh giá>.csv
      (trên Kaggle thư mục được giải nén thành Games_Reviews/Games_Reviews)
rồi lưu lại để các lần chạy sau đọc thẳng từ Parquet.

Cách chạy:
    Local : python data_layer.py --raw-dir ./data
    Kaggle: from data_layer import main; dataset = main()
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

# Tăng giới hạn đọc chuỗi lớn
csv.field_size_limit(min(sys.maxsize, 2147483647))


# --------------------------------------------------------------------------- #
# 1. Hệ thống ngoại lệ (Custom Exceptions)
# --------------------------------------------------------------------------- #

class RecommendationSystemError(Exception):
    """Lỗi cơ sở cho hệ thống gợi ý."""


class DataFormatError(RecommendationSystemError):
    """Lỗi khi dữ liệu thiếu hoặc sai định dạng."""


class UnknownUserError(RecommendationSystemError):
    """Lỗi khi không tìm thấy người dùng."""


class UnknownItemError(RecommendationSystemError):
    """Lỗi khi không tìm thấy game."""


# --------------------------------------------------------------------------- #
# 2. Chuẩn hóa dữ liệu & Tính điểm Wilson Score
# --------------------------------------------------------------------------- #

def _clean_id(val: object) -> str:
    if val is None or pd.isna(val):
        return ""
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val).strip()


def _clean_id_list(values: Iterable[object]) -> List[str]:
    """Chuẩn hóa cả cột ID về str"""
    out: List[str] = []
    append = out.append
    for v in values:
        t = type(v)
        if t is str:
            append(v.strip())
        elif t is int:
            append(str(v))
        else:
            append(_clean_id(v))
    return out


def _parse_list(val: object) -> List[str]:
    """Xử lý linh hoạt genres/tags: hỗ trợ chuỗi JSON, mảng ndarray, list, v.v."""
    if val is None:
        return []

    if hasattr(val, "tolist"):
        items = val.tolist()
    elif isinstance(val, (list, tuple, set)):
        items = list(val)
    elif isinstance(val, dict):
        items = list(val.keys())
    elif isinstance(val, str):
        text = val.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                items = parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                items = text.split(",")
        else:
            items = text.split(",")
    else:
        try:
            if bool(pd.isna(val)):
                return []
        except (ValueError, TypeError):
            pass
        items = [val]

    return list(dict.fromkeys(str(x).strip() for x in items if str(x).strip()))


def wilson_score(positive: int, negative: int, z: float = 1.96) -> float:
    """
    Tính điểm Wilson Score Lower Bound (thang 0.0 -> 1.0) với độ tin cậy 95% (z = 1.96).
    Giúp cân bằng giữa tỷ lệ đánh giá tích cực và tổng số lượng đánh giá,
    tránh việc game chỉ có 1-2 đánh giá tích cực (100%) vượt mặt game có hàng nghìn đánh giá.
    """
    total = positive + negative
    if total <= 0:
        return 0.5  # Mặc định trung lập nếu chưa có đánh giá

    p = positive / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center_adjusted = p + z2 / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)

    return max(0.0, min(1.0, (center_adjusted - margin) / denominator))


# --------------------------------------------------------------------------- #
# 3. (Domain Entities) - Dùng slots để tiết kiệm bộ nhớ
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class Game:
    game_id: str
    name: str = ""
    price: float = 0.0
    genres: List[str] = field(default_factory=list)
    positive: int = 0
    negative: int = 0
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.game_id = _clean_id(self.game_id)
        self.name = "" if self.name is None else str(self.name)
        self.price = float(self.price or 0.0)
        self.positive = int(self.positive or 0)
        self.negative = int(self.negative or 0)
        self.genres = _parse_list(self.genres)
        self.tags = _parse_list(self.tags)

    @property
    def wilson_rating(self) -> float:
        """Điểm đánh giá chung của game trên thang 1.0 - 5.0 dựa trên Wilson Score."""
        return round(1.0 + 4.0 * wilson_score(self.positive, self.negative), 2)


@dataclass(slots=True)
class Interaction:
    user_id: str
    game_id: str
    playtime: float = 0.0
    helpfulness: float = 0.0
    recommended: bool = False
    review: str = ""  # Nội dung review, nạp từ file CSV


class User:
    __slots__ = ("user_id", "_interactions")

    def __init__(self, user_id: str) -> None:
        self.user_id = _clean_id(user_id)
        self._interactions: Dict[str, Interaction] = {}

    @property
    def interactions(self) -> Tuple[Interaction, ...]:
        return tuple(self._interactions.values())

    def add_interaction(self, interaction: Interaction) -> None:
        self._interactions[interaction.game_id] = interaction


# --------------------------------------------------------------------------- #
# 4. Quản lý tập dữ liệu (Dataset)
# --------------------------------------------------------------------------- #

class Dataset:
    def __init__(self) -> None:
        self._games: Dict[str, Game] = {}
        self._users: Dict[str, User] = {}

        # Lưu dữ liệu tương tác theo dạng mảng/cột để tối ưu RAM và tốc độ
        self._interaction_data: Dict[str, list] = {
            "user_id": [],
            "game_id": [],
            "recommended": [],
            "playtime": [],
            "helpfulness": [],
        }

        # Review lưu riêng theo khóa (user_id, game_id): chỉ tốn RAM cho dòng thực sự có review
        self._reviews: Dict[Tuple[str, str], str] = {}

        self._user2id: Dict[str, int] = {}
        self._id2user: Dict[int, str] = {}
        self._game2id: Dict[str, int] = {}
        self._id2game: Dict[int, str] = {}
        self._dirty = True

    @property
    def games(self) -> Dict[str, Game]:
        return self._games

    @property
    def users(self) -> Dict[str, User]:
        return self._users

    @property
    def num_games(self) -> int:
        return len(self._games)

    @property
    def num_users(self) -> int:
        return len(self._users)

    @property
    def num_interactions(self) -> int:
        return len(self._interaction_data["user_id"])

    @property
    def num_reviews(self) -> int:
        return len(self._reviews)

    def get_game(self, game_id: str) -> Game:
        """Tra cứu chi tiết đối tượng Game theo game_id."""
        gid = _clean_id(game_id)
        if gid not in self._games:
            raise UnknownItemError(f"Không tìm thấy game '{game_id}'")
        return self._games[gid]

    def get_user(self, user_id: str) -> User:
        """Tra cứu đối tượng User theo user_id."""
        uid = _clean_id(user_id)
        if uid not in self._users:
            raise UnknownUserError(f"Không tìm thấy user '{user_id}'")
        return self._users[uid]

    def add_game(self, game: Game) -> None:
        self._games[game.game_id] = game
        self._dirty = True

    def add_interactions_batch(self, batch_data: Dict[str, list]) -> None:
        """
        Nạp dữ liệu theo lô.

        user_id / game_id được chuẩn hóa về str ngay tại đây, để khóa trong _users,
        _interaction_data và mapping luôn cùng kiểu với Game.game_id.
        Dòng có user_id hoặc game_id rỗng/NaN bị bỏ qua.
        """
        user_ids = _clean_id_list(batch_data["user_id"])
        game_ids = _clean_id_list(batch_data["game_id"])

        keep = [i for i, (u, g) in enumerate(zip(user_ids, game_ids)) if u and g]
        all_kept = len(keep) == len(user_ids)

        for col_name, col_values in batch_data.items():
            if col_name == "user_id":
                values = user_ids
            elif col_name == "game_id":
                values = game_ids
            else:
                values = col_values
            if not all_kept:
                values = [values[i] for i in keep]
            self._interaction_data.setdefault(col_name, []).extend(values)

        new_users = user_ids if all_kept else [user_ids[i] for i in keep]
        for uid in set(new_users):
            if uid not in self._users:
                self._users[uid] = User(uid)

        self._dirty = True

    def add_reviews_batch(
        self,
        user_ids: Iterable[object],
        game_ids: Iterable[object],
        reviews: Iterable[object],
    ) -> int:
        """
        Gắn review vào các cặp (user, game). Chỉ giữ review của user/game đã có trong dataset
        và có nội dung không rỗng. Trùng cặp (user, game) thì dòng sau ghi đè dòng trước.
        Trả về số review đã nhận.
        """
        users, games = self._users, self._games
        added = 0
        for uid, gid, text in zip(_clean_id_list(user_ids), _clean_id_list(game_ids), reviews):
            if not isinstance(text, str):
                continue
            text = text.strip()
            if text and uid in users and gid in games:
                self._reviews[(uid, gid)] = text
                added += 1
        return added

    def _column_value(self, name: str, index: int, default: object) -> object:
        """Lấy giá trị cột tại index; trả về default nếu cột không tồn tại hoặc rỗng."""
        col = self._interaction_data.get(name)
        if not col or len(col) != self.num_interactions:
            return default
        return col[index]

    def get_interaction(self, index: int) -> Interaction:
        """Tính Trừu tượng (Abstraction): Trả về đối tượng Interaction tại vị trí index."""
        uid = self._interaction_data["user_id"][index]
        gid = self._interaction_data["game_id"][index]
        return Interaction(
            user_id=uid,
            game_id=gid,
            playtime=self._column_value("playtime", index, 0.0),
            helpfulness=self._column_value("helpfulness", index, 0.0),
            recommended=self._column_value("recommended", index, False),
            review=self._reviews.get((uid, gid), ""),
        )

    # ----------------------------------------------------------------------- #
    # Mapping 2 chiều: ID thực <-> chỉ số ma trận
    # ----------------------------------------------------------------------- #

    # 4 bảng mapping là property: truy cập lúc nào cũng được rebuild nếu dữ liệu đã đổi,
    # nên không bao giờ nhận về bảng cũ/rỗng.
    @property
    def user2id(self) -> Dict[str, int]:
        self.build_mapping()
        return self._user2id

    @property
    def id2user(self) -> Dict[int, str]:
        self.build_mapping()
        return self._id2user

    @property
    def game2id(self) -> Dict[str, int]:
        self.build_mapping()
        return self._game2id

    @property
    def id2game(self) -> Dict[int, str]:
        self.build_mapping()
        return self._id2game

    def build_mapping(self) -> None:
        """Tạo bảng ánh xạ ID thực tế sang chỉ số nguyên (0..N-1) cho ma trận."""
        if not self._dirty:
            return
        self._user2id = {uid: idx for idx, uid in enumerate(sorted(self._users))}
        self._id2user = {idx: uid for uid, idx in self._user2id.items()}
        self._game2id = {gid: idx for idx, gid in enumerate(sorted(self._games))}
        self._id2game = {idx: gid for gid, idx in self._game2id.items()}
        self._dirty = False

    def validate_mapping(self) -> Dict[str, int]:
        """Kiểm tra nhanh tính nhất quán giữa dữ liệu tương tác và bảng mapping."""
        self.build_mapping()
        game_ids = set(self._games)
        g_col = self._interaction_data["game_id"]
        seen_games = set(g_col)
        return {
            "Số user đã ánh xạ": len(self._user2id),
            "Số game đã ánh xạ": len(self._game2id),
            "Tương tác có game không nằm trong games": sum(1 for g in g_col if g not in game_ids),
            "Game chưa có tương tác nào": len(game_ids - seen_games),
        }

    def user_index(self, user_id: str) -> int:
        """user_id thực -> chỉ số hàng của ma trận."""
        self.build_mapping()
        uid = _clean_id(user_id)
        if uid not in self.user2id:
            raise UnknownUserError(f"Không tìm thấy user '{user_id}'")
        return self.user2id[uid]

    def game_index(self, game_id: str) -> int:
        """game_id thực -> chỉ số cột của ma trận."""
        self.build_mapping()
        gid = _clean_id(game_id)
        if gid not in self.game2id:
            raise UnknownItemError(f"Không tìm thấy game '{game_id}'")
        return self.game2id[gid]

    def user_from_index(self, index: int) -> str:
        """Chỉ số hàng của ma trận -> user_id thực."""
        self.build_mapping()
        if index not in self.id2user:
            raise UnknownUserError(f"Chỉ số user không hợp lệ: {index}")
        return self.id2user[index]

    def game_from_index(self, index: int) -> str:
        """Chỉ số cột của ma trận -> game_id thực."""
        self.build_mapping()
        if index not in self.id2game:
            raise UnknownItemError(f"Chỉ số game không hợp lệ: {index}")
        return self.id2game[index]

    # ----------------------------------------------------------------------- #
    # Ma trận tương tác
    # ----------------------------------------------------------------------- #

    def get_interaction_matrix_data(
        self,
        mode: str = "rating_weighted",
        negative_score: float = 0.0,
    ) -> List[Tuple[int, int, float]]:
        """
        Trả về danh sách các bộ ba (user_idx, game_idx, score) đã khử trùng lặp.

        Tham số mode:
            - 'rating_weighted': Thang điểm 1.0 - 5.0 kết hợp Wilson Score (từ positive/negative của game)
                                 cùng với recommended và log(playtime) cá nhân của user.
            - 'playtime_weighted': Thang điểm 1.0 - 5.0 chỉ kết hợp recommended và log(playtime).
            - 'binary': 1.0 nếu recommended=True, ngược lại bằng negative_score.
        """
        self.build_mapping()

        u_col = self._interaction_data["user_id"]
        g_col = self._interaction_data["game_id"]
        r_col = self._interaction_data["recommended"]
        if len(r_col) != len(u_col):
            raise DataFormatError("Thiếu cột 'recommended' trong dữ liệu tương tác")
        p_col = self._interaction_data.get("playtime", [])
        has_playtime = len(p_col) == len(u_col)

        dedup_scores: Dict[Tuple[int, int], float] = {}
        u2id = self.user2id
        g2id = self.game2id
        games_dict = self._games

        # Tiền tính toán (cache) base_rating theo Wilson Score cho toàn bộ game
        # để tránh tính căn bậc hai lặp lại hàng chục triệu lần trong vòng lặp.
        game_base_ratings: Dict[str, float] = {}
        if mode == "rating_weighted":
            for gid, game_obj in games_dict.items():
                w_score = wilson_score(game_obj.positive, game_obj.negative)
                game_base_ratings[gid] = 1.0 + 4.0 * w_score  # Thang 1.0 -> 5.0

        for i in range(len(u_col)):
            gid = g_col[i]
            if gid in g2id:
                uid = u_col[i]
                rec = bool(r_col[i])

                if mode == "binary":
                    score = 1.0 if rec else negative_score
                elif mode == "playtime_weighted":
                    playtime = float(p_col[i] or 0.0) if has_playtime else 0.0
                    # Chuẩn hóa thời gian chơi với mốc bão hòa ~100 giờ (log1p(100) ≈ 4.62)
                    norm_play = min(math.log1p(max(0.0, playtime)) / 4.62, 1.0)
                    if rec:
                        score = round(3.0 + 2.0 * norm_play, 2)  # Từ 3.0 đến 5.0
                    else:
                        score = round(1.0 + 1.5 * norm_play, 2)  # Từ 1.0 đến 2.5
                elif mode == "rating_weighted":
                    base_rating = game_base_ratings.get(gid, 3.0)
                    playtime = float(p_col[i] or 0.0) if has_playtime else 0.0
                    norm_play = min(math.log1p(max(0.0, playtime)) / 4.62, 1.0)
                    if rec:
                        # User thích: lấy trung bình có trọng số giữa độ uy tín game và trải nghiệm cá nhân (3.0 -> 5.0)
                        personal_score = 3.0 + 2.0 * norm_play
                        score = min(5.0, round(0.4 * base_rating + 0.6 * personal_score, 2))
                    else:
                        # User không thích: điểm rơi vào khoảng thấp (1.0 -> 2.5), có điều chỉnh theo chất lượng game và playtime
                        personal_score = 1.0 + 1.5 * norm_play
                        score = max(1.0, round(0.3 * base_rating + 0.7 * personal_score - 0.5, 2))
                else:
                    score = 1.0

                dedup_scores[(u2id[uid], g2id[gid])] = score

        return [(u_idx, g_idx, score) for (u_idx, g_idx), score in dedup_scores.items()]

    def get_sparse_matrix(
        self, mode: str = "rating_weighted"
    ) -> csr_matrix:
        """Trả về trực tiếp ma trận thưa CSR (scipy.sparse) kích thước (num_users, num_games)."""
        triples = self.get_interaction_matrix_data(mode=mode)
        rows = [t[0] for t in triples]
        cols = [t[1] for t in triples]
        data = [t[2] for t in triples]

        return csr_matrix(
            (data, (rows, cols)),
            shape=(self.num_users, self.num_games),
            dtype=float,
        )

    def train_test_split(
        self,
        test_ratio: float = 0.2,
        mode: str = "rating_weighted",
        random_state: int = 42,
    ) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
        """Chia danh sách tương tác (u_idx, i_idx, score) thành 2 tập Train và Test."""
        triples = self.get_interaction_matrix_data(mode=mode)
        rng = random.Random(random_state)
        rng.shuffle(triples)

        split_idx = int(len(triples) * (1.0 - test_ratio))
        return triples[:split_idx], triples[split_idx:]


# --------------------------------------------------------------------------- #
# 5. Lưu và Đọc dữ liệu (Parquet + CSV review)
# --------------------------------------------------------------------------- #

class ParquetDatasetRepository:
    @staticmethod
    def games_dataframe(games: Iterable[Game]) -> pd.DataFrame:
        """Chuyển danh sách Game thành DataFrame đúng schema của games.parquet."""
        games = list(games)
        return pd.DataFrame(
            {
                "game_id": [g.game_id for g in games],
                "name": [g.name for g in games],
                "price": [g.price for g in games],
                "genres": [json.dumps(g.genres, ensure_ascii=False) for g in games],
                "positive": [g.positive for g in games],
                "negative": [g.negative for g in games],
                "tags": [json.dumps(g.tags, ensure_ascii=False) for g in games],
            }
        )

    @staticmethod
    def save(
        dataset: Dataset,
        games_path: str = "games.parquet",
        interactions_path: str = "interactions.parquet",
    ) -> None:
        games_df = ParquetDatasetRepository.games_dataframe(dataset.games.values())

        # Chỉ lưu các cột đủ độ dài (cột không có trong dữ liệu gốc sẽ bị bỏ qua).
        # Review không lưu ở đây vì được nạp riêng từ CSV.
        n = dataset.num_interactions
        interactions_df = pd.DataFrame(
            {k: v for k, v in dataset._interaction_data.items() if len(v) == n}
        )

        try:
            Path(games_path).parent.mkdir(parents=True, exist_ok=True)
            games_df.to_parquet(games_path, index=False)
            interactions_df.to_parquet(interactions_path, index=False)
        except Exception as e:
            raise DataFormatError(f"Không thể lưu file Parquet: {e}") from e

    @staticmethod
    def load(
        games_path: str,
        interactions_path: str,
        batch_size: int = 500000,
        reviews_path: Optional[str] = None,
        review_user_col: str = "user",
        review_game_col: str = "game_id",
        review_text_col: str = "review",
    ) -> Dataset:
        dataset = Dataset()

        try:
            # 1. Đọc games.parquet
            games_df = pd.read_parquet(games_path)
            game_cols = ["game_id", "name", "price", "genres", "positive", "negative", "tags"]
            for row in zip(*(games_df[c].tolist() for c in game_cols)):
                dataset.add_game(
                    Game(
                        game_id=row[0],
                        name=row[1],
                        price=row[2],
                        genres=row[3],
                        positive=row[4],
                        negative=row[5],
                        tags=row[6],
                    )
                )

            # 2. Đọc interactions.parquet theo lô lớn (500k dòng/batch)
            parquet_file = pq.ParquetFile(interactions_path)

            available_cols = set(parquet_file.schema.names)
            target_cols = [
                "user_id",
                "game_id",
                "recommended",
                "playtime",
                "helpfulness",
            ]
            cols_to_read = [c for c in target_cols if c in available_cols]

            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=cols_to_read):
                batch_dict = {col: batch[col].to_pylist() for col in cols_to_read}
                dataset.add_interactions_batch(batch_dict)

        except Exception as e:
            raise DataFormatError(f"Không thể tải dữ liệu Parquet: {e}") from e

        # 3. Đọc review từ CSV (sau khi đã có users/games để lọc đúng cặp)
        if reviews_path:
            ParquetDatasetRepository.load_reviews(
                dataset,
                reviews_path,
                user_col=review_user_col,
                game_col=review_game_col,
                review_col=review_text_col,
            )

        return dataset

    @staticmethod
    def load_reviews(
        dataset: Dataset,
        reviews_path: str,
        user_col: str = "user_id",
        game_col: str = "game_id",
        review_col: str = "review",
        chunksize: int = 200000,
    ) -> int:
        """
        Stream file CSV review theo chunk và gắn vào dataset theo khóa (user_id, game_id).
        Trả về tổng số review đã gắn.
        """
        wanted = [user_col, game_col, review_col]
        try:
            header = list(pd.read_csv(reviews_path, nrows=0).columns)
        except Exception as e:
            raise DataFormatError(f"Không thể đọc file CSV review: {e}") from e

        # Tự nhận tên cột user: file review có thể dùng "user_id" hoặc "user"
        if user_col not in header:
            for alt in ("user_id", "user"):
                if alt in header:
                    user_col = alt
                    wanted = [user_col, game_col, review_col]
                    break

        missing = [c for c in wanted if c not in header]
        if missing:
            raise DataFormatError(
                f"File review thiếu cột {missing}. Các cột hiện có: {header}"
            )

        total = 0
        try:
            # dtype=str: giữ nguyên ID (tránh 730 -> 730.0); na_filter=False: ô rỗng thành "".
            reader = pd.read_csv(
                reviews_path,
                usecols=wanted,
                dtype=str,
                na_filter=False,
                chunksize=chunksize,
                encoding="utf-8",
                encoding_errors="replace",
            )
            for chunk in reader:
                total += dataset.add_reviews_batch(
                    chunk[user_col].tolist(),
                    chunk[game_col].tolist(),
                    chunk[review_col].tolist(),
                )
        except Exception as e:
            raise DataFormatError(f"Không thể tải dữ liệu review từ CSV: {e}") from e

        return total


# --------------------------------------------------------------------------- #
# 6. Tạo Parquet từ dữ liệu gốc (games.json + Games_Reviews/*.csv)
# --------------------------------------------------------------------------- #

# Tên file review: <game_id>_<số lượng đánh giá>.csv, ví dụ 730_1523.csv
_REVIEW_FILE_RE = re.compile(r"^(\d+)_(\d+)\.csv$", re.IGNORECASE)

# Cột trong CSV review (theo file mẫu 251590_49.csv):
#   user, playtime, post_date, helpfulness, review, recommend, early_access_review
# - game_id KHÔNG nằm trong CSV mà lấy từ tên file.
# - user là tên hiển thị (không phải steamid).
# - playtime tính bằng GIỜ; recommend là "Recommended" / "Not Recommended".
_REVIEW_COLUMNS: Dict[str, str] = {
    "user_id": "user",
    "recommended": "recommend",
    "playtime": "playtime",
    "helpfulness": "helpfulness",
    "review": "review",
}
_REQUIRED_REVIEW_COLUMNS = ("user_id", "recommended")
_RECOMMENDED = "recommended"
_NOT_RECOMMENDED = "not recommended"

_SKIP_DIRS = {"__pycache__", "node_modules", "site-packages", "venv", "env"}


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if x != x else x  # x != x: NaN


def read_games_json(path: Path) -> Dict[str, Game]:
    """
    Đọc games.json -> {game_id: Game}.
    Cấu trúc: {"<appid>": {"name":..., "price":..., "genres": [...],
                           "tags": {...}, "positive":..., "negative":..., ...}, ...}
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise DataFormatError(f"Không thể đọc {path}: {e}") from e

    if not isinstance(raw, dict):
        raise DataFormatError(
            f"games.json phải là dict khóa theo appid, nhưng đang là {type(raw).__name__}"
        )

    games: Dict[str, Game] = {}
    for key, rec in raw.items():
        gid = _clean_id(key)
        if not gid or not isinstance(rec, dict):
            continue
        games[gid] = Game(
            game_id=gid,
            name=rec.get("name") or "",
            price=_to_float(rec.get("price")),
            genres=rec.get("genres"),
            positive=int(_to_float(rec.get("positive"))),
            negative=int(_to_float(rec.get("negative"))),
            tags=rec.get("tags"),
        )

    if not games:
        raise DataFormatError(f"Không đọc được game nào từ {path}")
    return games


def _resolve_review_columns(header: List[str]) -> Dict[str, str]:
    """Ánh xạ tên cột logic -> tên cột thực tế; báo lỗi nếu thiếu cột bắt buộc."""
    present = {h.strip(): h for h in header}
    missing = [_REVIEW_COLUMNS[k] for k in _REQUIRED_REVIEW_COLUMNS if _REVIEW_COLUMNS[k] not in present]
    if missing:
        raise DataFormatError(f"thiếu cột bắt buộc {missing} (các cột hiện có: {header})")
    return {k: present[v] for k, v in _REVIEW_COLUMNS.items() if v in present}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)


class _InteractionWriter:
    """Ghi interactions.parquet theo từng lô, không giữ toàn bộ dữ liệu trong RAM."""

    SCHEMA = pa.schema(
        [
            ("user_id", pa.string()),
            ("game_id", pa.string()),
            ("recommended", pa.bool_()),
            ("playtime", pa.float64()),
            ("helpfulness", pa.float64()),
        ]
    )

    def __init__(self, path: Path, flush_rows: int = 500_000) -> None:
        self._writer = pq.ParquetWriter(str(path), self.SCHEMA)
        self._flush_rows = flush_rows
        self._buf: List[pd.DataFrame] = []
        self._buf_rows = 0
        self.rows = 0
        self.game_ids: Set[str] = set()

    def add(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        self.game_ids.update(df["game_id"].unique().tolist())
        self._buf.append(df)
        self._buf_rows += len(df)
        self.rows += len(df)
        if self._buf_rows >= self._flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        df = pd.concat(self._buf, ignore_index=True)
        self._writer.write_table(pa.Table.from_pandas(df, schema=self.SCHEMA, preserve_index=False))
        self._buf, self._buf_rows = [], 0

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._writer.close()


def _scan_raw_data(roots: Iterable[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Quét đệ quy để tìm games.json và thư mục chứa các file <gameid>_<n>.csv.
    Thư mục review được nhận diện theo nội dung nên tự xử lý được cả trường hợp
    Games_Reviews/Games_Reviews (Kaggle) lẫn Games_Reviews (local).
    """
    games_json: Optional[Path] = None
    best_dir: Optional[Path] = None
    best_count = 0

    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS]
            if games_json is None:
                for f in filenames:
                    if f.lower() == "games.json":
                        games_json = Path(dirpath) / f
                        break
            n = sum(1 for f in filenames if _REVIEW_FILE_RE.match(f))
            if n > best_count:
                best_count, best_dir = n, Path(dirpath)

    return games_json, best_dir


def find_raw_data(raw_dir: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Tìm (games.json, thư mục review).
        - Có raw_dir: chỉ quét trong raw_dir.
        - Không có: quét /kaggle/input (Kaggle) rồi thư mục hiện tại và thư mục chứa file code (local).
    """
    if raw_dir:
        root = Path(raw_dir)
        if not root.exists():
            raise FileNotFoundError(f"Thư mục dữ liệu gốc không tồn tại: {root}")
        roots = [root]
    else:
        roots = [Path("/kaggle/input"), Path.cwd()]
        try:
            roots.append(Path(__file__).resolve().parent)
        except NameError:  # chạy trong notebook, không có __file__
            pass
        roots = list(dict.fromkeys(r.resolve() if r.exists() else r for r in roots))

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
    Lọc & chuyển dữ liệu gốc thành games.parquet, interactions.parquet (+ reviews.csv).

    Các bước lọc:
        - Bỏ file review của game không có trong games.json.
        - Bỏ file có số đánh giá (ghi trong tên file) < min_reviews_per_game.
        - Bỏ dòng thiếu user_id; mỗi cặp (user, game) chỉ giữ bản ghi đầu tiên.
        - Bỏ user có ít hơn min_interactions_per_user tương tác (mặc định 1 = không lọc).
        - Chỉ giữ game có ít nhất 1 tương tác (only_games_with_reviews=True).

    Ghi ra file tạm rồi mới đổi tên, nên nếu bị ngắt giữa chừng sẽ không để lại
    file Parquet dở dang.
    """
    log = print if verbose else (lambda *a, **k: None)

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
            "Không tìm thấy thư mục Games_Reviews (chứa các file dạng <gameid>_<số lượng>.csv). "
            "Hãy truyền raw_dir hoặc thêm dataset gốc vào Kaggle Input."
        )
    games_json, reviews_dir = Path(games_json), Path(reviews_dir)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    games_p, inter_p, rev_p = out / "games.parquet", out / "interactions.parquet", out / "reviews.csv"

    def tmp(p: Path) -> Path:
        return p.with_name(p.name + ".tmp")

    stage1_p = out / "interactions.stage1.tmp"
    temp_files = [tmp(games_p), tmp(inter_p), tmp(rev_p), stage1_p]

    log(f"[build] games.json     : {games_json}")
    log(f"[build] Games_Reviews  : {reviews_dir}")
    log(f"[build] Thư mục xuất   : {out}")

    games = read_games_json(games_json)
    log(f"[build] Đọc {len(games):,} game từ games.json")

    # Lập danh sách file review cần xử lý (lọc theo tên file, chưa phải mở file)
    files: List[Tuple[str, Path]] = []
    for path in sorted(reviews_dir.iterdir()):
        m = _REVIEW_FILE_RE.match(path.name)
        if not m or not path.is_file():
            continue
        gid, declared_count = m.group(1), int(m.group(2))
        if gid in games and declared_count >= min_reviews_per_game:
            files.append((gid, path))
    log(f"[build] {len(files):,} file review hợp lệ cần xử lý")
    if not files:
        raise DataFormatError(
            f"Không có file review nào khớp mẫu <gameid>_<số lượng>.csv và game_id có trong games.json "
            f"(thư mục: {reviews_dir})"
        )

    second_pass = min_interactions_per_user > 1
    user_counts: Dict[str, int] = {}
    skipped = 0
    bad_rec = 0
    n_reviews = 0
    col_cache: Dict[Tuple[str, ...], Dict[str, str]] = {}

    pass1_writer: Optional[_InteractionWriter] = None
    final_writer: Optional[_InteractionWriter] = None
    rev_file = None
    try:
        pass1_writer = _InteractionWriter(stage1_p if second_pass else tmp(inter_p))
        if include_reviews:
            rev_file = open(tmp(rev_p), "w", encoding="utf-8", newline="")
            rev_writer = csv.writer(rev_file)
            rev_writer.writerow(["user_id", "game_id", "review"])

        # ---- Pass 1: đọc từng file CSV, chuẩn hóa, ghi ra Parquet ----
        for n, (gid, path) in enumerate(files, 1):
            try:
                header = list(
                    pd.read_csv(path, nrows=0, encoding="utf-8", encoding_errors="replace").columns
                )
                key = tuple(header)
                if key not in col_cache:
                    col_cache[key] = _resolve_review_columns(header)
                cols = col_cache[key]
                usecols = list(dict.fromkeys(cols.values()))
                seen: Set[str] = set()

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
                    users = chunk[cols["user_id"]].str.strip()
                    rec_text = chunk[cols["recommended"]].str.strip().str.lower()

                    # Chỉ nhận đúng 2 giá trị "Recommended" / "Not Recommended"
                    valid_rec = rec_text.isin((_RECOMMENDED, _NOT_RECOMMENDED))
                    bad_rec += int((~valid_rec & (users != "")).sum())

                    keep = (users != "") & valid_rec & ~users.duplicated(keep="first")
                    if seen:
                        keep &= ~users.isin(seen)
                    chunk, users, rec_text = chunk[keep], users[keep], rec_text[keep]
                    if chunk.empty:
                        continue
                    seen.update(users.tolist())

                    df = pd.DataFrame(
                        {
                            "user_id": users,
                            "game_id": gid,
                            "recommended": rec_text == _RECOMMENDED,
                            "playtime": _numeric(chunk[cols["playtime"]])
                            if "playtime" in cols
                            else 0.0,
                            "helpfulness": _numeric(chunk[cols["helpfulness"]])
                            if "helpfulness" in cols
                            else 0.0,
                        }
                    )
                    pass1_writer.add(df)

                    if second_pass:
                        for u in users.tolist():
                            user_counts[u] = user_counts.get(u, 0) + 1

                    if include_reviews and "review" in cols:
                        texts = chunk[cols["review"]].str.strip()
                        has = texts != ""
                        k = int(has.sum())
                        if k:
                            rev_writer.writerows(zip(users[has].tolist(), [gid] * k, texts[has].tolist()))
                            n_reviews += k

            except Exception as e:  # một file lỗi không làm hỏng cả quá trình
                skipped += 1
                log(f"[build] Bỏ qua {path.name}: {e}")

            if verbose and (n % 500 == 0 or n == len(files)):
                log(f"[build] {n:,}/{len(files):,} file | {pass1_writer.rows:,} tương tác")

        pass1_writer.close()
        if rev_file:
            rev_file.close()
            rev_file = None

        if pass1_writer.rows == 0:
            raise DataFormatError(
                f"Không đọc được dòng tương tác nào (bỏ qua {skipped} file). "
                f"Hãy kiểm tra tên cột CSV so với _REVIEW_COLUMNS."
            )

        kept_games = pass1_writer.game_ids
        total_rows = pass1_writer.rows

        # ---- Pass 2 (tùy chọn): lọc user có quá ít tương tác ----
        if second_pass:
            valid_users = {u for u, c in user_counts.items() if c >= min_interactions_per_user}
            user_counts.clear()
            log(f"[build] Lọc user (>= {min_interactions_per_user} tương tác): giữ {len(valid_users):,} user")
            final_writer = _InteractionWriter(tmp(inter_p))
            for batch in pq.ParquetFile(str(stage1_p)).iter_batches(batch_size=500_000):
                df = batch.to_pandas()
                final_writer.add(df[df["user_id"].isin(valid_users)])
            final_writer.close()
            if final_writer.rows == 0:
                raise DataFormatError("Không còn tương tác nào sau khi lọc user.")
            kept_games = final_writer.game_ids
            total_rows = final_writer.rows

        # ---- games.parquet ----
        selected = [g for gid, g in games.items() if not only_games_with_reviews or gid in kept_games]
        ParquetDatasetRepository.games_dataframe(selected).to_parquet(tmp(games_p), index=False)

        # ---- Đổi tên file tạm: games.parquet cuối cùng (làm dấu hiệu đã build xong) ----
        os.replace(tmp(inter_p), inter_p)
        if include_reviews:
            os.replace(tmp(rev_p), rev_p)
        os.replace(tmp(games_p), games_p)

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
        if rev_file:
            rev_file.close()
        for f in temp_files:
            f.unlink(missing_ok=True)
        raise
    finally:
        stage1_p.unlink(missing_ok=True)

    log(
        f"[build] Xong: {len(selected):,} game, {total_rows:,} tương tác, "
        f"{n_reviews:,} review chữ, bỏ qua {skipped} file lỗi"
    )
    if bad_rec:
        log(f"[build] Cảnh báo: bỏ {bad_rec:,} dòng có cột recommend không phải Recommended/Not Recommended")
    return games_p, inter_p


# --------------------------------------------------------------------------- #
# 7. Hàm chạy chính (tự quét thư mục Kaggle & Local)
# --------------------------------------------------------------------------- #

def find_dataset_dir() -> Path:
    """Tự động quét đệ quy để tìm thư mục chứa games.parquet và interactions.parquet."""
    candidates = [
        Path("/kaggle/input/datasets/synnix/database"),
        Path("/kaggle/input/database"),
        Path("/kaggle/input/Database"),
        Path("/kaggle/working/processed"),
    ]
    for p in candidates:
        if (p / "games.parquet").is_file() and (p / "interactions.parquet").is_file():
            return p

    search_roots = [Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()]
    for base in search_roots:
        if base.exists():
            for root, dirnames, files in os.walk(base):
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS]
                if "games.parquet" in files and "interactions.parquet" in files:
                    return Path(root)

    return Path.cwd()


def default_output_dir() -> Path:
    """Kaggle: /kaggle/working/processed (Input chỉ đọc). Local: ./processed."""
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/processed")
    return Path.cwd() / "processed"


def _pick_output_dir(data_dir: Optional[Path], output_dir: Optional[Path]) -> Path:
    if output_dir:
        return Path(output_dir)
    # /kaggle/input là chỉ-đọc nên không thể ghi parquet vào đó
    if data_dir and not str(data_dir).replace("\\", "/").startswith("/kaggle/input"):
        return Path(data_dir)
    return default_output_dir()


def main(
    data_dir: Optional[Path] = None,
    reviews_path: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    rebuild: bool = False,
    **build_options: object,
) -> Dataset:
    """
    Nạp dataset. Nếu chưa có file Parquet (hoặc rebuild=True) thì tự tạo từ
    games.json + Games_Reviews/*.csv rồi mới nạp.

    Tham số:
        data_dir   : thư mục chứa games.parquet / interactions.parquet (bỏ trống = tự tìm).
        raw_dir    : thư mục chứa games.json và Games_Reviews (bỏ trống = tự tìm
                     trong /kaggle/input, thư mục hiện tại).
        output_dir : nơi lưu Parquet mới tạo (mặc định: Kaggle -> /kaggle/working/processed,
                     local -> ./processed).
        rebuild    : True = bỏ qua Parquet cũ và tạo lại từ dữ liệu gốc.
        build_options: truyền thẳng cho build_parquet_from_raw
                     (min_reviews_per_game, min_interactions_per_user, ...).
    """
    target_dir = Path(data_dir) if data_dir else find_dataset_dir()
    games_p = target_dir / "games.parquet"
    inter_p = target_dir / "interactions.parquet"

    if rebuild or not (games_p.is_file() and inter_p.is_file()):
        out_dir = _pick_output_dir(Path(data_dir) if data_dir else None, output_dir)
        if rebuild:
            print("Rebuild: tạo lại Parquet từ dữ liệu gốc...")
        else:
            print(f"Chưa có file Parquet tại {target_dir} -> tạo từ dữ liệu gốc...")
        build_parquet_from_raw(out_dir, raw_dir=raw_dir, **build_options)  # type: ignore[arg-type]
        target_dir = out_dir
        games_p = target_dir / "games.parquet"
        inter_p = target_dir / "interactions.parquet"

    # Nếu không truyền đường dẫn, thử tìm reviews.csv cạnh các file parquet
    rev_p = Path(reviews_path) if reviews_path else target_dir / "reviews.csv"

    print(f"Đang nạp dữ liệu từ: {target_dir}")
    if not games_p.is_file() or not inter_p.is_file():
        raise FileNotFoundError(f"Không tìm thấy file parquet tại {target_dir}")

    dataset = ParquetDatasetRepository.load(
        str(games_p),
        str(inter_p),
        reviews_path=str(rev_p) if rev_p.is_file() else None,
    )

    print("\n--- THỐNG KÊ DỮ LIỆU ---")
    print(f"Tổng số Game: {dataset.num_games:,}")
    print(f"Tổng số User: {dataset.num_users:,}")
    print(f"Tổng số Tương tác: {dataset.num_interactions:,}")
    print(f"Tổng số Review: {dataset.num_reviews:,}")

    report = dataset.validate_mapping()
    print("\n--- KIỂM TRA MAPPING ---")
    for key, value in report.items():
        print(f"{key}: {value:,}")

    return dataset


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nạp dataset (tự tạo Parquet nếu chưa có).")
    parser.add_argument("--data-dir", type=Path, help="Thư mục chứa file Parquet")
    parser.add_argument("--raw-dir", type=Path, help="Thư mục chứa games.json và Games_Reviews")
    parser.add_argument("--output-dir", type=Path, help="Nơi lưu Parquet mới tạo")
    parser.add_argument("--reviews-path", type=Path, help="File reviews.csv có sẵn (tùy chọn)")
    parser.add_argument("--rebuild", action="store_true", help="Tạo lại Parquet từ dữ liệu gốc")
    parser.add_argument("--min-reviews-per-game", type=int, default=1)
    parser.add_argument("--min-interactions-per-user", type=int, default=1)
    # parse_known_args: bỏ qua tham số lạ khi chạy trong notebook/Kaggle
    args, _ = parser.parse_known_args(argv)
    return args


if __name__ == "__main__":
    _a = _parse_args()
    main(
        data_dir=_a.data_dir,
        reviews_path=_a.reviews_path,
        raw_dir=_a.raw_dir,
        output_dir=_a.output_dir,
        rebuild=_a.rebuild,
        min_reviews_per_game=_a.min_reviews_per_game,
        min_interactions_per_user=_a.min_interactions_per_user,
    )