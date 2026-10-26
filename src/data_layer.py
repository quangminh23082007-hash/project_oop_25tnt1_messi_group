import json
import csv
import os
from typing import Dict, List, Tuple

import pandas as pd

GAMES_FILE = "games.parquet"
INTERACTIONS_FILE = "interactions.parquet"


def _to_list(value) -> list:
    """Chuẩn hóa genres/tags về list để Parquet lưu được.
    Nếu JSON gốc để dạng dict (tên tag -> số vote) thì chỉ giữ lại tên tag."""
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.keys())
    return list(value)


class Game:
    """Lưu trữ thông tin metadata của một trò chơi."""
    def __init__(self, game_id: str, name: str, price: float, genres: list, positive: int, negative: int, tags: list):
        self.game_id = str(game_id)
        self.name = name
        self.price = price
        self.genres = genres
        self.positive = positive
        self.negative = negative
        self.tags = tags


class Interaction:
    """Lưu trữ một lượt đánh giá/tương tác của User đối với Game."""
    def __init__(self, user_id: str, game_id: str, playtime: float, post_date: str, helpfulness: float, recommended: bool, early_access_review: bool):
        self.user_id = str(user_id)
        self.game_id = str(game_id)
        self.playtime = playtime
        self.post_date = post_date
        self.helpfulness = helpfulness
        self.recommended = recommended
        self.early_access_review = early_access_review


class User:
    """Lưu trữ thông tin người dùng và danh sách các game họ đã tương tác."""
    def __init__(self, user_id: str):
        self.user_id = str(user_id)
        self.interactions: List[Interaction] = []

    def add_interaction(self, interaction: Interaction):
        self.interactions.append(interaction)


class Dataset:
    """Quản lý toàn bộ dữ liệu trung tâm, ánh xạ User, Game và Interaction."""
    def __init__(self):
        self.games: Dict[str, Game] = {}
        self.users: Dict[str, User] = {}
        self.interactions: List[Interaction] = []

    # ------------------------------------------------------------------
    # BƯỚC 1: ĐỌC DỮ LIỆU GỐC
    # ------------------------------------------------------------------
    def _add_game(self, game_id: str, item: dict):
        self.games[game_id] = Game(
            game_id=game_id,
            name=item.get('name', ''),
            price=float(item.get('price', 0.0) or 0.0),
            genres=_to_list(item.get('genres')),
            positive=int(item.get('positive', 0) or 0),
            negative=int(item.get('negative', 0) or 0),
            tags=_to_list(item.get('tags')),
        )

    def load_games_from_json(self, json_path: str):
        """Đọc thông tin Game từ file JSON (dạng dict hoặc dạng list)."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict):
                # Key là Game ID, Value là thông tin game
                for game_key, item in data.items():
                    game_id = str(item.get('game_id') or item.get('appid') or game_key)
                    if game_id:
                        self._add_game(game_id, item)
            elif isinstance(data, list):
                for item in data:
                    game_id = str(item.get('game_id') or item.get('appid', ''))
                    if game_id:
                        self._add_game(game_id, item)
        except Exception as e:
            print(f"[Lỗi] Không thể đọc file JSON {json_path}: {e}")

    def load_interactions_from_csv(self, csv_path: str):
        """Đọc một file CSV đánh giá, game_id lấy từ tên file (phần trước dấu '_')."""
        filename = os.path.basename(csv_path)
        game_id = filename.split('_')[0]

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    user_id = row.get('user', '').strip()
                    if not user_id:
                        continue

                    interaction = Interaction(
                        user_id=user_id,
                        game_id=game_id,
                        playtime=float(row.get('playtime', 0.0) or 0.0),
                        post_date=row.get('post_date', ''),
                        helpfulness=float(row.get('helpfulness', 0.0) or 0.0),
                        recommended=str(row.get('recommended', '')).lower() == 'true',
                        early_access_review=str(row.get('early_access_review', '')).lower() == 'true'
                    )
                    self.interactions.append(interaction)

                    if user_id not in self.users:
                        self.users[user_id] = User(user_id)
                    self.users[user_id].add_interaction(interaction)
        except Exception as e:
            print(f"[Lỗi] Không thể đọc file CSV {csv_path}: {e}")

    def load_interactions_from_folder(self, folder_path: str):
        """Đọc toàn bộ file .csv trong thư mục."""
        abs_folder_path = os.path.abspath(folder_path)
        print(f"\n-> Đang kiểm tra thư mục tại: {abs_folder_path}")

        if not os.path.exists(abs_folder_path):
            print("[Lỗi] Python KHÔNG TÌM THẤY thư mục ở đường dẫn trên!")
            return

        all_files = os.listdir(abs_folder_path)
        csv_files = [os.path.join(abs_folder_path, f) for f in all_files if f.lower().endswith(".csv")]

        if not csv_files:
            print("[Cảnh báo] Thư mục tồn tại nhưng không có file .csv!")
            print("=> 5 file đầu tiên Python nhìn thấy:", all_files[:5])
            return

        print(f"-> Đã tìm thấy {len(csv_files)} file CSV. Đang xử lý...")
        for file_path in csv_files:
            self.load_interactions_from_csv(file_path)
        print("Đã tải xong toàn bộ dữ liệu đánh giá.")

    # ------------------------------------------------------------------
    # BƯỚC 2: CHUYỂN SANG BẢNG (DataFrame)
    # ------------------------------------------------------------------
    def to_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Chuyển dữ liệu object Python thành 2 bảng: games_df và interactions_df."""
        games_df = pd.DataFrame({
            "game_id":  [g.game_id for g in self.games.values()],
            "name":     [g.name for g in self.games.values()],
            "price":    [g.price for g in self.games.values()],
            "genres":   [g.genres for g in self.games.values()],
            "positive": [g.positive for g in self.games.values()],
            "negative": [g.negative for g in self.games.values()],
            "tags":     [g.tags for g in self.games.values()],
        })

        interactions_df = pd.DataFrame({
            "user_id":     [i.user_id for i in self.interactions],
            "game_id":     [i.game_id for i in self.interactions],
            "playtime":    [i.playtime for i in self.interactions],
            "post_date":   [i.post_date for i in self.interactions],
            "helpfulness": [i.helpfulness for i in self.interactions],
            "recommended": [i.recommended for i in self.interactions],
            "early_access_review": [i.early_access_review for i in self.interactions],
        })
        # Tối ưu kiểu dữ liệu để file nhỏ và tốn ít RAM
        interactions_df["user_id"] = interactions_df["user_id"].astype("category")
        interactions_df["game_id"] = interactions_df["game_id"].astype("category")
        interactions_df["playtime"] = interactions_df["playtime"].astype("float32")
        interactions_df["helpfulness"] = interactions_df["helpfulness"].astype("float32")
        return games_df, interactions_df

    # ------------------------------------------------------------------
    # BƯỚC 3: LƯU / NẠP CACHE DẠNG PARQUET
    # ------------------------------------------------------------------
    def save_to_parquet(self, cache_dir: str = "parquet_cache"):
        """Lưu dữ liệu thành 2 file Parquet trong thư mục cache_dir."""
        try:
            os.makedirs(cache_dir, exist_ok=True)
            games_df, interactions_df = self.to_dataframes()
            games_df.to_parquet(os.path.join(cache_dir, GAMES_FILE), compression="zstd", index=False)
            interactions_df.to_parquet(os.path.join(cache_dir, INTERACTIONS_FILE), compression="zstd", index=False)
            print(f"Đã lưu dữ liệu dạng Parquet vào thư mục: '{cache_dir}'")
        except Exception as e:
            print(f"[Lỗi] Không thể lưu Parquet: {e}")

    @staticmethod
    def parquet_exists(cache_dir: str) -> bool:
        return (os.path.exists(os.path.join(cache_dir, GAMES_FILE))
                and os.path.exists(os.path.join(cache_dir, INTERACTIONS_FILE)))

    @staticmethod
    def load_tables(cache_dir: str = "parquet_cache") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Đọc thẳng 2 bảng Parquet (nhanh, nhẹ RAM). Dùng cho model/evaluate."""
        games_df = pd.read_parquet(os.path.join(cache_dir, GAMES_FILE))
        interactions_df = pd.read_parquet(os.path.join(cache_dir, INTERACTIONS_FILE))
        return games_df, interactions_df

    @classmethod
    def load_from_parquet(cls, cache_dir: str = "parquet_cache"):
        """Dựng lại đối tượng Dataset (Game, User, Interaction) từ file Parquet."""
        try:
            games_df, inter_df = cls.load_tables(cache_dir)
        except Exception as e:
            print(f"[Lỗi] Không thể đọc Parquet: {e}")
            return None

        dataset = cls()

        for gid, name, price, genres, pos, neg, tags in zip(
            games_df["game_id"].astype(str).tolist(),
            games_df["name"].tolist(),
            games_df["price"].tolist(),
            games_df["genres"].tolist(),
            games_df["positive"].tolist(),
            games_df["negative"].tolist(),
            games_df["tags"].tolist(),
        ):
            dataset.games[gid] = Game(gid, name, price, list(genres), pos, neg, list(tags))

        for uid, gid, playtime, date, helpful, rec, early in zip(
            inter_df["user_id"].astype(str).tolist(),
            inter_df["game_id"].astype(str).tolist(),
            inter_df["playtime"].tolist(),
            inter_df["post_date"].tolist(),
            inter_df["helpfulness"].tolist(),
            inter_df["recommended"].tolist(),
            inter_df["early_access_review"].tolist(),
        ):
            interaction = Interaction(uid, gid, playtime, date, helpful, bool(rec), bool(early))
            dataset.interactions.append(interaction)
            if uid not in dataset.users:
                dataset.users[uid] = User(uid)
            dataset.users[uid].add_interaction(interaction)

        print(f"Đã nạp dữ liệu từ Parquet: '{cache_dir}'")
        return dataset


def get_dataset(data_dir: str, cache_dir: str = None) -> Dataset:
    """Nếu đã có cache Parquet thì nạp, chưa có thì đọc dữ liệu gốc rồi tạo cache."""
    cache_dir = cache_dir or os.path.join(data_dir, "parquet_cache")

    if Dataset.parquet_exists(cache_dir):
        return Dataset.load_from_parquet(cache_dir)

    dataset = Dataset()
    dataset.load_games_from_json(os.path.join(data_dir, "games.json"))
    dataset.load_interactions_from_folder(os.path.join(data_dir, "Game_Reviews"))

    # Chỉ lưu cache khi tải thành công cả game lẫn tương tác
    if dataset.interactions and dataset.games:
        dataset.save_to_parquet(cache_dir)
    else:
        print("[Cảnh báo] Dữ liệu chưa đầy đủ nên KHÔNG tạo cache.")
    return dataset


# ==========================================
# CÁCH SỬ DỤNG (MAIN WORKFLOW)
# ==========================================
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset = get_dataset(SCRIPT_DIR)

    if dataset:
        print("\n--- THỐNG KÊ DỮ LIỆU ---")
        print(f"Tổng số game: {len(dataset.games)}")
        print(f"Tổng số user: {len(dataset.users)}")
        print(f"Tổng số tương tác: {len(dataset.interactions)}")