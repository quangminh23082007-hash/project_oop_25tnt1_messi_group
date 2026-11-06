"""InteractionDataset, IdMapper và chia DatasetSplits theo Contracts v1."""

from __future__ import annotations

import hashlib
from numbers import Integral
from types import MappingProxyType
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from src.contracts.dto import (
    DatasetSplits,
    GameItem,
    Interaction,
    InteractionRecord,
    ReviewRecord,
)
from src.contracts.exceptions import (
    DataValidationError,
    EmptyDatasetError,
    InvalidParameterError,
    UnknownItemError,
    UnknownUserError,
)
from src.data.entities import GameStats
from src.data.parsing import clean_id

_EMPTY_INT = np.empty(0, dtype=np.int64)


def _sha256(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


class IdMapper:
    """Ánh xạ id thô (str) sang chỉ số 0..N-1 và ngược lại."""

    def __init__(
        self,
        ordered_raw_ids: Sequence[str],
        *,
        missing_error: type[Exception],
        kind: str,
    ) -> None:
        ids = list(ordered_raw_ids)
        self._to_idx: Dict[str, int] = {raw: index for index, raw in enumerate(ids)}
        if len(self._to_idx) != len(ids):
            raise DataValidationError(f"Danh sách id {kind} có phần tử trùng lặp.")
        self._to_raw: List[str] = ids
        self._missing_error = missing_error
        self._kind = kind
        self._fingerprint: Optional[str] = None

    def __len__(self) -> int:
        return len(self._to_raw)

    @property
    def size(self) -> int:
        """Số id đã ánh xạ (contract IdMapper.size)."""
        return len(self._to_raw)

    @property
    def raw_ids(self) -> Tuple[str, ...]:
        return tuple(self._to_raw)

    @property
    def fingerprint(self) -> str:
        if self._fingerprint is None:
            payload = "\x00".join(self._to_raw).encode("utf-8")
            self._fingerprint = _sha256(self._kind.encode("utf-8"), b"\x01", payload)
        return self._fingerprint

    def contains_raw(self, raw_id: str) -> bool:
        return clean_id(raw_id) in self._to_idx

    def to_idx(self, raw_id: str) -> int:
        key = clean_id(raw_id)
        try:
            return self._to_idx[key]
        except KeyError:
            raise self._missing_error(
                f"Không tìm thấy {self._kind} '{raw_id}'.",
                **{f"{self._kind}_id": raw_id},
            ) from None

    def to_raw(self, index: int) -> str:
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(self._to_raw):
            raise self._missing_error(f"Chỉ số {self._kind} không hợp lệ: {index}.")
        return self._to_raw[int(index)]


class InteractionDataset:
    """
    Tập tương tác đã gắn chỉ số nguyên, thỏa Protocol InteractionDataset của contract.

    - Nhãn rating nhị phân 1.0 (Recommended) / 0.0 (Not Recommended).
    - Mỗi cặp (user, item) chỉ có một dòng; trùng thì giữ dòng có timestamp mới nhất.
    - Các split sinh ra từ split() dùng chung user_mapper/item_mapper với dataset gốc,
      nên cùng num_users, num_items và mapping_fingerprint.
    - Review chữ gắn theo cặp (user, item); split nào chứa cặp đó thì mang theo review.
    """

    def __init__(
        self,
        records: Sequence[InteractionRecord],
        games: Mapping[str, GameItem],
        *,
        game_stats: Optional[Mapping[str, GameStats]] = None,
        split_name: str = "full",
        user_mapper: Optional[IdMapper] = None,
        item_mapper: Optional[IdMapper] = None,
    ) -> None:
        if records is None or isinstance(records, (str, bytes)):
            raise DataValidationError("records phải là một dãy InteractionRecord.")
        if not isinstance(games, Mapping):
            raise DataValidationError("games phải là dict game_id -> GameItem.")
        if not isinstance(split_name, str) or not split_name.strip():
            raise DataValidationError("split_name phải là chuỗi không rỗng.")
        for mapper, label in ((user_mapper, "user_mapper"), (item_mapper, "item_mapper")):
            if mapper is not None and not isinstance(mapper, IdMapper):
                raise DataValidationError(f"{label} phải là IdMapper.")

        parsed: List[InteractionRecord] = []
        for record in records:
            if not isinstance(record, InteractionRecord):
                raise DataValidationError(
                    f"Mỗi phần tử phải là InteractionRecord, nhận {type(record).__name__}."
                )
            parsed.append(record)

        catalog: Dict[str, GameItem] = {}
        for game in games.values():
            if not isinstance(game, GameItem):
                raise DataValidationError(f"Catalog chỉ chứa GameItem, nhận {type(game).__name__}.")
            catalog[game.item_id] = game

        synthetic: Set[str] = set()
        for record in parsed:
            item_key = clean_id(record.item_id)
            if item_key not in catalog:
                catalog[item_key] = GameItem(item_id=item_key, title=item_key)
                synthetic.add(item_key)

        self.split_name = split_name.strip()
        self._games: Dict[str, GameItem] = catalog
        self._games_view: Mapping[str, GameItem] = MappingProxyType(self._games)
        self._synthetic_ids: Set[str] = synthetic
        self._game_stats: Dict[str, GameStats] = dict(game_stats or {})
        self.user_mapper = user_mapper if user_mapper is not None else IdMapper(
            sorted({clean_id(r.user_id) for r in parsed}), missing_error=UnknownUserError, kind="user"
        )
        self.item_mapper = item_mapper if item_mapper is not None else IdMapper(
            sorted(catalog), missing_error=UnknownItemError, kind="item"
        )
        self._stride = max(self.item_mapper.size, 1)

        count = len(parsed)
        users = np.empty(count, dtype=np.int64)
        items = np.empty(count, dtype=np.int64)
        ratings = np.empty(count, dtype=np.float64)
        stamps = np.empty(count, dtype=np.int64)
        texts: Dict[int, str] = {}
        user_lookup, item_lookup = self.user_mapper._to_idx, self.item_mapper._to_idx
        for pos, record in enumerate(parsed):
            user_key, item_key = clean_id(record.user_id), clean_id(record.item_id)
            user_idx, item_idx = user_lookup.get(user_key), item_lookup.get(item_key)
            if user_idx is None:
                raise UnknownUserError(f"Không tìm thấy user '{user_key}'.", user_id=user_key)
            if item_idx is None:
                raise UnknownItemError(f"Không tìm thấy item '{item_key}'.", item_id=item_key)
            users[pos], items[pos] = user_idx, item_idx
            ratings[pos], stamps[pos] = float(record.rating), int(record.timestamp)
            if record.text is not None and record.text.strip():
                texts[pos] = record.text.strip()

        # Trùng (user, item): giữ dòng timestamp mới nhất (hòa thì dòng đến sau).
        keys = users * self._stride + items
        order = np.lexsort((stamps, keys))
        sorted_keys = keys[order]
        is_last = np.ones(count, dtype=bool)
        is_last[:-1] = sorted_keys[1:] != sorted_keys[:-1]
        keep = order[is_last]  # đã tăng dần theo (user, item)
        self.num_duplicates_dropped = count - len(keep)

        self._users, self._items = users[keep], items[keep]
        self._ratings, self._stamps = ratings[keep], stamps[keep]
        self._texts: Dict[int, str] = {
            int(keys[pos]): texts[pos] for pos in keep.tolist() if pos in texts
        }
        self._finalize()

    # ------------------------------------------------------------------ nội bộ

    def _finalize(self) -> None:
        """Dựng chỉ mục (keys, indptr) và xóa cache sau khi mảng dữ liệu đã được gán."""
        self._keys = self._users * self._stride + self._items
        counts = np.bincount(self._users, minlength=self.user_mapper.size)
        self._indptr = np.zeros(self.user_mapper.size + 1, dtype=np.int64)
        np.cumsum(counts, out=self._indptr[1:])
        self._interactions: Optional[Tuple[Interaction, ...]] = None
        self._records: Optional[Tuple[InteractionRecord, ...]] = None
        self._fingerprint: Optional[str] = None
        self._review_fingerprint: Optional[str] = None
        self._mapping_fingerprint: Optional[str] = None

    def _derive(self, rows: np.ndarray, split_name: str) -> "InteractionDataset":
        """Dataset con gồm các dòng `rows` (đã sắp tăng), dùng chung mapper và catalog."""
        child = object.__new__(InteractionDataset)
        child.split_name = split_name
        child._games, child._games_view = self._games, self._games_view
        child._synthetic_ids, child._game_stats = self._synthetic_ids, self._game_stats
        child.user_mapper, child.item_mapper = self.user_mapper, self.item_mapper
        child._stride = self._stride
        child.num_duplicates_dropped = 0
        child._users, child._items = self._users[rows], self._items[rows]
        child._ratings, child._stamps = self._ratings[rows], self._stamps[rows]
        child._texts = {}
        child._finalize()
        child._mapping_fingerprint = self.mapping_fingerprint
        if self._texts and len(child._keys):
            text_keys = np.fromiter(self._texts.keys(), dtype=np.int64, count=len(self._texts))
            pos = np.minimum(np.searchsorted(child._keys, text_keys), len(child._keys) - 1)
            present = child._keys[pos] == text_keys
            child._texts = {int(k): self._texts[int(k)] for k in text_keys[present]}
        return child

    def _check_user_id(self, user_id: int) -> int:
        if isinstance(user_id, bool) or not isinstance(user_id, Integral) or not 0 <= user_id < self.num_users:
            raise UnknownUserError(f"Chỉ số user không hợp lệ: {user_id}.", user_id=user_id)
        return int(user_id)

    def _user_slice(self, user_id: int) -> slice:
        user = self._check_user_id(user_id)
        return slice(int(self._indptr[user]), int(self._indptr[user + 1]))

    # ------------------------------------------------------------ thuộc tính

    @property
    def interactions(self) -> Tuple[Interaction, ...]:
        """Interaction với id là chỉ số int (dựng lười lần đầu truy cập)."""
        if self._interactions is None:
            self._interactions = tuple(
                Interaction(user_id=u, item_id=i, rating=r, timestamp=t)
                for u, i, r, t in zip(
                    self._users.tolist(), self._items.tolist(), self._ratings.tolist(), self._stamps.tolist()
                )
            )
        return self._interactions

    @property
    def records(self) -> Tuple[InteractionRecord, ...]:
        """InteractionRecord với id thô, kèm text nếu có review (dựng lười)."""
        if self._records is None:
            self._records = tuple(
                InteractionRecord(
                    user_id=self.user_mapper.to_raw(u),
                    item_id=self.item_mapper.to_raw(i),
                    rating=r,
                    timestamp=t,
                    text=self._texts.get(k),
                )
                for u, i, r, t, k in zip(
                    self._users.tolist(),
                    self._items.tolist(),
                    self._ratings.tolist(),
                    self._stamps.tolist(),
                    self._keys.tolist(),
                )
            )
        return self._records

    @property
    def games(self) -> Mapping[str, GameItem]:
        return self._games_view

    @property
    def game_stats(self) -> Mapping[str, GameStats]:
        return MappingProxyType(self._game_stats)

    @property
    def num_users(self) -> int:
        return self.user_mapper.size

    @property
    def num_items(self) -> int:
        return self.item_mapper.size

    @property
    def num_interactions(self) -> int:
        return int(len(self._users))

    @property
    def num_reviews(self) -> int:
        return len(self._texts)

    @property
    def num_synthetic_items(self) -> int:
        """Số game chỉ xuất hiện trong interactions, không có trong catalog."""
        return len(self._synthetic_ids)

    @property
    def mapping_fingerprint(self) -> str:
        """Giống nhau giữa mọi split sinh ra từ cùng một dataset gốc."""
        if self._mapping_fingerprint is None:
            self._mapping_fingerprint = _sha256(
                self.user_mapper.fingerprint.encode(), b"|", self.item_mapper.fingerprint.encode()
            )
        return self._mapping_fingerprint

    @property
    def fingerprint(self) -> str:
        """Băm nội dung tương tác (user, item, rating, timestamp) cùng mapping."""
        if self._fingerprint is None:
            self._fingerprint = _sha256(
                self.mapping_fingerprint.encode(),
                np.ascontiguousarray(self._users).tobytes(),
                np.ascontiguousarray(self._items).tobytes(),
                np.ascontiguousarray(self._ratings).tobytes(),
                np.ascontiguousarray(self._stamps).tobytes(),
            )
        return self._fingerprint

    @property
    def review_fingerprint(self) -> str:
        """Băm toàn bộ review chữ (khóa cặp + nội dung)."""
        if self._review_fingerprint is None:
            digest = hashlib.sha256()
            for key in sorted(self._texts):
                digest.update(str(key).encode())
                digest.update(b"\x00")
                digest.update(self._texts[key].encode("utf-8"))
                digest.update(b"\x01")
            self._review_fingerprint = digest.hexdigest()
        return self._review_fingerprint

    # ------------------------------------------------------------------ truy vấn

    def get_game_by_raw_id(self, raw_game_id: str) -> GameItem:
        game = self._games.get(clean_id(raw_game_id))
        if game is None:
            raise UnknownItemError(f"Không tìm thấy game '{raw_game_id}'.", item_id=raw_game_id)
        return game

    def get_user_seen_items(self, user_id: int) -> frozenset[int]:
        return frozenset(self._items[self._user_slice(user_id)].tolist())

    def get_user_positive_items(self, user_id: int) -> frozenset[int]:
        window = self._user_slice(user_id)
        return frozenset(self._items[window][self._ratings[window] == 1.0].tolist())

    def get_review(self, user_id: str, item_id: str) -> str:
        user_idx = self.user_mapper._to_idx.get(clean_id(user_id))
        item_idx = self.item_mapper._to_idx.get(clean_id(item_id))
        if user_idx is None or item_idx is None:
            return ""
        return self._texts.get(user_idx * self._stride + item_idx, "")

    def iter_reviews(self) -> Iterator[ReviewRecord]:
        """ReviewRecord (id int, rating, timestamp, text) theo thứ tự (user, item)."""
        for key in sorted(self._texts):
            pos = int(np.searchsorted(self._keys, key))
            yield ReviewRecord(
                user_id=int(self._users[pos]),
                item_id=int(self._items[pos]),
                rating=float(self._ratings[pos]),
                timestamp=int(self._stamps[pos]),
                text=self._texts[key],
            )

    def add_reviews_batch(
        self,
        user_ids: Sequence[object],
        item_ids: Sequence[object],
        reviews: Sequence[object],
    ) -> int:
        """
        Gắn review cho cặp (user, item) ĐÃ CÓ tương tác trong dataset này.
        Cặp không có tương tác bị bỏ qua (ReviewRecord cần rating/timestamp của tương tác).
        Trùng khóa thì dòng sau ghi đè. Nên gọi trước split() để các split mang theo review.
        """
        added = 0
        user_lookup, item_lookup = self.user_mapper._to_idx, self.item_mapper._to_idx
        total = len(self._keys)
        for user_id, item_id, text in zip(user_ids, item_ids, reviews):
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            user_idx, item_idx = user_lookup.get(clean_id(user_id)), item_lookup.get(clean_id(item_id))
            if user_idx is None or item_idx is None:
                continue
            key = user_idx * self._stride + item_idx
            pos = int(np.searchsorted(self._keys, key))
            if pos < total and self._keys[pos] == key:
                self._texts[key] = text
                added += 1
        if added:
            self._review_fingerprint = None
            self._records = None
        return added

    def indexed_interactions(self) -> List[Tuple[int, int, float]]:
        """(user_id, item_id, rating) với id là chỉ số int."""
        return list(zip(self._users.tolist(), self._items.tolist(), self._ratings.tolist()))

    def get_sparse_matrix(self, positive_only: bool = True):
        """
        Ma trận CSR (num_users, num_items).
        positive_only=True (mặc định): chỉ các tương tác rating 1.0, giá trị 1.0.
        positive_only=False: giữ cả 0.0 (không lưu được trong CSR nên chỉ có ý nghĩa với positive).
        """
        from scipy.sparse import csr_matrix

        mask = self._ratings == 1.0 if positive_only else np.ones(len(self._ratings), dtype=bool)
        return csr_matrix(
            (self._ratings[mask], (self._users[mask], self._items[mask])),
            shape=(self.num_users, self.num_items),
            dtype=float,
        )

    def mapping_report(self) -> Dict[str, int]:
        interacted = {self.item_mapper.to_raw(i) for i in np.unique(self._items).tolist()}
        return {
            "Số user đã ánh xạ": self.num_users,
            "Số game đã ánh xạ": self.num_items,
            "Tương tác có game không nằm trong catalog": int(
                sum(1 for i in self._items.tolist() if self.item_mapper.to_raw(i) in self._synthetic_ids)
            ),
            "Game chưa có tương tác nào": len(set(self._games) - interacted),
            "Số dòng trùng (user, game) đã loại": self.num_duplicates_dropped,
        }

    # --------------------------------------------------------------------- chia

    def split(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        random_state: int = 42,
        strategy: str = "random",
    ) -> DatasetSplits:
        """
        Chia theo từng user thành train / validation / test (DatasetSplits của contract).

        - User có < 3 tương tác: toàn bộ vào train (không đủ để đánh giá).
        - User có >= 3: ít nhất 1 dòng mỗi tập, train luôn còn >= 1.
        - strategy="random": xáo trộn theo seed; "temporal": cũ nhất vào train, mới nhất vào test.
        - Cả ba split dùng chung user_mapper/item_mapper và mapping_fingerprint.
        """
        for name, value in (("val_ratio", val_ratio), ("test_ratio", test_ratio)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < float(value) < 1.0:
                raise InvalidParameterError(f"{name} phải nằm trong (0, 1), nhận {value!r}.")
        if float(val_ratio) + float(test_ratio) >= 1.0:
            raise InvalidParameterError("val_ratio + test_ratio phải < 1.")
        if isinstance(random_state, bool) or not isinstance(random_state, Integral):
            raise InvalidParameterError(f"random_state phải là int, nhận {random_state!r}.")
        if strategy not in ("random", "temporal"):
            raise InvalidParameterError(f"strategy phải là 'random' hoặc 'temporal', nhận {strategy!r}.")
        if self.num_interactions == 0:
            raise EmptyDatasetError("Không có tương tác để chia.")
        if strategy == "temporal" and not (self._stamps > 0).any():
            raise InvalidParameterError(
                "strategy='temporal' cần timestamp > 0. Hãy tạo lại Parquet (rebuild) để có cột timestamp."
            )

        rng = np.random.default_rng(int(random_state))
        train_parts: List[np.ndarray] = []
        val_parts: List[np.ndarray] = []
        test_parts: List[np.ndarray] = []
        users_train_only = 0
        for user in range(self.num_users):
            lo, hi = int(self._indptr[user]), int(self._indptr[user + 1])
            size = hi - lo
            if size == 0:
                continue
            rows = np.arange(lo, hi, dtype=np.int64)
            if size < 3:
                train_parts.append(rows)
                users_train_only += 1
                continue
            if strategy == "random":
                rows = rng.permutation(rows)
            else:
                rows = rows[np.argsort(self._stamps[rows], kind="stable")]
            n_test = max(1, int(round(size * float(test_ratio))))
            n_val = max(1, int(round(size * float(val_ratio))))
            while size - n_test - n_val < 1:
                if n_val > 1:
                    n_val -= 1
                else:
                    n_test -= 1
            train_end, val_end = size - n_test - n_val, size - n_test
            train_parts.append(rows[:train_end])
            val_parts.append(rows[train_end:val_end])
            test_parts.append(rows[val_end:])

        def join(parts: List[np.ndarray]) -> np.ndarray:
            return np.sort(np.concatenate(parts)) if parts else _EMPTY_INT

        train = self._derive(join(train_parts), "train")
        validation = self._derive(join(val_parts), "validation")
        test = self._derive(join(test_parts), "test")

        train_items = set(np.unique(train._items).tolist())
        report = {
            "strategy": strategy,
            "random_state": int(random_state),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio),
            "num_users": self.num_users,
            "num_items": self.num_items,
            "train_interactions": train.num_interactions,
            "validation_interactions": validation.num_interactions,
            "test_interactions": test.num_interactions,
            "users_train_only": users_train_only,
            "validation_cold_items": len(set(np.unique(validation._items).tolist()) - train_items),
            "test_cold_items": len(set(np.unique(test._items).tolist()) - train_items),
            "mapping_fingerprint": self.mapping_fingerprint,
            "train_fingerprint": train.fingerprint,
            "validation_fingerprint": validation.fingerprint,
            "test_fingerprint": test.fingerprint,
        }
        return DatasetSplits(train=train, validation=validation, test=test, report=report)


__all__ = ["IdMapper", "InteractionDataset"]
