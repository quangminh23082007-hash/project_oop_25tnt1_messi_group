"""IdMapper và InteractionDataset (chỉ đọc) theo Contracts v1."""

from __future__ import annotations

import hashlib
from numbers import Integral
from types import MappingProxyType
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from src.contracts.dto import GameItem, Interaction, InteractionRecord, ReviewRecord
from src.contracts.exceptions import (
    DataValidationError,
    UnknownItemError,
    UnknownUserError,
)
from src.data.entities import GameStats
from src.data.parsing import clean_id, text_or_none


def _sha256(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


class IdMapper:
    """
    Ánh xạ id thô (str) sang chỉ số 0..N-1 và ngược lại. Bất biến sau khi tạo.

    Contract: mapping được fit từ train rồi đóng băng; validation/test không tự mở rộng mapping.
    """

    def __init__(
        self,
        ordered_raw_ids: Sequence[str],
        *,
        missing_error: type[Exception],
        kind: str,
    ) -> None:
        ids = tuple(ordered_raw_ids)
        self._to_idx: Dict[str, int] = {raw: index for index, raw in enumerate(ids)}
        if len(self._to_idx) != len(ids):
            raise DataValidationError(f"Danh sách id {kind} có phần tử trùng lặp.")
        self._to_raw: Tuple[str, ...] = ids
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
        """Id thô theo thứ tự chỉ số: raw_ids[i] là id thô của chỉ số i."""
        return self._to_raw

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


class Catalog:
    """Metadata game dùng chung cho mọi split (chỉ đọc)."""

    def __init__(
        self,
        games: Mapping[str, GameItem],
        stats: Optional[Mapping[str, GameStats]] = None,
        extra_item_ids: Iterable[str] = (),
    ) -> None:
        if not isinstance(games, Mapping):
            raise DataValidationError("games phải là dict game_id -> GameItem.")
        games_map: Dict[str, GameItem] = {}
        for game in games.values():
            if not isinstance(game, GameItem):
                raise DataValidationError(f"Catalog chỉ chứa GameItem, nhận {type(game).__name__}.")
            games_map[game.item_id] = game
        synthetic: Set[str] = set()
        for item_id in extra_item_ids:
            if item_id not in games_map:
                games_map[item_id] = GameItem(item_id=item_id, title=item_id)
                synthetic.add(item_id)
        self.games: Dict[str, GameItem] = games_map
        self.view: Mapping[str, GameItem] = MappingProxyType(games_map)
        self.synthetic_ids: frozenset[str] = frozenset(synthetic)
        self.stats: Mapping[str, GameStats] = MappingProxyType(dict(stats or {}))


def _freeze(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


class InteractionDataset:
    """
    Một split (train / validation / test) đã gắn chỉ số nguyên, thỏa Protocol của contract.

    - Nhãn rating nhị phân 1.0 (tích cực) / 0.0 (tiêu cực).
    - Mỗi cặp (user, item) một dòng. Dữ liệu và mapping chỉ đọc sau khi tạo.
    - Review text đi theo đúng dòng tương tác được giữ; split nào chứa dòng đó thì mang theo text.
    - Tạo bằng DataPipeline.prepare() để ba split dùng chung mapping fit từ train.
      Constructor này nhận danh sách InteractionRecord để dùng độc lập và trong test.
    """

    def __init__(
        self,
        records: Sequence[InteractionRecord],
        games: Mapping[str, GameItem],
        *,
        game_stats: Optional[Mapping[str, GameStats]] = None,
        split_name: str = "train",
        user_mapper: Optional[IdMapper] = None,
        item_mapper: Optional[IdMapper] = None,
    ) -> None:
        if records is None or isinstance(records, (str, bytes)):
            raise DataValidationError("records phải là một dãy InteractionRecord.")
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

        if user_mapper is None:
            user_mapper = IdMapper(
                sorted({clean_id(r.user_id) for r in parsed}), missing_error=UnknownUserError, kind="user"
            )
        if item_mapper is None:
            item_mapper = IdMapper(
                sorted({clean_id(r.item_id) for r in parsed}), missing_error=UnknownItemError, kind="item"
            )
        catalog = Catalog(games, game_stats, extra_item_ids=item_mapper.raw_ids)

        count = len(parsed)
        users = np.empty(count, dtype=np.int64)
        items = np.empty(count, dtype=np.int64)
        ratings = np.empty(count, dtype=np.float64)
        stamps = np.empty(count, dtype=np.int64)
        row_texts: List[Optional[str]] = [None] * count
        stride = max(item_mapper.size, 1)
        for pos, record in enumerate(parsed):
            user_key, item_key = clean_id(record.user_id), clean_id(record.item_id)
            user_idx, item_idx = user_mapper._to_idx.get(user_key), item_mapper._to_idx.get(item_key)
            if user_idx is None:
                raise UnknownUserError(f"Không tìm thấy user '{user_key}'.", user_id=user_key)
            if item_idx is None:
                raise UnknownItemError(f"Không tìm thấy item '{item_key}'.", item_id=item_key)
            users[pos], items[pos] = user_idx, item_idx
            ratings[pos], stamps[pos] = float(record.rating), int(record.timestamp)
            row_texts[pos] = text_or_none(record.text)

        # Trùng (user, item): giữ dòng đầu tiên theo thứ tự đầu vào (argsort ổn định).
        keys = users * stride + items
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        is_first = np.ones(count, dtype=bool)
        is_first[1:] = sorted_keys[1:] != sorted_keys[:-1]
        keep = order[is_first]
        self.num_duplicates_dropped = count - len(keep)
        self.num_conflicting_pairs = 0
        if count:
            starts = np.flatnonzero(is_first)
            sorted_ratings = ratings[order]
            lows = np.minimum.reduceat(sorted_ratings, starts)
            highs = np.maximum.reduceat(sorted_ratings, starts)
            self.num_conflicting_pairs = int((lows != highs).sum())

        texts = {
            int(keys[pos]): row_texts[pos] for pos in keep.tolist() if row_texts[pos] is not None
        }
        self._init_state(
            split_name.strip(), catalog, user_mapper, item_mapper,
            users[keep], items[keep], ratings[keep], stamps[keep], texts,
        )

    # ------------------------------------------------------------------ nội bộ

    @classmethod
    def from_arrays(
        cls,
        *,
        split_name: str,
        catalog: Catalog,
        user_mapper: IdMapper,
        item_mapper: IdMapper,
        users: np.ndarray,
        items: np.ndarray,
        ratings: np.ndarray,
        timestamps: np.ndarray,
        texts: Dict[int, str],
    ) -> "InteractionDataset":
        """
        Dựng split từ mảng đã ánh xạ và đã sắp tăng theo (user, item), không trùng cặp.
        `texts` khóa bằng user*max(num_items,1)+item. DataPipeline dùng hàm này.
        """
        dataset = object.__new__(cls)
        dataset.num_duplicates_dropped = 0
        dataset.num_conflicting_pairs = 0
        dataset._init_state(
            split_name, catalog, user_mapper, item_mapper,
            np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64),
            np.asarray(ratings, dtype=np.float64), np.asarray(timestamps, dtype=np.int64), texts,
        )
        return dataset

    def _init_state(
        self,
        split_name: str,
        catalog: Catalog,
        user_mapper: IdMapper,
        item_mapper: IdMapper,
        users: np.ndarray,
        items: np.ndarray,
        ratings: np.ndarray,
        stamps: np.ndarray,
        texts: Dict[int, str],
    ) -> None:
        if len(users) and (
            users.min() < 0 or users.max() >= user_mapper.size or items.min() < 0 or items.max() >= item_mapper.size
        ):
            raise DataValidationError("Chỉ số nằm ngoài mapping của dataset.")
        self.split_name = split_name
        self._catalog = catalog
        self.user_mapper, self.item_mapper = user_mapper, item_mapper
        self._stride = max(item_mapper.size, 1)
        self._users = _freeze(np.array(users, dtype=np.int64, copy=True))
        self._items = _freeze(np.array(items, dtype=np.int64, copy=True))
        self._ratings = _freeze(np.array(ratings, dtype=np.float64, copy=True))
        self._stamps = _freeze(np.array(stamps, dtype=np.int64, copy=True))
        self._keys = _freeze(self._users * self._stride + self._items)
        counts = np.bincount(self._users, minlength=user_mapper.size)
        indptr = np.zeros(user_mapper.size + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        self._indptr = _freeze(indptr)
        self._texts: Dict[int, str] = dict(texts)
        self._interactions: Optional[Tuple[Interaction, ...]] = None
        self._records: Optional[Tuple[InteractionRecord, ...]] = None
        self._fingerprint: Optional[str] = None
        self._review_fingerprint: Optional[str] = None
        self._mapping_fingerprint: Optional[str] = None

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
        """InteractionRecord với id thô, kèm text nếu có (dựng lười)."""
        if self._records is None:
            self._records = tuple(
                InteractionRecord(user_id=u, item_id=i, rating=r, timestamp=t, text=x)
                for u, i, r, t, x in self.iter_raw_rows()
            )
        return self._records

    def iter_raw_rows(self) -> Iterator[Tuple[str, str, float, int, Optional[str]]]:
        """(user_id thô, item_id thô, rating, timestamp, text hoặc None) theo thứ tự (user, item)."""
        to_user, to_item = self.user_mapper.raw_ids, self.item_mapper.raw_ids
        for u, i, r, t, k in zip(
            self._users.tolist(), self._items.tolist(), self._ratings.tolist(),
            self._stamps.tolist(), self._keys.tolist(),
        ):
            yield to_user[u], to_item[i], r, t, self._texts.get(k)

    @property
    def games(self) -> Mapping[str, GameItem]:
        return self._catalog.view

    @property
    def game_stats(self) -> Mapping[str, GameStats]:
        return self._catalog.stats

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
    def num_positive(self) -> int:
        return int((self._ratings == 1.0).sum())

    @property
    def num_negative(self) -> int:
        return int((self._ratings == 0.0).sum())

    @property
    def num_active_users(self) -> int:
        """Số user có ít nhất một tương tác trong split này."""
        return int((np.diff(self._indptr) > 0).sum())

    @property
    def num_reviews(self) -> int:
        return len(self._texts)

    @property
    def num_synthetic_items(self) -> int:
        """Số game có trong mapping nhưng không có trong catalog gốc."""
        return len(self._catalog.synthetic_ids)

    @property
    def mapping_fingerprint(self) -> str:
        """Giống nhau giữa mọi split cùng một lần chuẩn bị dữ liệu."""
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
                self._users.tobytes(),
                self._items.tobytes(),
                self._ratings.tobytes(),
                self._stamps.tobytes(),
            )
        return self._fingerprint

    @property
    def review_fingerprint(self) -> str:
        """Băm review của split: cặp user–game, nhãn, timestamp và nội dung text."""
        if self._review_fingerprint is None:
            digest = hashlib.sha256()
            for key in sorted(self._texts):
                pos = int(np.searchsorted(self._keys, key))
                digest.update(
                    f"{int(self._users[pos])}|{int(self._items[pos])}|"
                    f"{int(self._ratings[pos])}|{int(self._stamps[pos])}|".encode()
                )
                digest.update(self._texts[key].encode("utf-8"))
                digest.update(b"\x01")
            self._review_fingerprint = digest.hexdigest()
        return self._review_fingerprint

    # ------------------------------------------------------------------ truy vấn

    def get_game_by_raw_id(self, raw_game_id: str) -> GameItem:
        game = self._catalog.games.get(clean_id(raw_game_id))
        if game is None:
            raise UnknownItemError(f"Không tìm thấy game '{raw_game_id}'.", item_id=raw_game_id)
        return game

    def get_user_seen_items(self, user_id: int) -> frozenset[int]:
        """Mọi item user có tương tác trong split này, cả tích cực lẫn tiêu cực."""
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
        """ReviewRecord (id int, rating, timestamp, text không rỗng) của riêng split này."""
        for key in sorted(self._texts):
            pos = int(np.searchsorted(self._keys, key))
            yield ReviewRecord(
                user_id=int(self._users[pos]),
                item_id=int(self._items[pos]),
                rating=float(self._ratings[pos]),
                timestamp=int(self._stamps[pos]),
                text=self._texts[key],
            )

    def indexed_interactions(self) -> List[Tuple[int, int, float]]:
        """(user_id, item_id, rating) với id là chỉ số int."""
        return list(zip(self._users.tolist(), self._items.tolist(), self._ratings.tolist()))

    def get_sparse_matrix(self, positive_only: bool = True):
        """
        Ma trận CSR (num_users, num_items).
        positive_only=True (mặc định): chỉ tương tác rating 1.0, giá trị 1.0.
        positive_only=False: thêm cả 0.0 (CSR không phân biệt được 0 tường minh với ô trống).
        """
        from scipy.sparse import csr_matrix

        mask = self._ratings == 1.0 if positive_only else np.ones(len(self._ratings), dtype=bool)
        return csr_matrix(
            (self._ratings[mask], (self._users[mask], self._items[mask])),
            shape=(self.num_users, self.num_items),
            dtype=float,
        )

    def mapping_report(self) -> Dict[str, int]:
        in_mapping = set(self.item_mapper.raw_ids)
        return {
            "Số user đã ánh xạ": self.num_users,
            "Số game đã ánh xạ (từ train)": self.num_items,
            "Game trong catalog nhưng ngoài mapping": len(set(self._catalog.games) - in_mapping),
            "Game ánh xạ nhưng không có trong catalog": len(self._catalog.synthetic_ids),
            "Số dòng trùng (user, game) đã loại": self.num_duplicates_dropped,
            "Số cặp trùng có nhãn mâu thuẫn": self.num_conflicting_pairs,
        }


__all__ = ["Catalog", "IdMapper", "InteractionDataset"]
