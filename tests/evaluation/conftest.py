import hashlib
import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np

from src.contracts.dto import DatasetSplits, GameItem, Interaction, ReviewRecord
from src.contracts.exceptions import InvalidParameterError, UnknownEntityError


def fingerprint(value):
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FixtureMapper:
    raw_ids: tuple[str, ...]

    @property
    def size(self):
        return len(self.raw_ids)

    def contains_raw(self, raw_id):
        return raw_id in self.raw_ids

    def to_idx(self, raw_id):
        try:
            return self.raw_ids.index(raw_id)
        except ValueError as error:
            raise UnknownEntityError(
                f"Unknown fixture ID: {raw_id}."
            ) from error

    def to_raw(self, index):
        if (
            isinstance(index, bool)
            or not isinstance(index, Integral)
            or not 0 <= index < self.size
        ):
            raise InvalidParameterError("Invalid fixture index.")
        return self.raw_ids[int(index)]


@dataclass(frozen=True)
class FixtureDataset:
    split_name: str
    interactions: tuple[Interaction, ...]
    user_mapper: FixtureMapper
    item_mapper: FixtureMapper
    reviews: tuple[ReviewRecord, ...] = ()

    @property
    def num_users(self):
        return self.user_mapper.size

    @property
    def num_items(self):
        return self.item_mapper.size

    @property
    def num_interactions(self):
        return len(self.interactions)

    @property
    def mapping_fingerprint(self):
        return fingerprint([
            self.user_mapper.raw_ids,
            self.item_mapper.raw_ids,
        ])

    @property
    def fingerprint(self):
        rows = sorted(
            (r.user_id, r.item_id, r.rating, r.timestamp)
            for r in self.interactions
        )
        return fingerprint([
            self.mapping_fingerprint,
            self.split_name,
            rows,
        ])

    @property
    def review_fingerprint(self):
        rows = sorted(
            (r.user_id, r.item_id, r.rating, r.timestamp, r.text)
            for r in self.reviews
        )
        return fingerprint([
            self.mapping_fingerprint,
            self.split_name,
            rows,
        ])

    def get_user_seen_items(self, user_id):
        return frozenset(
            r.item_id
            for r in self.interactions
            if r.user_id == user_id
        )

    def get_user_positive_items(self, user_id):
        return frozenset(
            r.item_id
            for r in self.interactions
            if r.user_id == user_id and r.rating == 1.0
        )

    def get_game_by_raw_id(self, raw_game_id):
        index = self.item_mapper.to_idx(raw_game_id)
        return GameItem(raw_game_id, f"Fixture Game {index}")

    def iter_reviews(self):
        return iter(self.reviews)


def make_splits():
    users = FixtureMapper(tuple(f"user-{i}" for i in range(5)))
    items = FixtureMapper(tuple(f"game-{i}" for i in range(6)))

    rows = {
        "train": (
            (0, 0, 1), (0, 1, 0),
            (1, 0, 1), (1, 2, 0),
            (2, 3, 0), (3, 4, 1),
            (4, 5, 1),
        ),
        "validation": (
            (0, 4, 1), (0, 5, 0),
            (1, 3, 1), (1, 4, 0),
            (2, 1, 1), (4, 1, 0),
        ),
        "test": (
            (0, 2, 1), (0, 3, 1),
            (1, 1, 1), (2, 0, 1),
            (3, 2, 0), (4, 0, 1),
        ),
    }

    reviews = (
        ReviewRecord(
            0, 0, 1.0, 0,
            "Positive fixture review",
        ),
        ReviewRecord(
            0, 1, 0.0, 0,
            "Negative fixture review",
        ),
    )

    datasets = {
        name: FixtureDataset(
            name,
            tuple(Interaction(*row) for row in values),
            users,
            items,
            reviews if name == "train" else (),
        )
        for name, values in rows.items()
    }

    return DatasetSplits(
        datasets["train"],
        datasets["validation"],
        datasets["test"],
        {
            "source": "synthetic_test_fixture",
            "split_method": "fixed_fixture",
        },
    )


class FixedScoreModel:
    def __init__(self, train_dataset):
        self.name = "FixedScoreModel"
        self.is_fitted = True
        self.num_users = train_dataset.num_users
        self.num_items = train_dataset.num_items
        self.mapping_fingerprint = train_dataset.mapping_fingerprint
        self.train_fingerprint = train_dataset.fingerprint
        self.scores = {
            (0, 2): 5.0,
            (0, 3): 4.0,
            (1, 3): 5.0,
            (1, 4): 4.0,
            (4, 2): 5.0,
            (4, 3): 4.0,
        }
        self.calls = []
        self.output = None

    def score_items(self, user_id, item_ids):
        self.calls.append((user_id, tuple(item_ids)))

        if self.output is not None:
            return self.output(user_id, item_ids)

        return np.array([
            self.scores.get((user_id, item), 0.0)
            for item in item_ids
        ])

def legacy_fingerprint(mapping, arrays):
    digest = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode())
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name], dtype="<i8")
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_legacy_fixture(directory):
    root = Path(directory)
    prepared_dir = root / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "user_ids": ["fixture-user-0", "fixture-user-1"],
        "item_ids": [f"g{i}" for i in range(6)],
    }
    train = np.array(
        [[0, 0], [0, 1], [1, 2], [1, 3], [1, 4], [1, 5]],
        dtype=np.int64,
    )
    train_seen = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 2], [1, 3], [1, 4], [1, 5]],
        dtype=np.int64,
    )
    validation = np.array([[0, 3], [1, 0]], dtype=np.int64)
    arrays = {
        "train": train,
        "train_seen": train_seen,
        "validation": validation,
        "validation_seen": np.unique(np.vstack([train_seen, validation]), axis=0),
        "test": np.array([[0, 4], [1, 1]], dtype=np.int64),
    }
    source_fingerprint = legacy_fingerprint(mapping, arrays)
    data_config = {
        "split": "random",
        "seed": 42,
        "min_user": 1,
        "min_item": 1,
        "user_fraction": 1.0,
        "allow_unverified_users": True,
        "verified_user_ids": False,
        "train_fraction": 0.7,
        "validation_fraction": 0.15,
        "train_end_ms": None,
        "validation_end_ms": None,
    }
    catalog = {
        raw_id: {"title": f"Synthetic test game {index}"}
        for index, raw_id in enumerate(mapping["item_ids"])
    }
    report = {
        "format_version": 3,
        "n_users": len(mapping["user_ids"]),
        "n_items": len(mapping["item_ids"]),
        "fingerprint": source_fingerprint,
        "config": data_config,
        "limitations": ["Synthetic fixture only; never a measured production result."],
        "protocol": {
            "mapping_and_core": "train positives only",
            "validation_seen": "union of train and validation observations",
        },
    }
    for name, value in (
        ("mappings.json", mapping),
        ("catalog.json", catalog),
        ("report.json", report),
    ):
        (prepared_dir / name).write_text(json.dumps(value), encoding="utf-8")
    np.savez_compressed(prepared_dir / "dataset.npz", **arrays)
    bundles = {}
    for kind, model_type in (
        ("bpr", "BPRMatrixFactorization"),
        ("text_bpr", "TextBPRMatrixFactorization"),
    ):
        bundle = root / "models" / kind / "cpu"
        bundle.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            bundle / "weights.npz",
            P=np.eye(2, dtype=float),
            R=np.array([[6, 0], [5, 1], [4, 2], [3, 3], [2, 4], [1, 5]], dtype=float),
            popularity=np.ones(6, dtype=float),
            train_seen=train_seen,
        )
        info = {
            "format_version": 3,
            "representation": "compact",
            "model_type": model_type,
            "config": {"factors": 2},
            "user_ids": mapping["user_ids"],
            "item_ids": mapping["item_ids"],
            "catalog": catalog,
            "data_fingerprint": source_fingerprint,
            "data_config": data_config,
            "data_limitations": report["limitations"],
            "best_epoch": 1,
            "history": [],
            "feature_info": {"fixture": "synthetic effective item factors"} if kind == "text_bpr" else None,
            "training_info": {"fixture": "predefined factors; no training was run"},
            "weights_sha256": hashlib.sha256((bundle / "weights.npz").read_bytes()).hexdigest(),
        }
        (bundle / "model_info.json").write_text(json.dumps(info), encoding="utf-8")
        bundles[kind] = bundle
    return {
        "prepared_dir": prepared_dir,
        "bundles": bundles,
        "arrays": arrays,
        "user_ids": tuple(mapping["user_ids"]),
        "item_ids": tuple(mapping["item_ids"]),
        "fingerprint": source_fingerprint,
    }

