import hashlib
import json
from dataclasses import dataclass
from numbers import Integral

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