from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from zipfile import BadZipFile

import numpy as np

from src.contracts.dto import ScoredItem
from src.contracts.exceptions import DataValidationError, InvalidParameterError
from src.evaluation.protocol import RankingEvaluationData


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise InvalidParameterError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        json.dumps(value, allow_nan=False)
        return value
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as error:
        raise DataValidationError(f"Cannot read valid metadata from {path}.") from error


def _raw_ids(value, name):
    if not isinstance(value, list) or not value:
        raise DataValidationError(f"{name} must be a nonempty list.")
    if any(not isinstance(raw_id, str) or not raw_id.strip() for raw_id in value):
        raise DataValidationError(f"{name} must contain nonempty original ID strings.")
    if len(set(value)) != len(value):
        raise DataValidationError(f"{name} must not contain duplicate IDs.")
    return tuple(value)


def _pair_array(value, name, num_users, num_items):
    if value.ndim != 2 or value.shape[1] != 2 or value.dtype.kind not in "iu":
        raise DataValidationError(f"{name} must be an integer array with shape (N, 2).")
    if value.size and ((value < 0).any() or (value[:, 0] >= num_users).any()
                       or (value[:, 1] >= num_items).any()):
        raise DataValidationError(f"{name} contains an index outside the original mapping.")
    array = np.ascontiguousarray(value, dtype="<i8")
    codes = array[:, 0] * num_items + array[:, 1]
    order = None
    sorted_codes = codes
    if len(codes) > 1 and (codes[1:] < codes[:-1]).any():
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
    if len(sorted_codes) > 1 and (sorted_codes[1:] == sorted_codes[:-1]).any():
        raise DataValidationError(f"{name} contains duplicate user/item pairs.")
    return array, order, sorted_codes


def _subset(left, right, name):
    for start in range(0, len(left), 250_000):
        chunk = left[start:start + 250_000]
        positions = np.searchsorted(right, chunk)
        valid = positions < len(right)
        if not valid.all() or not np.array_equal(right[positions], chunk):
            raise DataValidationError(f"{name} is not a subset of its observed history.")


def _disjoint(left, right, name):
    for start in range(0, len(left), 250_000):
        chunk = left[start:start + 250_000]
        positions = np.searchsorted(right, chunk)
        valid = positions < len(right)
        if valid.any() and (right[positions[valid]] == chunk[valid]).any():
            raise DataValidationError(f"{name} contains overlapping train/holdout pairs.")


def _seen_indices(values, num_items):
    if isinstance(values, (str, bytes, Mapping)):
        raise InvalidParameterError("seen_items must contain internal item indices.")
    try:
        result = frozenset(_integer(value, "seen item") for value in values)
    except TypeError as error:
        raise InvalidParameterError("seen_items must be an iterable of item indices.") from error
    if any(value >= num_items for value in result):
        raise InvalidParameterError("seen_items contains an index outside the original mapping.")
    return result


class LegacyPrepared:
    @classmethod
    def load(cls, folder):
        folder = Path(folder).expanduser().resolve()
        mapping = _read_json(folder / "mappings.json")
        report = _read_json(folder / "report.json")
        catalog = _read_json(folder / "catalog.json")
        user_ids = _raw_ids(mapping.get("user_ids"), "user_ids")
        item_ids = _raw_ids(mapping.get("item_ids"), "item_ids")
        if set(mapping) != {"user_ids", "item_ids"}:
            raise DataValidationError("Original mappings must contain only user_ids and item_ids.")
        if len(user_ids) * len(item_ids) > np.iinfo(np.int64).max:
            raise DataValidationError("Original mapping dimensions exceed the supported index range.")
        try:
            with np.load(folder / "dataset.npz", allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
        except (OSError, ValueError, TypeError, EOFError, BadZipFile) as error:
            raise DataValidationError("Cannot read the original prepared dataset.npz.") from error
        required = {"train", "train_seen", "validation", "validation_seen", "test"}
        if not required.issubset(arrays):
            raise DataValidationError(f"Prepared arrays are missing: {sorted(required - set(arrays))}.")
        digest = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode())
        normalized = {}
        codes = {}
        for name in sorted(arrays):
            array, order, pair_codes = _pair_array(arrays.pop(name), name, len(user_ids), len(item_ids))
            digest.update(name.encode())
            digest.update(str(array.shape).encode())
            digest.update(memoryview(array))
            sorted_array = array if order is None else array[order]
            sorted_array.flags.writeable = False
            normalized[name] = sorted_array
            codes[name] = pair_codes
        fingerprint = digest.hexdigest()
        if report.get("fingerprint") != fingerprint:
            raise DataValidationError("Prepared data/mappings do not match their original fingerprint.")
        if not len(normalized["train"]) or not len(normalized["train_seen"]):
            raise DataValidationError("Prepared train positives and observed history must not be empty.")
        _subset(codes["train"], codes["train_seen"], "train positives")
        _subset(codes["train_seen"], codes["validation_seen"], "train_seen")
        _subset(codes["validation"], codes["validation_seen"], "validation positives")
        _disjoint(codes["validation"], codes["train_seen"], "validation")
        _disjoint(codes["test"], codes["validation_seen"], "test")
        obj = cls()
        obj.folder = folder
        obj.user_ids = user_ids
        obj.item_ids = item_ids
        obj.num_users = len(user_ids)
        obj.num_items = len(item_ids)
        obj.data_fingerprint = fingerprint
        obj.arrays = MappingProxyType(normalized)
        obj.train = normalized["train"]
        obj.report = report
        obj.catalog = catalog
        obj.source_files = {name: {"path": str(folder / name), "sha256": _file_hash(folder / name)}
                            for name in ("dataset.npz", "mappings.json", "report.json", "catalog.json")}
        return obj

    @staticmethod
    def _items(array, user_id):
        lo = np.searchsorted(array[:, 0], user_id, side="left")
        hi = np.searchsorted(array[:, 0], user_id, side="right")
        return frozenset(int(item) for item in array[lo:hi, 1])

    def context(self, split, *, max_users=0, seed=42):
        if split not in ("validation", "test"):
            raise InvalidParameterError("split must be validation or test.")
        max_users = _integer(max_users, "max_users")
        seed = _integer(seed, "seed")
        if seed > 2**32 - 1:
            raise InvalidParameterError("seed must not exceed 2**32 - 1.")
        targets = self.arrays[split]
        history = self.arrays["validation_seen" if split == "test" else "train_seen"]
        users = np.unique(targets[:, 0])
        if max_users and len(users) > max_users:
            users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))
        return RankingEvaluationData(
            split_name=split,
            num_users=self.num_users,
            num_items=self.num_items,
            user_ids=tuple(int(user_id) for user_id in users),
            get_relevant_items=lambda user_id: self._items(targets, user_id),
            get_seen_items=lambda user_id: self._items(history, user_id),
            has_positive_train=lambda user_id: bool(self._items(self.train, user_id)),
            data_fingerprint=self.data_fingerprint,
        )

    def metadata(self):
        return {"folder": str(self.folder), "data_fingerprint": self.data_fingerprint,
                "num_users": self.num_users, "num_items": self.num_items,
                "array_rows": {name: len(array) for name, array in self.arrays.items()},
                "target_users": {name: len(np.unique(self.arrays[name][:, 0]))
                                 for name in ("validation", "test")},
                "eligible_users": {name: len(np.intersect1d(np.unique(self.arrays[name][:, 0]),
                                                           np.unique(self.train[:, 0]), assume_unique=True))
                                   for name in ("validation", "test")},
                "source_files": self.source_files, "report": self.report}


class SavedModelRanker:
    @classmethod
    def load(cls, folder, prepared, *, name=None):
        if not isinstance(prepared, LegacyPrepared):
            raise DataValidationError("prepared must be a verified LegacyPrepared instance.")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise InvalidParameterError("name must be a nonempty string.")
        folder = Path(folder).expanduser().resolve()
        info = _read_json(folder / "model_info.json")
        if info.get("data_fingerprint") != prepared.data_fingerprint:
            raise DataValidationError("Saved model and original prepared fingerprint do not match.")
        if (info.get("user_ids") != list(prepared.user_ids)
                or info.get("item_ids") != list(prepared.item_ids)):
            raise DataValidationError("Saved model and original prepared ID order do not match.")
        from src.models.pretrained import PretrainedRecommender

        started = perf_counter()
        recommender = PretrainedRecommender.load(folder, allow_cold_start_fallback=False)
        elapsed = perf_counter() - started
        obj = cls()
        obj._recommender = recommender
        obj._info = {key: value for key, value in info.items()
                     if key not in {"user_ids", "item_ids", "catalog", "history"}}
        obj._item_ids = prepared.item_ids
        obj._item_map = {raw_id: index for index, raw_id in enumerate(prepared.item_ids)}
        obj._user_ids = prepared.user_ids
        obj._prepared = prepared
        obj._verified_users = set()
        obj.folder = folder
        obj.name = name.strip() if name is not None else recommender.model_name
        obj.model_type = info.get("model_type")
        obj.num_users = prepared.num_users
        obj.num_items = prepared.num_items
        obj.data_fingerprint = prepared.data_fingerprint
        obj.load_seconds = elapsed
        obj.fit_seconds = 0.0
        obj._metadata_hash = _file_hash(folder / "model_info.json")
        return obj

    def rank_items(self, user_id, *, top_k, seen_items):
        user_id = _integer(user_id, "user_id")
        top_k = _integer(top_k, "top_k", 1)
        if user_id >= self.num_users:
            raise InvalidParameterError("user_id is outside the original mapping.")
        seen = _seen_indices(seen_items, self.num_items)
        training_seen = self._prepared._items(self._prepared.arrays["train_seen"], user_id)
        if not training_seen.issubset(seen):
            raise DataValidationError("Ranking history must include all original TRAIN observations.")
        if user_id not in self._verified_users:
            original_seen = {game.item_id for game in self._recommender.get_user_played_games(self._user_ids[user_id])}
            expected_seen = {self._item_ids[index] for index in training_seen}
            if original_seen != expected_seen:
                raise DataValidationError("Saved model training history does not match original prepared history.")
            self._verified_users.add(user_id)
        rows = self._recommender.recommend_games(
            self._user_ids[user_id], top_k=top_k, exclude_played=True,
            extra_seen=tuple(self._item_ids[index] for index in sorted(seen)),
        )
        result = []
        used = set()
        for row in rows:
            item = self._item_map.get(row.game_id)
            score = row.predicted_score
            if (item is None or item in seen or item in used or row.is_fallback
                    or isinstance(score, bool) or not isinstance(score, Real) or not np.isfinite(score)):
                raise DataValidationError("Saved model returned an invalid personalized ranking.")
            used.add(item)
            result.append(ScoredItem(item, float(score)))
        return tuple(result)

    def metadata(self):
        info = self._info
        return {"folder": str(self.folder), "name": self.name,
                "class": "src.models.pretrained.PretrainedRecommender",
                "model_type": info.get("model_type"), "format_version": info.get("format_version"),
                "representation": info.get("representation"), "data_fingerprint": self.data_fingerprint,
                "weights_sha256": info.get("weights_sha256"), "model_info_sha256": self._metadata_hash,
                "num_users": self.num_users, "num_items": self.num_items,
                "best_epoch": info.get("best_epoch"), "config": info.get("config"),
                "data_config": info.get("data_config"), "training_info": info.get("training_info"),
                "feature_info": info.get("feature_info"), "limitations": info.get("data_limitations", []),
                "load_seconds": self.load_seconds, "fit_seconds": self.fit_seconds,
                "retrained_this_run": False}


__all__ = ["LegacyPrepared", "SavedModelRanker"]
