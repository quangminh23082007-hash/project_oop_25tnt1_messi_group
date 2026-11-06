from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from importlib import import_module
from numbers import Integral, Real

import numpy as np

from src.contracts.dto import Interaction
from src.contracts.exceptions import (
    ConfigurationError,
    DataValidationError,
    EmptyDatasetError,
    ModelNotFittedError,
    TrainingError,
)
from src.contracts.interfaces import BaseRecommender, InteractionDataset, checked_indices
from src.evaluation.protocol import validate_interactions


__all__ = ["CornacMostPopAdapter"]


def _load_cornac():
    try:
        return import_module("cornac")
    except ImportError as error:
        raise ImportError(
            "Cornac is required for this benchmark. Install it in the project "
            "environment with: python -m pip install cornac"
        ) from error


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise DataValidationError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{name} must be a nonempty string.")
    return value


def _cornac_mapping(
    values: object,
    expected_ids: set[str],
    name: str,
) -> dict[str, int]:
    if not isinstance(values, Mapping) or set(values) != expected_ids:
        raise DataValidationError(f"Cornac {name} does not match the positive train IDs.")

    normalized = {
        raw_id: _integer(index, f"Cornac {name} index")
        for raw_id, index in values.items()
    }

    if set(normalized.values()) != set(range(len(normalized))):
        raise DataValidationError(f"Cornac {name} must contain unique contiguous indices.")
    return normalized


def _cornac_scores(raw_scores: object, count: int) -> np.ndarray:
    if isinstance(raw_scores, (str, bytes, bytearray, Mapping, Set)):
        raise DataValidationError("Cornac must return scores in item-mapping order.")
    if isinstance(raw_scores, np.ndarray) and raw_scores.ndim != 1:
        raise DataValidationError("Cornac scores must be one-dimensional.")

    try:
        values = tuple(raw_scores)
    except TypeError as error:
        raise DataValidationError("Cornac scores must be an ordered iterable.") from error

    if len(values) != count:
        raise DataValidationError("Cornac score count does not match its item mapping.")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise DataValidationError("Cornac scores must be real numbers without bool.")

    try:
        scores = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise DataValidationError("Cornac scores must be finite real numbers.") from error

    if not np.isfinite(scores).all() or (scores < 0).any():
        raise DataValidationError("MostPop scores must be finite and nonnegative.")
    return scores


class CornacMostPopAdapter(BaseRecommender):
    model_kind = "cornac_most_pop"
    supports_callback = False

    def __init__(
        self,
        name: str = "CornacMostPop",
        positive_threshold: float = 1.0,
        seed: int = 42,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("name must be a nonempty string.")
        if (
            isinstance(positive_threshold, bool)
            or not isinstance(positive_threshold, Real)
            or positive_threshold != 1.0
        ):
            raise ConfigurationError("positive_threshold must be 1.0 for binary feedback.")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, Integral)
            or not 0 <= seed <= 2**32 - 1
        ):
            raise ConfigurationError("seed must be an integer between 0 and 2**32 - 1.")

        super().__init__(name.strip())
        self._seed = int(seed)
        self._invalidate()

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def positive_threshold(self) -> float:
        return 1.0

    def _invalidate(self) -> None:
        self._is_fitted = False
        self._num_users = 0
        self._num_items = 0
        self._mapping_fingerprint = None
        self._train_fingerprint = None
        self._model = None
        self._user_map: dict[str, int] = {}
        self._item_map: dict[str, int] = {}
        self._global_scores: np.ndarray | None = None

    def fit(
        self,
        train_dataset: InteractionDataset,
        *,
        validation_score=None,
        item_features=None,
        verbose: bool = False,
    ) -> CornacMostPopAdapter:
        self._invalidate()

        try:
            if validation_score is not None:
                raise ConfigurationError("Cornac MostPop does not support early stopping.")
            if item_features is not None:
                raise ConfigurationError("Cornac MostPop does not use item_features.")
            if not isinstance(verbose, bool):
                raise ConfigurationError("verbose must be a boolean.")

            interactions = self._training_rows(train_dataset)
            positives = sorted(
                (
                    (int(record.user_id), int(record.item_id))
                    for record in interactions
                    if record.rating == 1.0
                )
            )
            if not positives:
                raise EmptyDatasetError("Cornac MostPop requires positive train interactions.")

            rows = [(str(user_id), str(item_id), 1.0) for user_id, item_id in positives]

            cornac = _load_cornac()
            cornac_dataset = cornac.data.Dataset.from_uir(rows, seed=self.seed)
            model = cornac.models.MostPop(name=self.name)
            model.fit(cornac_dataset)

            user_map = _cornac_mapping(
                cornac_dataset.uid_map,
                {user_id for user_id, _, _ in rows},
                "uid_map",
            )
            item_map = _cornac_mapping(
                cornac_dataset.iid_map,
                {item_id for _, item_id, _ in rows},
                "iid_map",
            )

            reference_user = min(user_map.values())
            known_scores = _cornac_scores(model.score(reference_user), len(item_map))
            global_scores = np.zeros(int(train_dataset.num_items), dtype=float)

            for item_id, cornac_index in item_map.items():
                global_scores[int(item_id)] = known_scores[cornac_index]

            global_scores.flags.writeable = False

            self._model = model
            self._user_map = user_map
            self._item_map = item_map
            self._global_scores = global_scores
            self._num_users = int(train_dataset.num_users)
            self._num_items = int(train_dataset.num_items)
            self._mapping_fingerprint = train_dataset.mapping_fingerprint
            self._train_fingerprint = train_dataset.fingerprint
            self._is_fitted = True
            return self

        except BaseException as error:
            self._invalidate()
            if isinstance(error, (ConfigurationError, DataValidationError, ImportError)):
                raise
            if isinstance(error, Exception):
                raise TrainingError(f"Cornac MostPop training failed: {error}") from error
            raise

    def score_items(
        self,
        user_id: int,
        item_ids: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        if not self.is_fitted or self._global_scores is None:
            raise ModelNotFittedError("Call fit before score_items.")

        self._check_user(user_id)
        indices = checked_indices(item_ids, self.num_items)
        return self._global_scores[indices].copy()

    def config(self) -> dict[str, object]:
        return {
            "algorithm": "cornac.models.MostPop",
            "random_seed": self.seed,
            "positive_threshold": self.positive_threshold,
        }

    @staticmethod
    def _training_rows(
        dataset: InteractionDataset,
    ) -> tuple[Interaction, ...]:
        split_name = _text(getattr(dataset, "split_name", None), "train.split_name")
        if split_name != "train":
            raise DataValidationError("fit requires split_name='train'.")

        num_users = _integer(getattr(dataset, "num_users", None), "train.num_users")
        num_items = _integer(getattr(dataset, "num_items", None), "train.num_items")
        count = _integer(
            getattr(dataset, "num_interactions", None),
            "train.num_interactions",
        )
        if not num_users or not num_items or not count:
            raise EmptyDatasetError("Train must contain users, items and interactions.")

        for name in ("mapping_fingerprint", "fingerprint", "review_fingerprint"):
            _text(getattr(dataset, name, None), f"train.{name}")

        for name, expected in (("user_mapper", num_users), ("item_mapper", num_items)):
            mapper = getattr(dataset, name, None)
            size = _integer(getattr(mapper, "size", None), f"train.{name}.size")
            if size != expected:
                raise DataValidationError(f"train.{name}.size does not match dimensions.")

        records = getattr(dataset, "interactions", None)
        if (
            isinstance(records, (str, bytes, bytearray))
            or not isinstance(records, Sequence)
        ):
            raise DataValidationError("train.interactions must be an ordered sequence.")

        rows = validate_interactions(records, num_users, num_items, "train")
        if len(rows) != count:
            raise DataValidationError("train.num_interactions does not match its records.")
        return rows