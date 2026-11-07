"""Shared records from Contracts v1 (2026-11-05), defined only here."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real
from typing import Any
import numpy as np
from .exceptions import ConfigurationError, DataValidationError


def _index(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise DataValidationError(f"{name} must be a nonnegative integer")


def _label(value):
    if isinstance(value, bool) or not isinstance(value, Real) or value not in (0.0, 1.0):
        raise DataValidationError("rating must be numeric 0.0 or 1.0")


@dataclass(frozen=True)
class InteractionRecord:
    user_id: str
    item_id: str
    rating: float
    timestamp: int = 0

    def __post_init__(self):
        for name in ("user_id", "item_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DataValidationError(f"{name} must be a nonempty raw ID")
            object.__setattr__(self, name, value.strip())
        _label(self.rating)
        _index(self.timestamp, "timestamp")


@dataclass(frozen=True)
class Interaction:
    user_id: int
    item_id: int
    rating: float
    timestamp: int = 0

    def __post_init__(self):
        _index(self.user_id, "user_id")
        _index(self.item_id, "item_id")
        _label(self.rating)
        _index(self.timestamp, "timestamp")


@dataclass(frozen=True)
class ReviewRecord:
    user_id: int
    item_id: int
    rating: float
    timestamp: int
    text: str

    def __post_init__(self):
        Interaction(self.user_id, self.item_id, self.rating, self.timestamp)
        if not isinstance(self.text, str) or not self.text.strip():
            raise DataValidationError("ReviewRecord.text must contain nonblank source text")


@dataclass(frozen=True)
class GameItem:
    item_id: str
    title: str
    genres: tuple[str, ...] = ()
    price: float | None = None
    developer: str = "Unknown"

    def __post_init__(self):
        if not isinstance(self.item_id, str) or not self.item_id.strip():
            raise DataValidationError("item_id must be a nonempty raw ID")
        object.__setattr__(self, "item_id", self.item_id.strip())
        if not isinstance(self.title, str) or not self.title.strip():
            raise DataValidationError("title must be nonempty")
        if not isinstance(self.genres, tuple) or any(not isinstance(g, str) for g in self.genres):
            raise DataValidationError("genres must be a tuple of strings")
        if self.price is not None and (isinstance(self.price, bool)
                or not isinstance(self.price, Real) or not isfinite(self.price) or self.price < 0):
            raise DataValidationError("price must be finite and nonnegative, or None")


@dataclass(frozen=True)
class ScoredItem:
    item_idx: int
    score: float


@dataclass(frozen=True)
class RecommendationResult:
    rank: int
    game_id: str
    title: str
    genres: tuple[str, ...]
    predicted_score: float
    price: float | None = None
    developer: str = "Unknown"
    model_name: str = ""
    is_fallback: bool = False


@dataclass(frozen=True)
class ModelConfig:
    n_factors: int = 16
    learning_rate: float = 0.01
    reg: float = 0.01
    n_epochs: int = 20
    random_seed: int = 42
    patience: int = 5
    text_reg: float = 0.001

    def __post_init__(self):
        for name in ("n_factors", "n_epochs", "patience", "random_seed"):
            value = getattr(self, name)
            minimum = 0 if name == "random_seed" else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ConfigurationError(f"{name} must be an integer >= {minimum}")
        for name in ("learning_rate", "reg", "text_reg"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)
                    or value < 0 or (name == "learning_rate" and value == 0)):
                raise ConfigurationError(f"Invalid {name}: {value!r}")


@dataclass(frozen=True)
class ItemFeatures:
    values: Any
    item_ids: tuple[str, ...]
    mapping_fingerprint: str
    train_fingerprint: str
    text_fingerprint: str
    encoder_name: str

    def __post_init__(self):
        for name in ("mapping_fingerprint", "train_fingerprint", "text_fingerprint", "encoder_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ConfigurationError(f"ItemFeatures.{name} must be declared")
        ids = tuple(self.item_ids)
        if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
            raise DataValidationError("Feature item_ids must be unique raw string IDs")
        try:
            values = np.asarray(self.values)
            if values.dtype.kind not in "fiu":
                raise ValueError("values must be real numbers")
            values = np.array(values, dtype=np.float64, copy=True)
        except (ValueError, TypeError) as exc:
            raise DataValidationError("Invalid feature matrix") from exc
        if values.ndim != 2 or values.shape[0] != len(ids) or values.shape[1] == 0:
            raise DataValidationError("Feature matrix must have shape (len(item_ids), dimension > 0)")
        if not np.isfinite(values).all():
            raise DataValidationError("Feature matrix must be finite")
        norms = np.linalg.norm(values, axis=1)
        if not np.all(np.isclose(norms, 0, atol=1e-6) | np.isclose(norms, 1, atol=1e-5)):
            raise DataValidationError("Feature rows must have unit norm or be zero for missing text")
        values.flags.writeable = False
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "item_ids", ids)


@dataclass(frozen=True)
class DatasetSplits:
    train: Any
    validation: Any
    test: Any
    report: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UserEvaluation:
    user_id: int
    metrics: dict[str, float]
    recommended_items: tuple[int, ...]
    relevant_items: frozenset[int]
    candidate_count: int


@dataclass(frozen=True)
class SkippedUser:
    user_id: int
    reason: str


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    split_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    per_user: tuple[UserEvaluation, ...] = ()
    skipped_users: tuple[SkippedUser, ...] = ()
    fit_seconds: float = 0.0
    evaluate_seconds: float = 0.0


__all__ = ["InteractionRecord", "Interaction", "ReviewRecord", "GameItem", "ScoredItem",
           "RecommendationResult", "ModelConfig", "ItemFeatures", "DatasetSplits",
           "EvaluationResult", "UserEvaluation", "SkippedUser"]
