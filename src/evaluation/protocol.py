from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable, Protocol

from src.contracts.dto import Interaction, ScoredItem
from src.contracts.exceptions import (
    ConfigurationError,
    DataValidationError,
    InvalidParameterError,
)


__all__ = [
    "Interaction",
    "EvaluationProtocol",
    "RankingModel",
    "RankingEvaluationData",
    "RankingProviderProtocol",
    "validate_interactions",
]


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise InvalidParameterError(f"{name} must be an integer.")
    if value < minimum:
        raise InvalidParameterError(f"{name} must be at least {minimum}.")
    return int(value)


def _binary_rating(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or value not in (0, 1)
    ):
        raise DataValidationError(f"{name} must be numeric 0 or 1.")
    return float(value)


@dataclass(frozen=True)
class EvaluationProtocol:
    ks: tuple[int, ...] = (5, 10)
    positive_threshold: float = 1.0
    exclude_seen: bool = True
    empty_ground_truth_policy: str = "skip"
    cold_start_policy: str = "skip"

    def __post_init__(self) -> None:
        if isinstance(self.ks, (str, bytes, bytearray, Mapping, Set)):
            raise ConfigurationError("ks must be an ordered iterable of integers.")

        try:
            ks = tuple(_integer(k, "k", 1) for k in self.ks)
        except (TypeError, InvalidParameterError) as error:
            raise ConfigurationError(
                "ks must contain positive integer cutoffs."
            ) from error

        if not ks:
            raise ConfigurationError("ks must contain at least one cutoff.")
        if len(ks) != len(set(ks)):
            raise ConfigurationError("ks must not contain duplicate cutoffs.")

        if (
            isinstance(self.positive_threshold, bool)
            or not isinstance(self.positive_threshold, Real)
            or self.positive_threshold != 1.0
        ):
            raise ConfigurationError(
                "positive_threshold must be 1.0 for binary feedback."
            )

        if not isinstance(self.exclude_seen, bool):
            raise ConfigurationError("exclude_seen must be a boolean.")

        if (
            not isinstance(self.empty_ground_truth_policy, str)
            or self.empty_ground_truth_policy not in ("skip", "zero")
        ):
            raise ConfigurationError(
                "empty_ground_truth_policy must be 'skip' or 'zero'."
            )

        if (
            not isinstance(self.cold_start_policy, str)
            or self.cold_start_policy not in ("skip", "error")
        ):
            raise ConfigurationError(
                "cold_start_policy must be 'skip' or 'error'."
            )

        object.__setattr__(self, "ks", tuple(sorted(ks)))
        object.__setattr__(self, "positive_threshold", 1.0)

    @property
    def max_k(self) -> int:
        return self.ks[-1]

    def is_positive(self, rating: float) -> bool:
        return _binary_rating(rating, "rating") == 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "ks": list(self.ks),
            "positive_threshold": self.positive_threshold,
            "exclude_seen": self.exclude_seen,
            "empty_ground_truth_policy": self.empty_ground_truth_policy,
            "cold_start_policy": self.cold_start_policy,
        }


class RankingModel(Protocol):
    @property
    def name(self) -> str:
        ...

    @property
    def is_fitted(self) -> bool:
        ...

    @property
    def num_users(self) -> int:
        ...

    @property
    def num_items(self) -> int:
        ...

    @property
    def mapping_fingerprint(self) -> str | None:
        ...

    @property
    def train_fingerprint(self) -> str | None:
        ...

    def score_items(
        self,
        user_id: int,
        item_ids: Sequence[int],
    ) -> Iterable[float]:
        ...


@dataclass(frozen=True)
class RankingEvaluationData:
    split_name: str
    num_users: int
    num_items: int
    user_ids: tuple[int, ...]
    get_relevant_items: Callable[[int], frozenset[int]]
    get_seen_items: Callable[[int], frozenset[int]]
    has_positive_train: Callable[[int], bool]
    data_fingerprint: str


class RankingProviderProtocol(Protocol):
    name: str
    num_users: int
    num_items: int
    data_fingerprint: str

    def rank_items(
        self,
        user_id: int,
        *,
        top_k: int,
        seen_items: frozenset[int],
    ) -> Sequence[ScoredItem]:
        ...


def validate_interactions(
    interactions: Iterable[Interaction],
    num_users: int,
    num_items: int,
    label: str = "interactions",
) -> tuple[Interaction, ...]:
    num_users = _integer(num_users, "num_users", 1)
    num_items = _integer(num_items, "num_items", 1)

    if not isinstance(label, str) or not label.strip():
        raise InvalidParameterError("label must be a nonempty string.")

    if isinstance(interactions, (str, bytes, bytearray, Mapping, Set)):
        raise DataValidationError(
            f"{label} must be an ordered iterable of Interaction objects."
        )

    try:
        iterator = iter(interactions)
    except TypeError as error:
        raise DataValidationError(
            f"{label} must be an ordered iterable of Interaction objects."
        ) from error

    validated: list[Interaction] = []
    pairs: set[tuple[int, int]] = set()

    for index, interaction in enumerate(iterator):
        prefix = f"{label}[{index}]"

        if not isinstance(interaction, Interaction):
            raise DataValidationError(
                f"{prefix} must be a shared contracts Interaction."
            )

        try:
            user_id = _integer(interaction.user_id, f"{prefix}.user_id")
            item_id = _integer(interaction.item_id, f"{prefix}.item_id")
            _integer(interaction.timestamp, f"{prefix}.timestamp")
        except InvalidParameterError as error:
            raise DataValidationError(str(error)) from error

        _binary_rating(interaction.rating, f"{prefix}.rating")

        if user_id >= num_users or item_id >= num_items:
            raise DataValidationError(
                f"{prefix} contains an ID outside the configured dimensions."
            )

        pair = (user_id, item_id)
        if pair in pairs:
            raise DataValidationError(
                f"{label} contains duplicate user-item pair {pair}."
            )

        pairs.add(pair)
        validated.append(interaction)

    return tuple(validated)
