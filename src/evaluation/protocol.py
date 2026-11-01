import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Dict, Iterable, Protocol, Sequence, Tuple


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite.") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


@dataclass(frozen=True)
class Interaction:
    user_id: int
    item_id: int
    rating: float
    timestamp: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", _integer(self.user_id, "user_id"))
        object.__setattr__(self, "item_id", _integer(self.item_id, "item_id"))
        object.__setattr__(self, "rating", _real(self.rating, "rating"))
        object.__setattr__(self, "timestamp", _integer(self.timestamp, "timestamp"))


@dataclass(frozen=True)
class EvaluationProtocol:
    ks: Tuple[int, ...] = (3, 5)
    positive_threshold: float = 1.0
    exclude_seen: bool = True
    empty_ground_truth_policy: str = "skip"
    cold_start_policy: str = "skip"

    def __post_init__(self) -> None:
        ks = tuple(_integer(k, "k", 1) for k in self.ks)
        if not ks:
            raise ValueError("ks must contain at least one cutoff.")
        if len(ks) != len(set(ks)):
            raise ValueError("ks must not contain duplicate cutoffs.")
        if not isinstance(self.exclude_seen, bool):
            raise TypeError("exclude_seen must be a boolean.")
        if self.empty_ground_truth_policy not in ("skip", "zero"):
            raise ValueError("empty_ground_truth_policy must be 'skip' or 'zero'.")
        if self.cold_start_policy not in ("skip", "error"):
            raise ValueError("cold_start_policy must be 'skip' or 'error'.")
        object.__setattr__(self, "ks", tuple(sorted(ks)))
        object.__setattr__(self, "positive_threshold", _real(self.positive_threshold, "positive_threshold"))

    @property
    def max_k(self) -> int:
        return self.ks[-1]

    def is_positive(self, rating: float) -> bool:
        return _real(rating, "rating") >= self.positive_threshold

    def to_dict(self) -> Dict[str, object]:
        return {
            "ks": list(self.ks),
            "positive_threshold": self.positive_threshold,
            "exclude_seen": self.exclude_seen,
            "empty_ground_truth_policy": self.empty_ground_truth_policy,
            "cold_start_policy": self.cold_start_policy,
        }


class RankingModel(Protocol):
    name: str
    is_fitted: bool
    num_users: int
    num_items: int

    def score_items(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
        ...


def validate_interactions(
    interactions: Iterable[Interaction],
    num_users: int,
    num_items: int,
    label: str = "interactions",
) -> Tuple[Interaction, ...]:
    num_users = _integer(num_users, "num_users", 1)
    num_items = _integer(num_items, "num_items", 1)
    validated = []
    pairs = set()
    for interaction in interactions:
        if not isinstance(interaction, Interaction):
            raise TypeError(f"{label} must contain Interaction objects.")
        if interaction.user_id >= num_users or interaction.item_id >= num_items:
            raise ValueError(f"{label} contains an ID outside the configured dimensions.")
        pair = (interaction.user_id, interaction.item_id)
        if pair in pairs:
            raise ValueError(f"{label} contains duplicate user-item pair {pair}.")
        pairs.add(pair)
        validated.append(interaction)
    return tuple(validated)
