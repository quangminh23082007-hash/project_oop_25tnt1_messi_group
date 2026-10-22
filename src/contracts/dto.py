"""
Immutable Data Transfer Objects (DTOs) shared across all 4 architectural layers.

Design Justifications:
- Immutability via `@dataclass(frozen=True)`: Prevents unintended side effects and state
  mutation when transferring records across layer boundaries (Encapsulation).
- Defensive Validation in `__post_init__`: Guarantees that any invalid record, configuration,
  or result is rejected immediately at creation time rather than causing subtle failures later.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from src.contracts.exceptions import ConfigurationError, DataValidationError


@dataclass(frozen=True)
class InteractionRecord:
    """Raw interaction record loaded from source files before integer indexing."""

    user_id: str
    item_id: str
    rating: float
    timestamp: int = 0

    def __post_init__(self) -> None:
        if not self.user_id or not isinstance(self.user_id, str) or not self.user_id.strip():
            raise DataValidationError("user_id must be a non-empty string.")
        if not self.item_id or not isinstance(self.item_id, str) or not self.item_id.strip():
            raise DataValidationError("item_id must be a non-empty string.")
        try:
            rating_val = float(self.rating)
        except (ValueError, TypeError):
            raise DataValidationError(f"rating must be numeric, got {self.rating!r}.")
        if not (0.0 <= rating_val <= 5.0):
            raise DataValidationError(f"rating must be in [0.0, 5.0], got {self.rating}.")


@dataclass(frozen=True)
class ScoredItem:
    """Internal model output pairing an internal item index with its predicted score."""

    item_idx: int
    score: float


@dataclass(frozen=True)
class RecommendationResult:
    """User-facing recommendation item returned by the Service Layer to external clients."""

    rank: int
    game_id: str
    title: str
    genres: Tuple[str, ...]
    predicted_score: float
    developer: str = "Unknown"


@dataclass(frozen=True)
class ModelConfig:
    """Validated configuration parameters for Recommendation Models."""

    n_factors: int = 16
    learning_rate: float = 0.01
    reg: float = 0.01
    n_epochs: int = 20
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.n_factors <= 0:
            raise ConfigurationError(f"n_factors must be > 0, got {self.n_factors}.")
        if self.learning_rate <= 0.0:
            raise ConfigurationError(f"learning_rate must be > 0, got {self.learning_rate}.")
        if self.reg < 0.0:
            raise ConfigurationError(f"regularization must be >= 0, got {self.reg}.")
        if self.n_epochs <= 0:
            raise ConfigurationError(f"n_epochs must be > 0, got {self.n_epochs}.")


@dataclass(frozen=True)
class EvaluationReport:
    """Standardized evaluation output returned by Evaluator and Cornac Benchmark."""

    model_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    num_evaluated_users: int = 0
    notes: Optional[str] = None


__all__ = [
    "InteractionRecord",
    "ScoredItem",
    "RecommendationResult",
    "ModelConfig",
    "EvaluationReport",
]
