"""Contracts package defining shared DTOs, Exceptions, and Abstract Interfaces."""

from src.contracts.dto import (
    EvaluationReport,
    InteractionRecord,
    ModelConfig,
    RecommendationResult,
    ScoredItem,
)
from src.contracts.exceptions import (
    ConfigurationError,
    DataValidationError,
    EmptyDatasetError,
    InvalidParameterError,
    ModelNotFittedError,
    RecSysError,
    UnknownEntityError,
    UnknownItemError,
    UnknownUserError,
)
from src.contracts.interfaces import (
    BaseDataLoader,
    BaseMetric,
    BaseRecommender,
)

__all__ = [
    "RecSysError",
    "DataValidationError",
    "EmptyDatasetError",
    "ConfigurationError",
    "ModelNotFittedError",
    "UnknownEntityError",
    "UnknownUserError",
    "UnknownItemError",
    "InvalidParameterError",
    "InteractionRecord",
    "ScoredItem",
    "RecommendationResult",
    "ModelConfig",
    "EvaluationReport",
    "BaseDataLoader",
    "BaseRecommender",
    "BaseMetric",
]
