"""
Custom exception hierarchy covering all 5 required robustness test categories.

Design Justifications:
- Hierarchical classification: Inherits from custom RecSysError to allow callers to catch all
  domain exceptions with a single `except RecSysError:` handler.
- Standard Python compatibility: Subclasses also inherit from standard Python exceptions
  (ValueError, RuntimeError) where appropriate to maintain intuitive Pythonic error semantics.
- Contextual introspection: Supports optional keyword arguments to attach debug attributes
  (such as user_id, item_id, param_name) without breaking standard string representations.
"""

from typing import Any


class RecSysError(Exception):
    """Base exception for all domain and operational errors in the Game Recommendation System."""

    def __init__(self, message: str = "", **context: Any) -> None:
        super().__init__(message)
        self.message: str = message
        self.context: dict[str, Any] = context
        for key, value in context.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        return self.message


# 1. Empty or incomplete data errors (Member 1 raises, Member 4 tests)
class DataValidationError(RecSysError, ValueError):
    """Raised when input data schema is malformed, missing required columns, or contains invalid types/values."""


class EmptyDatasetError(DataValidationError):
    """Raised when the interaction dataset or source file contains zero valid interactions."""


# 2. Invalid configuration errors (Member 2/4 raises, Member 4 tests)
class ConfigurationError(RecSysError, ValueError):
    """Raised when model or system configuration hyperparameters are invalid (e.g., n_factors <= 0, lr <= 0)."""


# 3. Requests made before model training (Member 2 raises, Member 4 tests)
class ModelNotFittedError(RecSysError, RuntimeError):
    """Raised when predict, score, recommend, or evaluate is called before fit()/train() has been executed."""


# 4. Unknown users or items (Member 1/4 raises, Member 4 tests)
class UnknownEntityError(RecSysError):
    """Base exception raised when an entity identifier (user or item) is not found in the system."""


class UnknownUserError(UnknownEntityError):
    """Raised when requesting recommendations for an unseen user_id without cold-start fallback."""


class UnknownItemError(UnknownEntityError):
    """Raised when querying an item/game_id that does not exist in the catalog or dataset."""


# 5. Invalid recommendation parameters (Member 2/4 raises, Member 4 tests)
class InvalidParameterError(RecSysError, ValueError):
    """Raised when query or operational parameters such as top_k <= 0 or test_ratio outside (0, 1) are passed."""


class ColdStartError(RecSysError):
    """A user has no positive training profile and fallback is disabled."""


class TrainingError(RecSysError, RuntimeError):
    """Training or its validation callback failed; the model is invalidated."""


__all__ = [
    "ColdStartError",
    "TrainingError",
    "RecSysError",
    "DataValidationError",
    "EmptyDatasetError",
    "ConfigurationError",
    "ModelNotFittedError",
    "UnknownEntityError",
    "UnknownUserError",
    "UnknownItemError",
    "InvalidParameterError",
]
