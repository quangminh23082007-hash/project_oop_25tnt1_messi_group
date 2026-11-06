from __future__ import annotations

from numbers import Integral
from typing import Any

from src.contracts.exceptions import (
    ConfigurationError,
    DataValidationError,
    EmptyDatasetError,
    InvalidParameterError,
)
from src.contracts.interfaces import BaseRecommender


def validate_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidParameterError(f"{name} must be a non-empty string.")
    return value.strip()


def validate_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k <= 0:
        raise InvalidParameterError(f"top_k must be a positive integer, got {top_k!r}.")
    return int(top_k)


def validate_dataset(dataset: Any, role: str) -> None:
    required = (
        "split_name",
        "interactions",
        "num_interactions",
        "num_users",
        "num_items",
        "user_mapper",
        "item_mapper",
        "mapping_fingerprint",
        "fingerprint",
        "review_fingerprint",
        "get_game_by_raw_id",
        "get_user_seen_items",
        "get_user_positive_items",
        "iter_reviews",
    )
    missing = [attr for attr in required if not hasattr(dataset, attr)]
    if missing:
        raise DataValidationError(
            f"{role} dataset is missing required attributes: {', '.join(missing)}."
        )

    if dataset.split_name not in {"train", "validation", "test"}:
        raise DataValidationError("dataset.split_name must be 'train', 'validation' or 'test'.")

    for name in ("num_interactions", "num_users", "num_items"):
        value = getattr(dataset, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise DataValidationError(f"{role}.{name} must be a non-negative integer.")

    try:
        interaction_count = len(dataset.interactions)
    except TypeError as exc:
        raise DataValidationError(f"{role}.interactions must be a sequence.") from exc

    if interaction_count != dataset.num_interactions:
        raise DataValidationError(
            f"{role}.num_interactions must match the interactions sequence length."
        )

    for mapper_name in ("user_mapper", "item_mapper"):
        mapper = getattr(dataset, mapper_name)
        mapper_size = getattr(mapper, "size", None)
        expected_size = dataset.num_users if mapper_name == "user_mapper" else dataset.num_items
        if (
            isinstance(mapper_size, bool)
            or not isinstance(mapper_size, Integral)
            or mapper_size < 0
            or mapper_size != expected_size
            or not callable(getattr(mapper, "contains_raw", None))
            or not callable(getattr(mapper, "to_idx", None))
            or not callable(getattr(mapper, "to_raw", None))
        ):
            raise DataValidationError(f"{role}.{mapper_name} does not satisfy IdMapper.")

    for name in ("mapping_fingerprint", "fingerprint", "review_fingerprint"):
        val = getattr(dataset, name)
        if not isinstance(val, str) or not val.strip():
            raise DataValidationError(f"{role}.{name} must be a non-empty string.")

    for name in (
        "get_game_by_raw_id",
        "get_user_seen_items",
        "get_user_positive_items",
        "iter_reviews",
    ):
        if not callable(getattr(dataset, name)):
            raise DataValidationError(f"{role}.{name} must be callable.")


def model_matches_dataset(model: BaseRecommender, dataset: Any) -> bool:
    return (
        bool(getattr(model, "is_fitted", False))
        and getattr(model, "num_users", None) == dataset.num_users
        and getattr(model, "num_items", None) == dataset.num_items
        and getattr(model, "mapping_fingerprint", None) == dataset.mapping_fingerprint
        and getattr(model, "train_fingerprint", None) == dataset.fingerprint
    )


def validate_fallback_model(model: BaseRecommender) -> None:
    if not isinstance(model, BaseRecommender):
        raise InvalidParameterError("fallback_model must be a BaseRecommender instance.")
    model_name = getattr(model, "name", "").lower()
    class_name = model.__class__.__name__.lower()
    if "bpr" in model_name or "bpr" in class_name:
        raise ConfigurationError("BPR / TextBPR models cannot be used as global fallback models.")
    if not callable(getattr(model, "score_global", None)):
        raise InvalidParameterError("fallback_model must provide score_global(item_ids).")


def validate_evaluation_dataset(
    evaluation_dataset: Any,
    train_dataset: Any,
    additional_seen_dataset: Any = None,
) -> None:
    validate_dataset(evaluation_dataset, "evaluation")
    if evaluation_dataset.split_name not in {"validation", "test"}:
        raise DataValidationError(
            "evaluation_dataset must have split_name 'validation' or 'test'."
        )
    if evaluation_dataset.mapping_fingerprint != train_dataset.mapping_fingerprint:
        raise DataValidationError(
            "evaluation_dataset mapping_fingerprint does not match the train dataset."
        )
    if (
        evaluation_dataset.num_users != train_dataset.num_users
        or evaluation_dataset.num_items != train_dataset.num_items
    ):
        raise DataValidationError(
            "evaluation_dataset dimensions do not match the train dataset."
        )
    if evaluation_dataset.num_interactions == 0:
        raise EmptyDatasetError("evaluation_dataset contains zero interactions.")

    if additional_seen_dataset is not None:
        if evaluation_dataset.split_name != "test":
            raise DataValidationError(
                "additional_seen_dataset can only be provided when evaluating on test split."
            )
        validate_dataset(additional_seen_dataset, "additional_seen")
        if additional_seen_dataset.split_name != "validation":
            raise DataValidationError(
                "additional_seen_dataset must have split_name 'validation'."
            )
        if additional_seen_dataset.mapping_fingerprint != train_dataset.mapping_fingerprint:
            raise DataValidationError(
                "additional_seen_dataset mapping_fingerprint does not match train."
            )
        if (
            additional_seen_dataset.num_users != train_dataset.num_users
            or additional_seen_dataset.num_items != train_dataset.num_items
        ):
            raise DataValidationError(
                "additional_seen_dataset dimensions do not match the train dataset."
            )
