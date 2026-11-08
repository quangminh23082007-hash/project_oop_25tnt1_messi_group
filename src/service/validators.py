from __future__ import annotations

from numbers import Integral
from typing import Any

from src.contracts.dto import Interaction
from src.contracts.exceptions import (
    ConfigurationError,
    DataValidationError,
    EmptyDatasetError,
    InvalidParameterError,
)
from src.contracts.interfaces import BaseRecommender, checked_indices


def validate_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidParameterError(f"{name} must be a non-empty string.")
    return value.strip()


def validate_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k <= 0:
        raise InvalidParameterError(f"top_k must be a positive integer, got {top_k!r}.")
    return int(top_k)


def validate_dataset(dataset: Any, role: str) -> None:
    if dataset is None:
        raise DataValidationError(f"{role} dataset cannot be None.")

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

    seen_pairs = set()
    for interaction in dataset.interactions:
        if not isinstance(interaction, Interaction):
            raise DataValidationError(f"{role}.interactions must contain Interaction instances.")
        if interaction.user_id >= dataset.num_users or interaction.item_id >= dataset.num_items:
            raise DataValidationError(f"{role}.interactions contains an index outside the declared dimensions.")
        pair = (int(interaction.user_id), int(interaction.item_id))
        if pair in seen_pairs:
            raise DataValidationError(f"{role}.interactions contains duplicate user-item pairs.")
        seen_pairs.add(pair)

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
        raw_ids = []
        for index in range(int(mapper_size)):
            try:
                raw_id = mapper.to_raw(index)
                round_trip = mapper.to_idx(raw_id)
                contains = mapper.contains_raw(raw_id)
            except Exception as exc:
                raise DataValidationError(
                    f"{role}.{mapper_name} cannot round-trip internal index {index}."
                ) from exc
            if not isinstance(raw_id, str) or not raw_id.strip() or round_trip != index or not contains:
                raise DataValidationError(
                    f"{role}.{mapper_name} has an invalid raw/internal ID order."
                )
            raw_ids.append(raw_id)
        if len(set(raw_ids)) != len(raw_ids):
            raise DataValidationError(f"{role}.{mapper_name} contains duplicate raw IDs.")

    expected_seen = {(int(row.user_id), int(row.item_id)) for row in dataset.interactions}
    expected_positive = {
        (int(row.user_id), int(row.item_id))
        for row in dataset.interactions
        if float(row.rating) == 1.0
    }
    for user_id in range(dataset.num_users):
        try:
            seen = set(dataset.get_user_seen_items(user_id))
            positive = set(dataset.get_user_positive_items(user_id))
        except Exception as exc:
            raise DataValidationError(f"{role} history access failed for user {user_id}.") from exc
        try:
            checked_seen_values = checked_indices(list(seen), dataset.num_items)
            checked_positive_values = checked_indices(list(positive), dataset.num_items)
            checked_seen = {(user_id, int(item_id)) for item_id in checked_seen_values}
            checked_positive = {(user_id, int(item_id)) for item_id in checked_positive_values}
        except Exception as exc:
            raise DataValidationError(f"{role} history contains invalid item indices.") from exc
        if not checked_positive.issubset(checked_seen):
            raise DataValidationError(f"{role} positive history must be a subset of seen history.")
        expected_user_seen = {pair for pair in expected_seen if pair[0] == user_id}
        expected_user_positive = {pair for pair in expected_positive if pair[0] == user_id}
        if checked_seen != expected_user_seen or checked_positive != expected_user_positive:
            raise DataValidationError(
                f"{role} history does not match its interaction records for user {user_id}."
            )

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

    # Contract Section 7: BPR / TextBPR cannot be used as global fallback models.
    # Check class hierarchy and declared algorithm capability, NOT arbitrary display string model.name.
    class_names = [c.__name__.lower() for c in model.__class__.__mro__]
    is_bpr_type = any(
        "bpr" in c_name
        for c_name in class_names
        if c_name not in {"object", "baserecommender"}
    )
    declared_family = getattr(model, "algorithm_family", "").lower()
    is_declared_bpr = declared_family in {"bpr", "text_bpr", "textbpr"} or getattr(model, "is_bpr", False)

    if is_bpr_type or is_declared_bpr:
        raise ConfigurationError("BPR / TextBPR models cannot be used as global fallback models.")

    if not callable(getattr(model, "score_global", None)):
        raise InvalidParameterError("fallback_model must provide score_global(item_ids).")


def validate_evaluation_dataset(
    evaluation_dataset: Any,
    train_dataset: Any,
    additional_seen_dataset: Any = None,
) -> None:
    validate_dataset(train_dataset, "train")
    if train_dataset.split_name != "train":
        raise DataValidationError("train_dataset must have split_name 'train'.")
    if train_dataset.num_interactions == 0:
        raise EmptyDatasetError("train_dataset contains zero interactions.")
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

    train_pairs = {(int(row.user_id), int(row.item_id)) for row in train_dataset.interactions}
    evaluation_pairs = {(int(row.user_id), int(row.item_id)) for row in evaluation_dataset.interactions}
    if train_pairs.intersection(evaluation_pairs):
        raise DataValidationError("train and evaluation datasets contain duplicate user-item pairs.")

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
        additional_pairs = {
            (int(row.user_id), int(row.item_id)) for row in additional_seen_dataset.interactions
        }
        if train_pairs.intersection(additional_pairs):
            raise DataValidationError("train and additional_seen datasets contain duplicate user-item pairs.")
        if evaluation_pairs.intersection(additional_pairs):
            raise DataValidationError(
                "evaluation and additional_seen datasets contain duplicate user-item pairs."
            )
