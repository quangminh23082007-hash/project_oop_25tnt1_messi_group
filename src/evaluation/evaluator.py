from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence, Set
from math import fsum, isfinite
from numbers import Integral, Real
from time import perf_counter

import numpy as np

from src.contracts.dto import EvaluationResult, Interaction, SkippedUser, UserEvaluation
from src.contracts.exceptions import (
    ColdStartError,
    ConfigurationError,
    DataValidationError,
    EmptyDatasetError,
    ModelNotFittedError,
)
from src.contracts.interfaces import BaseEvaluator, BaseMetric, InteractionDataset

from .metrics import NDCGAtK, RecallAtK
from .protocol import EvaluationProtocol, RankingModel, validate_interactions


__all__ = ["Evaluator"]


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


class Evaluator(BaseEvaluator):
    def __init__(
        self,
        protocol: EvaluationProtocol,
        metrics: Iterable[BaseMetric] | None = None,
    ) -> None:
        if not isinstance(protocol, EvaluationProtocol):
            raise ConfigurationError("protocol must be an EvaluationProtocol.")

        if metrics is None:
            metrics = [
                metric_type(k)
                for k in protocol.ks
                for metric_type in (RecallAtK, NDCGAtK)
            ]
        elif isinstance(metrics, (str, bytes, bytearray, Mapping, Set)):
            raise ConfigurationError("metrics must be an ordered iterable.")

        try:
            metrics = tuple(metrics)
        except TypeError as error:
            raise ConfigurationError("metrics must be an ordered iterable.") from error

        if not metrics:
            raise ConfigurationError("At least one metric is required.")

        names = []
        for metric in metrics:
            if not isinstance(metric, BaseMetric):
                raise ConfigurationError("Every metric must inherit shared BaseMetric.")

            k = metric.k
            if (
                isinstance(k, bool)
                or not isinstance(k, Integral)
                or k < 1
                or k not in protocol.ks
            ):
                raise ConfigurationError(
                    "Every metric must have a positive cutoff in protocol.ks."
                )

            name = metric.name
            if not isinstance(name, str) or not name.strip():
                raise ConfigurationError("Metric names must be nonempty strings.")
            if name in names:
                raise ConfigurationError("Metric names must be unique.")
            names.append(name)

        self.protocol = protocol
        self.metrics = metrics
        self._metric_names = tuple(names)

    def evaluate(
        self,
        model: RankingModel,
        train_dataset: InteractionDataset,
        evaluation_dataset: InteractionDataset,
        *,
        additional_seen_dataset: InteractionDataset | None = None,
    ) -> EvaluationResult:
        started = perf_counter()

        train = self._dataset_rows(train_dataset, "train", ("train",))
        if not train:
            raise EmptyDatasetError("Training interactions must not be empty.")

        evaluation = self._dataset_rows(
            evaluation_dataset,
            "evaluation",
            ("validation", "test"),
            reference=train_dataset,
        )

        additional = ()
        if additional_seen_dataset is not None:
            if evaluation_dataset.split_name != "test":
                raise DataValidationError(
                    "Additional validation history is allowed only when evaluating test."
                )
            additional = self._dataset_rows(
                additional_seen_dataset,
                "additional validation",
                ("validation",),
                reference=train_dataset,
            )

        self._validate_model(model, train_dataset)
        self._validate_disjoint(train, evaluation, additional)

        num_items = int(train_dataset.num_items)
        seen_items: dict[int, set[int]] = defaultdict(set)
        positive_train_users = set()

        for interaction in train:
            user_id = int(interaction.user_id)
            item_id = int(interaction.item_id)
            seen_items[user_id].add(item_id)
            if self.protocol.is_positive(interaction.rating):
                positive_train_users.add(user_id)

        for interaction in additional:
            seen_items[int(interaction.user_id)].add(int(interaction.item_id))

        ground_truth: dict[int, set[int]] = defaultdict(set)
        evaluation_users = set()

        for interaction in evaluation:
            user_id = int(interaction.user_id)
            evaluation_users.add(user_id)
            if self.protocol.is_positive(interaction.rating):
                ground_truth[user_id].add(int(interaction.item_id))

        per_user = []
        skipped_users = []

        for user_id in sorted(evaluation_users):
            if user_id not in positive_train_users:
                if self.protocol.cold_start_policy == "error":
                    raise ColdStartError(
                        f"User {user_id} has no positive training profile.",
                        user_id=user_id,
                    )
                skipped_users.append(SkippedUser(user_id, "no_positive_train"))
                continue

            relevant = frozenset(ground_truth[user_id])
            if not relevant and self.protocol.empty_ground_truth_policy == "skip":
                skipped_users.append(SkippedUser(user_id, "empty_ground_truth"))
                continue

            candidates = tuple(
                item_id
                for item_id in range(num_items)
                if not self.protocol.exclude_seen or item_id not in seen_items[user_id]
            )

            if not candidates:
                skipped_users.append(SkippedUser(user_id, "no_candidates"))
                continue

            recommended = self._recommend(model, user_id, candidates)
            values = {}

            for name, metric in zip(self._metric_names, self.metrics):
                value = (
                    metric.compute(set(relevant), recommended, num_total_items=num_items)
                    if relevant
                    else 0.0
                )
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise DataValidationError(f"Metric {name} must return a real scalar.")
                try:
                    value = float(value)
                except (OverflowError, ValueError) as error:
                    raise DataValidationError(f"Metric {name} must be finite.") from error
                if not isfinite(value) or not 0.0 <= value <= 1.0:
                    raise DataValidationError(
                        f"Metric {name} must return a finite value in [0, 1]."
                    )
                values[name] = value

            per_user.append(
                UserEvaluation(
                    user_id=user_id,
                    metrics=values,
                    recommended_items=recommended,
                    relevant_items=relevant,
                    candidate_count=len(candidates),
                )
            )

        averages = (
            {
                name: fsum(user.metrics[name] for user in per_user) / len(per_user)
                for name in self._metric_names
            }
            if per_user
            else {}
        )

        return EvaluationResult(
            model_name=model.name,
            split_name=evaluation_dataset.split_name,
            metrics=averages,
            per_user=tuple(per_user),
            skipped_users=tuple(skipped_users),
            evaluate_seconds=perf_counter() - started,
        )

    @staticmethod
    def _dataset_rows(
        dataset: InteractionDataset,
        label: str,
        allowed_splits: tuple[str, ...],
        reference: InteractionDataset | None = None,
    ) -> tuple[Interaction, ...]:
        split_name = _text(getattr(dataset, "split_name", None), f"{label}.split_name")
        if split_name not in allowed_splits:
            raise DataValidationError(
                f"{label}.split_name must be one of {allowed_splits}."
            )

        num_users = _integer(getattr(dataset, "num_users", None), f"{label}.num_users", 1)
        num_items = _integer(getattr(dataset, "num_items", None), f"{label}.num_items", 1)
        count = _integer(
            getattr(dataset, "num_interactions", None),
            f"{label}.num_interactions",
        )

        for name in ("mapping_fingerprint", "fingerprint", "review_fingerprint"):
            _text(getattr(dataset, name, None), f"{label}.{name}")

        for name, expected in (("user_mapper", num_users), ("item_mapper", num_items)):
            mapper = getattr(dataset, name, None)
            size = _integer(getattr(mapper, "size", None), f"{label}.{name}.size")
            if size != expected:
                raise DataValidationError(f"{label}.{name}.size does not match dimensions.")

        if reference is not None:
            if (
                num_users != reference.num_users
                or num_items != reference.num_items
                or dataset.mapping_fingerprint != reference.mapping_fingerprint
            ):
                raise DataValidationError(
                    f"{label} must use the same dimensions and mapping as train."
                )

        records = getattr(dataset, "interactions", None)
        if (
            isinstance(records, (str, bytes, bytearray))
            or not isinstance(records, Sequence)
        ):
            raise DataValidationError(f"{label}.interactions must be an ordered sequence.")

        rows = validate_interactions(
            records,
            num_users,
            num_items,
            label,
        )
        if len(rows) != count:
            raise DataValidationError(f"{label}.num_interactions does not match its records.")
        return rows

    @staticmethod
    def _validate_model(
        model: RankingModel,
        train_dataset: InteractionDataset,
    ) -> None:
        if getattr(model, "is_fitted", False) is not True:
            raise ModelNotFittedError("The model must be fitted before evaluation.")

        _text(getattr(model, "name", None), "model.name")

        for name in ("num_users", "num_items"):
            value = _integer(getattr(model, name, None), f"model.{name}", 1)
            if value != getattr(train_dataset, name):
                raise DataValidationError(f"Model {name} does not match train.")

        mapping_fingerprint = _text(
            getattr(model, "mapping_fingerprint", None),
            "model.mapping_fingerprint",
        )
        train_fingerprint = _text(
            getattr(model, "train_fingerprint", None),
            "model.train_fingerprint",
        )
        if mapping_fingerprint != train_dataset.mapping_fingerprint:
            raise DataValidationError("Model and train mapping fingerprints must match.")
        if train_fingerprint != train_dataset.fingerprint:
            raise DataValidationError("Model must be fitted on the supplied train fingerprint.")
        if not callable(getattr(model, "score_items", None)):
            raise DataValidationError("Model must provide score_items(user_id, item_ids).")

    @staticmethod
    def _validate_disjoint(
        train: Sequence[Interaction],
        evaluation: Sequence[Interaction],
        additional: Sequence[Interaction],
    ) -> None:
        pair_sets = [
            {(row.user_id, row.item_id) for row in rows}
            for rows in (train, evaluation, additional)
        ]
        labels = ("train", "evaluation", "additional validation")

        for left, right in ((0, 1), (0, 2), (1, 2)):
            if pair_sets[left].intersection(pair_sets[right]):
                raise DataValidationError(
                    f"User-item pairs overlap between {labels[left]} and {labels[right]}."
                )

    def _recommend(
        self,
        model: RankingModel,
        user_id: int,
        candidates: Sequence[int],
    ) -> tuple[int, ...]:
        if not candidates:
            return ()

        raw_scores = model.score_items(user_id, candidates)

        if isinstance(raw_scores, np.ndarray):
            if raw_scores.ndim != 1 or raw_scores.dtype.kind not in "fiu":
                raise DataValidationError("Model scores must be a real numeric vector.")
            scores = np.asarray(raw_scores, dtype=float)
        else:
            if isinstance(raw_scores, (str, bytes, bytearray, Mapping, Set)):
                raise DataValidationError("Model scores must preserve candidate order.")
            try:
                values = tuple(raw_scores)
            except TypeError as error:
                raise DataValidationError("Model scores must be an ordered iterable.") from error
            if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
                raise DataValidationError("Model scores must contain real numbers without bool.")
            try:
                scores = np.asarray(values, dtype=float)
            except (TypeError, ValueError, OverflowError) as error:
                raise DataValidationError("Model scores must be finite real numbers.") from error

        if scores.shape != (len(candidates),):
            raise DataValidationError("Model must return one score for each candidate.")
        if not np.isfinite(scores).all():
            raise DataValidationError("Model scores must be finite.")

        item_ids = np.asarray(candidates, dtype=np.int64)
        order = np.lexsort((item_ids, -scores))[:self.protocol.max_k]
        return tuple(int(item_ids[index]) for index in order)