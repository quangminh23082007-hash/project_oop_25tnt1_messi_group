from collections import defaultdict
from collections.abc import Mapping, Set as AbstractSet
from math import isfinite
from numbers import Integral, Real
from time import perf_counter
from typing import Iterable, Optional, Sequence

from .metrics import NDCGAtK, RankingMetric, RecallAtK
from .protocol import EvaluationProtocol, Interaction, RankingModel, validate_interactions
from .result import EvaluationResult, UserEvaluation


class Evaluator:
    def __init__(
        self,
        protocol: EvaluationProtocol,
        metrics: Optional[Sequence[RankingMetric]] = None,
    ) -> None:
        if not isinstance(protocol, EvaluationProtocol):
            raise TypeError("protocol must be an EvaluationProtocol.")
        self.protocol = protocol
        if metrics is None:
            metrics = [metric(k) for k in protocol.ks for metric in (RecallAtK, NDCGAtK)]
        self.metrics = tuple(metrics)
        if not self.metrics:
            raise ValueError("At least one metric is required.")
        names = set()
        for metric in self.metrics:
            if not isinstance(metric, RankingMetric):
                raise TypeError("metrics must contain RankingMetric objects.")
            if metric.k not in protocol.ks:
                raise ValueError("Every metric cutoff must belong to protocol.ks.")
            if not isinstance(metric.name, str) or not metric.name.strip():
                raise ValueError("Metric names must be nonempty strings.")
            if metric.name in names:
                raise ValueError("Metric names must be unique.")
            names.add(metric.name)

    def evaluate(
        self,
        model: RankingModel,
        train_interactions: Iterable[Interaction],
        test_interactions: Iterable[Interaction],
        num_users: int,
        num_items: int,
        additional_seen_interactions: Iterable[Interaction] = (),
        split_name: str = "test",
    ) -> EvaluationResult:
        started = perf_counter()
        train = validate_interactions(train_interactions, num_users, num_items, "train")
        evaluation = validate_interactions(test_interactions, num_users, num_items, "evaluation")
        additional = validate_interactions(
            additional_seen_interactions, num_users, num_items, "additional history"
        )
        if not train:
            raise ValueError("Training interactions must not be empty.")
        if not evaluation:
            raise ValueError("Evaluation interactions must not be empty.")
        if not isinstance(split_name, str) or not split_name.strip():
            raise ValueError("split_name must be a nonempty string.")
        self._validate_model(model, num_users, num_items)
        self._validate_disjoint(train, evaluation, additional)

        seen_items = defaultdict(set)
        train_users = set()
        for interaction in train:
            train_users.add(interaction.user_id)
            seen_items[interaction.user_id].add(interaction.item_id)
        for interaction in additional:
            seen_items[interaction.user_id].add(interaction.item_id)

        ground_truth = defaultdict(set)
        evaluation_users = set()
        for interaction in evaluation:
            evaluation_users.add(interaction.user_id)
            if self.protocol.is_positive(interaction.rating):
                ground_truth[interaction.user_id].add(interaction.item_id)

        per_user = []
        skipped_reasons = {"cold_start": 0, "empty_ground_truth": 0}
        for user_id in sorted(evaluation_users):
            if user_id not in train_users:
                if self.protocol.cold_start_policy == "error":
                    raise ValueError(f"User {user_id} has no training interactions.")
                skipped_reasons["cold_start"] += 1
                continue
            relevant = ground_truth[user_id]
            if not relevant and self.protocol.empty_ground_truth_policy == "skip":
                skipped_reasons["empty_ground_truth"] += 1
                continue

            candidates = [
                item_id
                for item_id in range(num_items)
                if not self.protocol.exclude_seen or item_id not in seen_items[user_id]
            ]
            recommended = self._recommend(model, user_id, candidates)
            values = {}
            for metric in self.metrics:
                value = metric.compute(relevant, recommended, num_total_items=num_items)
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError(f"Metric {metric.name} must return a real number.")
                value = float(value)
                if not isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"Metric {metric.name} returned a value outside [0, 1].")
                values[metric.name] = value
            per_user.append(
                UserEvaluation(
                    user_id=user_id,
                    metrics=values,
                    recommended_items=tuple(recommended),
                    relevant_items=tuple(sorted(relevant)),
                    candidate_count=len(candidates),
                )
            )

        if not per_user:
            raise ValueError("No eligible users remain under the selected evaluation protocol.")
        averages = {
            metric.name: sum(user.metrics[metric.name] for user in per_user) / len(per_user)
            for metric in self.metrics
        }
        return EvaluationResult(
            model_name=model.name,
            split_name=split_name,
            metrics=averages,
            per_user=per_user,
            skipped_reasons=skipped_reasons,
            evaluate_seconds=perf_counter() - started,
        )

    @staticmethod
    def _validate_model(model: RankingModel, num_users: int, num_items: int) -> None:
        if getattr(model, "is_fitted", False) is not True:
            raise RuntimeError("The model must be fitted before evaluation.")
        if not isinstance(getattr(model, "name", None), str) or not model.name.strip():
            raise ValueError("The model must have a nonempty name.")
        for attribute, expected in (("num_users", num_users), ("num_items", num_items)):
            value = getattr(model, attribute, None)
            if isinstance(value, bool) or not isinstance(value, Integral) or value != expected:
                raise ValueError(f"Model {attribute} does not match the evaluation dimensions.")
        if not callable(getattr(model, "score_items", None)):
            raise TypeError("The model must provide score_items(user_id, item_ids).")

    @staticmethod
    def _validate_disjoint(train, evaluation, additional) -> None:
        pair_sets = [
            {(interaction.user_id, interaction.item_id) for interaction in values}
            for values in (train, evaluation, additional)
        ]
        labels = ("train", "evaluation", "additional history")
        for left, right in ((0, 1), (0, 2), (1, 2)):
            if pair_sets[left].intersection(pair_sets[right]):
                raise ValueError(f"User-item pairs overlap between {labels[left]} and {labels[right]}.")

    def _recommend(self, model: RankingModel, user_id: int, candidates: Sequence[int]):
        if not candidates:
            return []
        raw_scores = model.score_items(user_id, tuple(candidates))
        if isinstance(raw_scores, (Mapping, AbstractSet, str, bytes)):
            raise TypeError("score_items must return scores in candidate order, not a mapping or set.")
        try:
            scores = list(raw_scores)
        except TypeError as error:
            raise TypeError("score_items must return an iterable of real scores.") from error
        if len(scores) != len(candidates):
            raise ValueError("score_items must return one score per candidate in the same order.")
        normalized = []
        for score in scores:
            if isinstance(score, bool) or not isinstance(score, Real):
                raise TypeError("Model scores must be real numbers.")
            score = float(score)
            if not isfinite(score):
                raise ValueError("Model scores must be finite.")
            normalized.append(score)
        ranked = sorted(zip(candidates, normalized), key=lambda pair: (-pair[1], pair[0]))
        return [item_id for item_id, _ in ranked[:self.protocol.max_k]]
