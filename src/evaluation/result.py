from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence, Set
from numbers import Integral, Real
from pathlib import Path

from src.contracts.dto import EvaluationResult, SkippedUser, UserEvaluation
from src.contracts.exceptions import DataValidationError, InvalidParameterError


__all__ = [
    "EvaluationResult",
    "UserEvaluation",
    "SkippedUser",
    "summarize_result",
    "save_results",
]


_SUMMARY_COLUMNS = (
    "model_name",
    "split_name",
    "status",
    "num_evaluated_users",
    "num_skipped_users",
    "fit_seconds",
    "evaluate_seconds",
    "skipped_reasons",
)
_USER_COLUMNS = (
    "model_name",
    "split_name",
    "user_id",
    "candidate_count",
    "recommended_items",
    "relevant_items",
)
_SKIPPED_COLUMNS = ("model_name", "split_name", "user_id", "reason")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{name} must be a nonempty string.")
    return value


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise DataValidationError(f"{name} must be a nonnegative integer.")
    return int(value)


def _number(value: object, name: str, maximum: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DataValidationError(f"{name} must be a finite real number.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise DataValidationError(f"{name} must be finite.") from error
    if not math.isfinite(number) or not 0.0 <= number <= maximum:
        raise DataValidationError(f"{name} is outside its valid range.")
    return number


def _metrics(values: object, name: str) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise DataValidationError(f"{name} must be a metric mapping.")
    return {
        _text(key, f"{name} key"): _number(value, f"{name}.{key}", 1.0)
        for key, value in values.items()
    }


def _items(values: Iterable[int], name: str) -> tuple[int, ...]:
    items = tuple(_count(value, name) for value in values)
    if len(items) != len(set(items)):
        raise DataValidationError(f"{name} must not contain duplicate IDs.")
    return items


def _prepare_result(
    result: EvaluationResult,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    if not isinstance(result, EvaluationResult):
        raise DataValidationError("Expected a shared contracts EvaluationResult.")

    model_name = _text(result.model_name, "model_name")
    split_name = _text(result.split_name, "split_name")
    metric_values = _metrics(result.metrics, "result.metrics")

    if not isinstance(result.per_user, tuple):
        raise DataValidationError("per_user must be a tuple of UserEvaluation.")
    if not isinstance(result.skipped_users, tuple):
        raise DataValidationError("skipped_users must be a tuple of SkippedUser.")

    if result.per_user and not metric_values:
        raise DataValidationError("Evaluated users require aggregate metrics.")
    if not result.per_user and metric_values:
        raise DataValidationError("Aggregate metrics require evaluated users.")

    reserved = set(_SUMMARY_COLUMNS + _USER_COLUMNS + _SKIPPED_COLUMNS)
    if reserved.intersection(metric_values):
        raise DataValidationError("Metric names conflict with report columns.")

    user_rows = []
    score_rows: list[dict[str, float]] = []
    evaluated_ids = set()

    for user in result.per_user:
        if not isinstance(user, UserEvaluation):
            raise DataValidationError("per_user must contain shared UserEvaluation.")

        user_id = _count(user.user_id, "user_id")
        if user_id in evaluated_ids:
            raise DataValidationError(f"Duplicate evaluated user {user_id}.")
        evaluated_ids.add(user_id)

        scores = _metrics(user.metrics, f"user {user_id} metrics")
        if scores.keys() != metric_values.keys():
            raise DataValidationError("Per-user and aggregate metrics must match.")
        score_rows.append(scores)

        if not isinstance(user.recommended_items, tuple):
            raise DataValidationError("recommended_items must be a tuple.")
        if not isinstance(user.relevant_items, frozenset):
            raise DataValidationError("relevant_items must be a frozenset.")

        recommended = _items(user.recommended_items, "recommended_items")
        relevant = sorted(_items(user.relevant_items, "relevant_items"))
        candidate_count = _count(user.candidate_count, "candidate_count")

        if len(recommended) > candidate_count:
            raise DataValidationError(
                "The recommendation count cannot exceed candidate_count."
            )

        user_rows.append({
            "model_name": model_name,
            "split_name": split_name,
            "user_id": user_id,
            "candidate_count": candidate_count,
            "recommended_items": json.dumps(recommended),
            "relevant_items": json.dumps(relevant),
            **scores,
        })

    for name, value in metric_values.items():
        average = math.fsum(scores[name] for scores in score_rows) / len(score_rows)
        if not math.isclose(value, average, rel_tol=1e-9, abs_tol=1e-12):
            raise DataValidationError(
                f"Aggregate metric {name} must match the per-user mean."
            )

    skipped_rows = []
    skipped_ids = set()
    reasons = Counter()

    for skipped in result.skipped_users:
        if not isinstance(skipped, SkippedUser):
            raise DataValidationError("skipped_users must contain shared SkippedUser.")

        user_id = _count(skipped.user_id, "skipped user_id")
        reason = _text(skipped.reason, "skip reason")

        if user_id in skipped_ids:
            raise DataValidationError(f"Duplicate skipped user {user_id}.")
        if user_id in evaluated_ids:
            raise DataValidationError(
                f"User {user_id} cannot be both evaluated and skipped."
            )

        skipped_ids.add(user_id)
        reasons[reason] += 1
        skipped_rows.append({
            "model_name": model_name,
            "split_name": split_name,
            "user_id": user_id,
            "reason": reason,
        })

    summary = {
        "model_name": model_name,
        "split_name": split_name,
        "status": "ok" if user_rows else "no_eligible_users",
        "num_evaluated_users": len(user_rows),
        "num_skipped_users": len(skipped_rows),
        "fit_seconds": _number(result.fit_seconds, "fit_seconds"),
        "evaluate_seconds": _number(result.evaluate_seconds, "evaluate_seconds"),
        "skipped_reasons": dict(sorted(reasons.items())),
        **metric_values,
    }

    user_rows.sort(key=lambda row: row["user_id"])
    skipped_rows.sort(key=lambda row: row["user_id"])
    return summary, user_rows, skipped_rows


def summarize_result(result: EvaluationResult) -> dict[str, object]:
    summary, _, _ = _prepare_result(result)
    return summary


def _csv_text(
    columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def save_results(
    results: Iterable[EvaluationResult],
    output_dir: str | Path,
    run_config: Mapping[str, object],
) -> dict[str, Path]:
    if isinstance(results, (str, bytes, bytearray, Mapping, Set)):
        raise InvalidParameterError("results must be an ordered iterable.")

    try:
        results = tuple(results)
    except TypeError as error:
        raise InvalidParameterError("results must be an ordered iterable.") from error

    if not results:
        raise InvalidParameterError("At least one evaluation result is required.")
    if not isinstance(run_config, Mapping):
        raise InvalidParameterError("run_config must be a mapping.")

    summary_rows = []
    user_rows = []
    skipped_rows = []
    metric_names = None

    for result in results:
        summary, users, skipped = _prepare_result(result)
        names = tuple(sorted(result.metrics))

        if names:
            if metric_names is None:
                metric_names = names
            elif names != metric_names:
                raise DataValidationError("All results must use the same metric schema.")

        summary["skipped_reasons"] = json.dumps(
            summary["skipped_reasons"],
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        summary_rows.append(summary)
        user_rows.extend(users)
        skipped_rows.extend(skipped)

    metric_names = metric_names or ()

    try:
        config_text = json.dumps(
            dict(run_config),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError, OverflowError) as error:
        raise DataValidationError(
            "run_config must contain JSON-serializable values without NaN or infinity."
        ) from error

    contents = {
        "summary": _csv_text(_SUMMARY_COLUMNS + metric_names, summary_rows),
        "per_user": _csv_text(_USER_COLUMNS + metric_names, user_rows),
        "skipped_users": _csv_text(_SKIPPED_COLUMNS, skipped_rows),
        "run_config": config_text,
    }

    if isinstance(output_dir, str) and not output_dir.strip():
        raise InvalidParameterError("output_dir must not be empty.")

    try:
        directory = Path(output_dir)
    except TypeError as error:
        raise InvalidParameterError("output_dir must be a valid path.") from error

    paths = {
        "summary": directory / "summary.csv",
        "per_user": directory / "per_user.csv",
        "skipped_users": directory / "skipped_users.csv",
        "run_config": directory / "run_config.json",
    }

    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Refusing to overwrite an existing result file.")

    directory.mkdir(parents=True, exist_ok=True)
    created = []

    try:
        for name, path in paths.items():
            with path.open("x", encoding="utf-8", newline="") as stream:
                created.append(path)
                stream.write(contents[name])
    except BaseException:
        for path in reversed(created):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    return paths