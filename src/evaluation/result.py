import csv
import io
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer.")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def _number(value: object, name: str, maximum: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite.") from error
    if not math.isfinite(number) or not 0.0 <= number <= maximum:
        raise ValueError(f"{name} is outside its valid range.")
    return number


def _metrics(values: Mapping[str, float]) -> Dict[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("metrics must be a nonempty mapping.")
    normalized = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Metric names must be nonempty strings.")
        normalized[name] = _number(value, name, 1.0)
    return normalized


def _items(values: Sequence[int], name: str) -> Tuple[int, ...]:
    items = tuple(_count(value, name) for value in values)
    if len(items) != len(set(items)):
        raise ValueError(f"{name} must not contain duplicate IDs.")
    return items


@dataclass(frozen=True)
class UserEvaluation:
    user_id: int
    metrics: Dict[str, float]
    recommended_items: Tuple[int, ...]
    relevant_items: Tuple[int, ...]
    candidate_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", _count(self.user_id, "user_id"))
        object.__setattr__(self, "metrics", _metrics(self.metrics))
        object.__setattr__(self, "recommended_items", _items(self.recommended_items, "recommended_items"))
        object.__setattr__(self, "relevant_items", _items(self.relevant_items, "relevant_items"))
        object.__setattr__(self, "candidate_count", _count(self.candidate_count, "candidate_count"))
        if len(self.recommended_items) > self.candidate_count:
            raise ValueError("The recommendation count cannot exceed candidate_count.")


@dataclass
class EvaluationResult:
    model_name: str
    split_name: str
    metrics: Dict[str, float]
    per_user: List[UserEvaluation]
    skipped_reasons: Dict[str, int]
    fit_seconds: float = 0.0
    evaluate_seconds: float = 0.0

    @property
    def num_evaluated_users(self) -> int:
        return len(self.per_user)

    @property
    def num_skipped_users(self) -> int:
        return sum(self.skipped_reasons.values())


def _csv_text(columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def save_results(
    results: Sequence[EvaluationResult],
    output_dir: Union[str, Path],
    run_config: Mapping[str, object],
) -> Dict[str, Path]:
    results = tuple(results)
    if not results:
        raise ValueError("At least one evaluation result is required.")
    if not isinstance(run_config, Mapping):
        raise TypeError("run_config must be a mapping.")
    summary_columns = [
        "model_name", "split_name", "num_evaluated_users", "num_skipped_users",
        "fit_seconds", "evaluate_seconds", "skipped_reasons",
    ]
    user_columns = [
        "model_name", "split_name", "user_id", "candidate_count",
        "recommended_items", "relevant_items",
    ]
    metric_names = None
    summary_rows = []
    user_rows = []
    for result in results:
        if not isinstance(result, EvaluationResult):
            raise TypeError("results must contain EvaluationResult objects.")
        if not isinstance(result.model_name, str) or not result.model_name.strip():
            raise ValueError("model_name must be a nonempty string.")
        if not isinstance(result.split_name, str) or not result.split_name.strip():
            raise ValueError("split_name must be a nonempty string.")
        metric_values = _metrics(result.metrics)
        names = sorted(metric_values)
        if metric_names is None:
            metric_names = names
            if set(names).intersection(summary_columns + user_columns):
                raise ValueError("Metric names conflict with report column names.")
        elif names != metric_names:
            raise ValueError("All results must use the same metric schema.")
        if not isinstance(result.skipped_reasons, Mapping):
            raise TypeError("skipped_reasons must be a mapping.")
        skipped = {}
        for reason, count in result.skipped_reasons.items():
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("Skip reasons must be nonempty strings.")
            skipped[reason] = _count(count, "skip count")
        summary_rows.append({
            "model_name": result.model_name,
            "split_name": result.split_name,
            "num_evaluated_users": len(result.per_user),
            "num_skipped_users": sum(skipped.values()),
            "fit_seconds": _number(result.fit_seconds, "fit_seconds"),
            "evaluate_seconds": _number(result.evaluate_seconds, "evaluate_seconds"),
            "skipped_reasons": json.dumps(skipped, ensure_ascii=False, sort_keys=True),
            **metric_values,
        })
        user_ids = set()
        for user in result.per_user:
            if not isinstance(user, UserEvaluation):
                raise TypeError("per_user must contain UserEvaluation objects.")
            user_id = _count(user.user_id, "user_id")
            if user_id in user_ids:
                raise ValueError("A result cannot contain the same user twice.")
            user_ids.add(user_id)
            scores = _metrics(user.metrics)
            if sorted(scores) != metric_names:
                raise ValueError("Per-user metrics must match the result metric schema.")
            recommended = _items(user.recommended_items, "recommended_items")
            relevant = _items(user.relevant_items, "relevant_items")
            candidate_count = _count(user.candidate_count, "candidate_count")
            if len(recommended) > candidate_count:
                raise ValueError("The recommendation count cannot exceed candidate_count.")
            user_rows.append({
                "model_name": result.model_name,
                "split_name": result.split_name,
                "user_id": user_id,
                "candidate_count": candidate_count,
                "recommended_items": json.dumps(recommended),
                "relevant_items": json.dumps(relevant),
                **scores,
            })
    config_text = json.dumps(dict(run_config), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    contents = {
        "summary": _csv_text(summary_columns + metric_names, summary_rows),
        "per_user": _csv_text(user_columns + metric_names, user_rows),
        "run_config": config_text,
    }
    directory = Path(output_dir)
    paths = {
        "summary": directory / "summary.csv",
        "per_user": directory / "per_user.csv",
        "run_config": directory / "run_config.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Refusing to overwrite an existing result file.")
    directory.mkdir(parents=True, exist_ok=True)
    opened = []
    try:
        for name, path in paths.items():
            stream = path.open("x", encoding="utf-8", newline="")
            opened.append((stream, path))
            stream.write(contents[name])
            stream.close()
    except BaseException:
        for stream, path in opened:
            stream.close()
            path.unlink(missing_ok=True)
        raise
    return paths
