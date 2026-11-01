import argparse
import csv
import hashlib
import json
import platform
import random
import re
import sys
from dataclasses import replace
from datetime import datetime, timezone
from importlib import metadata
from math import isfinite
from numbers import Integral
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
from uuid import uuid4

from src.evaluation.evaluator import Evaluator
from src.evaluation.protocol import EvaluationProtocol, Interaction, validate_interactions
from src.evaluation.result import EvaluationResult, save_results

from .cornac_adapter import CornacMostPopAdapter


class DemoPopularityModel:
    def __init__(self, name: str = "DemoPopularity", positive_threshold: float = 1.0) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Model name must be a non-empty string.")
        self.name = name.strip()
        self.positive_threshold = EvaluationProtocol(positive_threshold=positive_threshold).positive_threshold
        self.is_fitted = False
        self.num_users = 0
        self.num_items = 0
        self._scores: List[float] = []

    def fit(
        self,
        train_interactions: Sequence[Interaction],
        num_users: int,
        num_items: int,
    ) -> "DemoPopularityModel":
        training = validate_interactions(train_interactions, num_users, num_items, "train_interactions")
        if not training:
            raise ValueError("Cannot train DemoPopularity on an empty training set.")
        scores = [0.0] * int(num_items)
        for interaction in training:
            if interaction.rating >= self.positive_threshold:
                scores[interaction.item_id] += 1.0
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self._scores = scores
        self.is_fitted = True
        return self

    def score_items(self, user_id: int, item_ids: Sequence[int]) -> List[float]:
        if not self.is_fitted:
            raise RuntimeError("Call fit before requesting scores.")
        if isinstance(user_id, bool) or not isinstance(user_id, Integral):
            raise TypeError("user_id must be an integer.")
        if not 0 <= user_id < self.num_users:
            raise ValueError("user_id is outside the user catalog.")
        if isinstance(item_ids, (str, bytes)) or not isinstance(item_ids, Sequence):
            raise TypeError("item_ids must be an ordered sequence of integer IDs.")
        scores = []
        for item_id in item_ids:
            if isinstance(item_id, bool) or not isinstance(item_id, Integral):
                raise TypeError("item_ids contains a non-integer ID.")
            if not 0 <= item_id < self.num_items:
                raise ValueError("item_ids contains an ID outside the item catalog.")
            scores.append(self._scores[int(item_id)])
        return scores


def _validate_keys(
    value: object,
    allowed: Sequence[str],
    label: str,
    required: Sequence[str] = (),
) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object.")
    unknown = set(value) - set(allowed)
    missing = set(required) - set(value)
    if unknown:
        raise ValueError(f"Unknown keys in {label}: {', '.join(sorted(unknown))}.")
    if missing:
        raise ValueError(f"Missing keys in {label}: {', '.join(sorted(missing))}.")
    return value


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer.")
    if value <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    return int(value)


def _read_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    return _validate_keys(
        config,
        ("run_name", "seed", "protocol", "data", "benchmark", "output_dir"),
        "config",
        ("run_name", "seed", "protocol", "data", "benchmark", "output_dir"),
    )


def _read_csv(path: Path) -> Tuple[Interaction, ...]:
    interactions = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}.")
        headers = [header.strip() for header in reader.fieldnames]
        if len(set(headers)) != len(headers):
            raise ValueError(f"CSV contains duplicate column names: {path}.")
        required = {"user_id", "item_id", "rating"}
        allowed = required | {"timestamp"}
        if not required.issubset(headers) or set(headers) - allowed:
            raise ValueError(f"CSV columns must be user_id,item_id,rating and optional timestamp: {path}.")
        reader.fieldnames = headers
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(column) is None for column in headers):
                raise ValueError(f"CSV row has the wrong number of fields at {path}:{line_number}.")
            try:
                user_id = int(row["user_id"].strip())
                item_id = int(row["item_id"].strip())
                rating = float(row["rating"].strip())
                raw_timestamp = row.get("timestamp", "").strip()
                timestamp = int(raw_timestamp) if raw_timestamp else 0
                interaction = Interaction(user_id, item_id, rating, timestamp)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid interaction at {path}:{line_number}: {error}") from error
            interactions.append(interaction)
    return tuple(interactions)


def _demo_data() -> Tuple[Tuple[Interaction, ...], Tuple[Interaction, ...], Tuple[Interaction, ...], int, int]:
    train_rows = (
        (0, 0, 1), (0, 1, 1), (0, 2, 0),
        (1, 0, 1), (1, 3, 1), (1, 4, 0),
        (2, 1, 1), (2, 4, 1), (2, 5, 0),
        (3, 2, 1), (3, 5, 1), (3, 6, 0),
    )
    validation_rows = (
        (0, 3, 1), (1, 1, 1), (2, 2, 1), (3, 0, 1), (4, 6, 1),
    )
    test_rows = (
        (0, 4, 1), (0, 5, 0),
        (1, 5, 1), (1, 6, 0),
        (2, 6, 1), (2, 7, 0),
        (3, 1, 1), (3, 7, 0),
        (4, 7, 1),
    )
    train = tuple(Interaction(*row, timestamp=index) for index, row in enumerate(train_rows, start=1))
    validation = tuple(Interaction(*row, timestamp=index) for index, row in enumerate(validation_rows, start=101))
    test = tuple(Interaction(*row, timestamp=index) for index, row in enumerate(test_rows, start=201))
    return train, validation, test, 5, 8


def _check_disjoint_splits(splits: Mapping[str, Sequence[Interaction]]) -> None:
    seen = {}
    for split_name, interactions in splits.items():
        for interaction in interactions:
            pair = (interaction.user_id, interaction.item_id)
            if pair in seen:
                raise ValueError(f"User-item pair {pair} occurs in both {seen[pair]} and {split_name}.")
            seen[pair] = split_name


def _resolve_data(
    data_config: Dict[str, object],
    package_root: Path,
) -> Tuple[Dict[str, Tuple[Interaction, ...]], int, int, Dict[str, object]]:
    data_config = _validate_keys(
        data_config,
        ("source", "train_csv", "validation_csv", "test_csv", "num_users", "num_items"),
        "data",
        ("source",),
    )
    source = data_config["source"]
    source_paths: Dict[str, object] = {}
    if source == "demo":
        if any(data_config.get(key) is not None for key in ("train_csv", "validation_csv", "test_csv", "num_users", "num_items")):
            raise ValueError("Demo data does not accept CSV paths or custom catalog dimensions.")
        train, validation, test, num_users, num_items = _demo_data()
        splits = {"train": train, "validation": validation, "test": test}
    elif source == "csv":
        num_users = _positive_integer(data_config.get("num_users"), "data.num_users")
        num_items = _positive_integer(data_config.get("num_items"), "data.num_items")
        splits = {}
        for split_name in ("train", "validation", "test"):
            configured_path = data_config.get(f"{split_name}_csv")
            if configured_path is None and split_name == "validation":
                splits[split_name] = ()
                source_paths[split_name] = None
                continue
            if not isinstance(configured_path, str) or not configured_path.strip():
                raise ValueError(f"data.{split_name}_csv must be a non-empty path.")
            path = Path(configured_path).expanduser()
            if not path.is_absolute():
                path = package_root / path
            path = path.resolve()
            splits[split_name] = _read_csv(path)
            source_paths[split_name] = str(path)
    else:
        raise ValueError("data.source must be 'demo' or 'csv'.")
    for split_name, interactions in splits.items():
        splits[split_name] = validate_interactions(interactions, num_users, num_items, split_name)
    if not splits["train"]:
        raise ValueError("The training split must not be empty.")
    _check_disjoint_splits(splits)
    return splits, num_users, num_items, source_paths


def _fingerprint(interactions: Sequence[Interaction]) -> str:
    records = sorted((item.user_id, item.item_id, item.rating, item.timestamp) for item in interactions)
    encoded = json.dumps(records, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_benchmark(
    config_path: Union[str, Path],
    *,
    include_cornac: Optional[bool] = None,
    split: Optional[str] = None,
    output_dir: Optional[str] = None,
    models: Optional[Sequence[object]] = None,
) -> Dict[str, Path]:
    config_file = Path(config_path).expanduser().resolve()
    config = _read_config(config_file)
    package_root = config_file.parent.parent
    run_name = config["run_name"]
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be a non-empty string.")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer.")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be in [0, 2**32 - 1].")
    seed = int(seed)
    protocol_config = _validate_keys(
        config["protocol"],
        ("ks", "positive_threshold", "exclude_seen", "empty_ground_truth_policy", "cold_start_policy"),
        "protocol",
    )
    protocol = EvaluationProtocol(**protocol_config)
    benchmark_config = _validate_keys(config["benchmark"], ("include_cornac", "split"), "benchmark")
    configured_cornac = benchmark_config.get("include_cornac", False)
    if not isinstance(configured_cornac, bool):
        raise TypeError("benchmark.include_cornac must be a boolean.")
    if include_cornac is not None and not isinstance(include_cornac, bool):
        raise TypeError("include_cornac must be a boolean or None.")
    use_cornac = configured_cornac if include_cornac is None else include_cornac
    selected_split = benchmark_config.get("split", "test") if split is None else split
    if selected_split not in ("validation", "test"):
        raise ValueError("The evaluation split must be 'validation' or 'test'.")
    configured_output = config["output_dir"]
    if not isinstance(configured_output, str) or not configured_output.strip():
        raise ValueError("output_dir must be a non-empty path.")
    if output_dir is None:
        output_base = Path(configured_output).expanduser()
        if not output_base.is_absolute():
            output_base = package_root / output_base
    else:
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("The output_dir override must be a non-empty path.")
        output_base = Path(output_dir).expanduser()
    output_base = output_base.resolve()
    if output_base.exists() and not output_base.is_dir():
        raise ValueError("The output directory path points to a file.")
    splits, num_users, num_items, source_paths = _resolve_data(config["data"], package_root)
    if not splits[selected_split]:
        raise ValueError(f"The selected {selected_split} split is empty.")
    random.seed(seed)
    if models is None:
        benchmark_models = [DemoPopularityModel(positive_threshold=protocol.positive_threshold)]
    else:
        if isinstance(models, (str, bytes)) or not isinstance(models, Sequence):
            raise TypeError("models must be a sequence of model implementations.")
        benchmark_models = list(models)
        if not benchmark_models:
            raise ValueError("At least one model is required.")
    if use_cornac:
        benchmark_models.append(CornacMostPopAdapter(positive_threshold=protocol.positive_threshold, seed=seed))
    names = []
    for model in benchmark_models:
        name = getattr(model, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every model must have a non-empty name.")
        if not callable(getattr(model, "fit", None)):
            raise TypeError(f"Model {name!r} must implement fit(train_interactions, num_users, num_items).")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("Model names must be unique within a benchmark run.")
    evaluator = Evaluator(protocol)
    results: List[EvaluationResult] = []
    for model in benchmark_models:
        started = perf_counter()
        model.fit(splits["train"], num_users=num_users, num_items=num_items)
        fit_seconds = perf_counter() - started
        if not isfinite(fit_seconds):
            raise RuntimeError("The training timer returned a non-finite duration.")
        result = evaluator.evaluate(
            model=model,
            train_interactions=splits["train"],
            test_interactions=splits[selected_split],
            num_users=num_users,
            num_items=num_items,
            split_name=selected_split,
        )
        results.append(replace(result, fit_seconds=fit_seconds))
    created_at = datetime.now(timezone.utc)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name.strip()).strip("._-")[:64] or "evaluation"
    run_directory = output_base / f"{safe_name}_{created_at.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"
    uses_cornac_adapter = any(isinstance(model, CornacMostPopAdapter) for model in benchmark_models)
    cornac_version = None
    if uses_cornac_adapter:
        try:
            cornac_version = metadata.version("cornac")
        except metadata.PackageNotFoundError:
            cornac_version = "unknown"
    run_config = {
        "created_at_utc": created_at.isoformat(),
        "config_file": str(config_file),
        "seed": seed,
        "split": selected_split,
        "model_names": names,
        "versions": {"python": platform.python_version(), "cornac": cornac_version},
        "effective_config": {
            "run_name": run_name.strip(),
            "seed": seed,
            "protocol": protocol.to_dict(),
            "data": {"source": config["data"]["source"], "num_users": num_users, "num_items": num_items},
            "benchmark": {"include_cornac": uses_cornac_adapter, "split": selected_split},
            "output_dir": str(output_base),
        },
        "data": {
            "paths": source_paths,
            "counts": {name: len(interactions) for name, interactions in splits.items()},
            "sha256": {name: _fingerprint(interactions) for name, interactions in splits.items()},
        },
        "models": [
            {
                "name": model.name,
                "class": f"{type(model).__module__}.{type(model).__qualname__}",
                "positive_threshold": getattr(model, "positive_threshold", None),
                "seed": getattr(model, "seed", None),
            }
            for model in benchmark_models
        ],
    }
    return save_results(results, run_directory, run_config)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate demo or injected recommendation models with a shared protocol.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "evaluation.json"))
    cornac_options = parser.add_mutually_exclusive_group()
    cornac_options.add_argument("--with-cornac", dest="include_cornac", action="store_const", const=True)
    cornac_options.add_argument("--without-cornac", dest="include_cornac", action="store_const", const=False)
    parser.set_defaults(include_cornac=None)
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    try:
        paths = run_benchmark(
            args.config,
            include_cornac=args.include_cornac,
            split=args.split,
            output_dir=args.output_dir,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
