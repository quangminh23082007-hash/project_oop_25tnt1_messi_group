from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib import import_module, metadata
from numbers import Integral
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from src.contracts.dto import DatasetSplits, GameItem, Interaction, ItemFeatures, ModelConfig
from src.contracts.exceptions import (
    ConfigurationError,
    DataValidationError,
    EmptyDatasetError,
    TrainingError,
    UnknownEntityError,
)
from src.contracts.interfaces import BaseRecommender, checked_indices
from src.evaluation.evaluator import Evaluator
from src.evaluation.protocol import EvaluationProtocol, validate_interactions
from src.evaluation.result import save_results, summarize_result


ROOT = Path(__file__).resolve().parents[1]
KINDS = ("demo_popularity", "popularity", "bpr", "text_bpr", "cornac_most_pop")
PARAMETERS = (
    "n_factors", "learning_rate", "reg", "n_epochs",
    "random_seed", "patience", "text_reg",
)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable.")


def _json_copy(value):
    try:
        return json.loads(json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        ))
    except (TypeError, ValueError, OverflowError) as error:
        raise DataValidationError("Metadata must contain finite JSON values.") from error


def _hash(value):
    payload = json.dumps(
        _json_copy(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _keys(value, allowed, label, required=()):
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ConfigurationError(f"{label} must be an object with string keys.")

    unknown = set(value) - set(allowed)
    missing = set(required) - set(value)

    if unknown or missing:
        raise ConfigurationError(
            f"Invalid {label}: unknown={sorted(unknown)}, missing={sorted(missing)}."
        )
    return dict(value)


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ConfigurationError(f"{label} must be an integer >= {minimum}.")
    return int(value)


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} must be a nonempty string.")
    return value.strip()


def _read_config(path):
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)

    fields = (
        "run_name",
        "seed",
        "protocol",
        "data",
        "models",
        "benchmark",
        "output_dir",
    )
    return _keys(value, fields, "config", fields)


def _provider(spec, *args):
    spec = _keys(spec, ("provider", "kwargs"), "provider", ("provider",))
    address = _text(spec["provider"], "provider")

    module_name, separator, function_name = address.partition(":")
    if not separator or not module_name or not function_name:
        raise ConfigurationError("A provider must use module:function syntax.")

    kwargs = spec.get("kwargs", {})
    if not isinstance(kwargs, Mapping):
        raise ConfigurationError("Provider kwargs must be an object.")

    function = getattr(import_module(module_name), function_name)
    if not callable(function):
        raise ConfigurationError(f"{address} is not callable.")

    return function(*args, **_json_copy(dict(kwargs)))


@dataclass(frozen=True)
class _DemoMapper:
    raw_ids: tuple[str, ...]

    @property
    def size(self):
        return len(self.raw_ids)

    def contains_raw(self, raw_id):
        return raw_id in self.raw_ids

    def to_idx(self, raw_id):
        try:
            return self.raw_ids.index(raw_id)
        except ValueError as error:
            raise UnknownEntityError(f"Unknown demo ID: {raw_id}.") from error

    def to_raw(self, index):
        indices = checked_indices([index], self.size)
        return self.raw_ids[int(indices[0])]


@dataclass(frozen=True)
class _DemoDataset:
    split_name: str
    interactions: tuple[Interaction, ...]
    user_mapper: _DemoMapper
    item_mapper: _DemoMapper

    @property
    def num_users(self):
        return self.user_mapper.size

    @property
    def num_items(self):
        return self.item_mapper.size

    @property
    def num_interactions(self):
        return len(self.interactions)

    @property
    def mapping_fingerprint(self):
        return _hash([self.user_mapper.raw_ids, self.item_mapper.raw_ids])

    @property
    def fingerprint(self):
        rows = sorted(
            (int(r.user_id), int(r.item_id), float(r.rating), int(r.timestamp))
            for r in self.interactions
        )
        return _hash([self.mapping_fingerprint, self.split_name, rows])

    @property
    def review_fingerprint(self):
        return _hash([self.mapping_fingerprint, self.split_name, []])

    def get_user_seen_items(self, user_id):
        return frozenset(
            r.item_id
            for r in self.interactions
            if r.user_id == user_id
        )

    def get_user_positive_items(self, user_id):
        return frozenset(
            r.item_id
            for r in self.interactions
            if r.user_id == user_id and r.rating == 1.0
        )

    def get_game_by_raw_id(self, raw_game_id):
        index = self.item_mapper.to_idx(raw_game_id)
        return GameItem(raw_game_id, f"Demo Game {index}")

    def iter_reviews(self):
        return iter(())


def _demo_splits():
    users = _DemoMapper(tuple(f"demo-user-{i}" for i in range(5)))
    items = _DemoMapper(tuple(f"demo-game-{i}" for i in range(8)))

    rows = {
        "train": (
            (0, 0, 1), (0, 1, 1), (0, 2, 0),
            (1, 0, 1), (1, 3, 1), (1, 4, 0),
            (2, 1, 1), (2, 4, 1), (2, 5, 0),
            (3, 2, 1), (3, 5, 1), (3, 6, 0),
            (4, 7, 0),
        ),
        "validation": (
            (0, 3, 1),
            (1, 1, 1),
            (2, 2, 1),
            (3, 0, 1),
            (4, 6, 1),
        ),
        "test": (
            (0, 4, 1), (0, 5, 0),
            (1, 5, 1), (1, 6, 0),
            (2, 6, 1), (2, 7, 0),
            (3, 1, 1), (3, 7, 0),
            (4, 0, 1),
        ),
    }

    splits = {
        name: _DemoDataset(
            name,
            tuple(Interaction(*row) for row in values),
            users,
            items,
        )
        for name, values in rows.items()
    }

    return DatasetSplits(
        splits["train"],
        splits["validation"],
        splits["test"],
        {"source": "synthetic_demo", "has_review_text": False},
    )


class DemoPopularityModel(BaseRecommender):
    model_kind = "demo_popularity"
    supports_callback = False

    def __init__(self, name="DemoPopularity"):
        super().__init__(name)
        self._scores = None

    def fit(
        self,
        train_dataset,
        *,
        validation_score=None,
        item_features=None,
        verbose=False,
    ):
        self._is_fitted = False
        self._num_users = self._num_items = 0
        self._mapping_fingerprint = self._train_fingerprint = None
        self._scores = None

        if validation_score is not None or item_features is not None:
            raise ConfigurationError(
                "DemoPopularity does not use callbacks or features."
            )
        if not isinstance(verbose, bool):
            raise ConfigurationError("verbose must be boolean.")
        if (
            not isinstance(train_dataset, _DemoDataset)
            or train_dataset.split_name != "train"
        ):
            raise DataValidationError(
                "DemoPopularity requires the synthetic train Dataset."
            )

        rows = validate_interactions(
            train_dataset.interactions,
            train_dataset.num_users,
            train_dataset.num_items,
        )
        positives = [
            int(r.item_id)
            for r in rows
            if r.rating == 1.0
        ]
        if not positives:
            raise EmptyDatasetError("Demo train requires positive interactions.")

        self._scores = np.bincount(
            positives,
            minlength=train_dataset.num_items,
        ).astype(float)

        self._num_users = train_dataset.num_users
        self._num_items = train_dataset.num_items
        self._mapping_fingerprint = train_dataset.mapping_fingerprint
        self._train_fingerprint = train_dataset.fingerprint
        self._is_fitted = True
        return self

    def score_items(self, user_id, item_ids):
        self._check_fitted()
        self._check_user(user_id)
        return self._scores[checked_indices(item_ids, self.num_items)].copy()

    def config(self):
        return {}


def _load_splits(config, supplied):
    config = _keys(
        config,
        ("source", "provider", "kwargs"),
        "data",
        ("source",),
    )
    source = config["source"]

    if source == "provided":
        if supplied is None or set(config) != {"source"}:
            raise ConfigurationError(
                "provided data requires dataset_splits and no provider."
            )
        return supplied

    if supplied is not None:
        raise ConfigurationError(
            "Use data.source=provided for injected DatasetSplits."
        )

    if source == "demo":
        if set(config) != {"source"}:
            raise ConfigurationError(
                "Demo data does not accept provider settings."
            )
        return _demo_splits()

    if source == "provider":
        return _provider({
            "provider": config.get("provider"),
            "kwargs": config.get("kwargs", {}),
        })

    raise ConfigurationError(
        "data.source must be demo, provided, or provider."
    )


def _validate_splits(bundle):
    datasets = {}
    records = {}
    seen = set()
    train = getattr(bundle, "train", None)

    for name in ("train", "validation", "test"):
        dataset = getattr(bundle, name, None)
        split_name = getattr(dataset, "split_name", None)

        if not isinstance(split_name, str) or split_name != name:
            raise DataValidationError(f"Expected a {name} Dataset.")

        try:
            users = _integer(
                getattr(dataset, "num_users", None),
                f"{name}.num_users",
                1,
            )
            items = _integer(
                getattr(dataset, "num_items", None),
                f"{name}.num_items",
                1,
            )
            count = _integer(
                getattr(dataset, "num_interactions", None),
                f"{name}.count",
            )

            for field in (
                "mapping_fingerprint",
                "fingerprint",
                "review_fingerprint",
            ):
                _text(getattr(dataset, field, None), f"{name}.{field}")

            for mapper, expected in (
                ("user_mapper", users),
                ("item_mapper", items),
            ):
                size = _integer(
                    getattr(getattr(dataset, mapper, None), "size", None),
                    mapper,
                )
                if size != expected:
                    raise DataValidationError(
                        f"{name}.{mapper}.size does not match."
                    )

        except ConfigurationError as error:
            raise DataValidationError(str(error)) from error

        if name != "train" and (
            users != train.num_users
            or items != train.num_items
            or dataset.mapping_fingerprint != train.mapping_fingerprint
        ):
            raise DataValidationError(
                "All splits must use the frozen train mapping."
            )

        values = getattr(dataset, "interactions", None)
        if (
            isinstance(values, (str, bytes, bytearray))
            or not isinstance(values, Sequence)
        ):
            raise DataValidationError(
                "Dataset interactions must be an ordered sequence."
            )

        rows = validate_interactions(values, users, items, name)
        if len(rows) != count:
            raise DataValidationError(
                f"{name}.num_interactions does not match its records."
            )

        pairs = {
            (int(r.user_id), int(r.item_id))
            for r in rows
        }
        if seen.intersection(pairs):
            raise DataValidationError(
                "User-item pairs overlap between splits."
            )

        seen.update(pairs)
        datasets[name] = dataset
        records[name] = rows

    if (
        not records["train"]
        or not any(r.rating == 1.0 for r in records["train"])
    ):
        raise EmptyDatasetError("Train requires positive interactions.")

    if len({r.user_id for r in records["train"]}) != train.num_users:
        raise DataValidationError(
            "User mapping must be fitted from all train interactions."
        )

    if len({r.item_id for r in records["train"]}) != train.num_items:
        raise DataValidationError(
            "Item mapping must be fitted from all train interactions."
        )

    return datasets, records


def _model_specs(values, include_cornac):
    if not isinstance(values, list) or not values:
        raise ConfigurationError("models must be a nonempty list.")

    specs = []
    for value in values:
        spec = _keys(
            value,
            ("kind", "name", "params", "features", "early_stopping"),
            "model",
            ("kind", "name"),
        )

        if spec["kind"] not in KINDS:
            raise ConfigurationError(
                f"Unsupported model kind: {spec['kind']}."
            )

        spec["name"] = _text(spec["name"], "model.name")
        params = _keys(
            spec.get("params", {}),
            PARAMETERS,
            "model.params",
        )

        if spec["kind"] not in ("bpr", "text_bpr") and params:
            raise ConfigurationError(
                f"{spec['kind']} has no BPR hyperparameters."
            )

        if spec["kind"] == "bpr" and "text_reg" in params:
            raise ConfigurationError("text_reg applies only to TextBPR.")

        if (
            "early_stopping" in spec
            and not isinstance(spec["early_stopping"], bool)
        ):
            raise ConfigurationError(
                "Model early_stopping must be boolean."
            )

        if (
            spec.get("early_stopping") is True
            and spec["kind"] not in ("bpr", "text_bpr")
        ):
            raise ConfigurationError(
                "Only BPR/TextBPR support early stopping."
            )

        if (
            spec.get("features") is not None
            and spec["kind"] != "text_bpr"
        ):
            raise ConfigurationError("Only TextBPR accepts features.")

        spec["params"] = params
        specs.append(spec)

    if len({s["name"] for s in specs}) != len(specs):
        raise ConfigurationError("Benchmark labels must be unique.")

    if not include_cornac:
        specs = [
            s for s in specs
            if s["kind"] != "cornac_most_pop"
        ]
    elif not any(s["kind"] == "cornac_most_pop" for s in specs):
        specs.append({
            "kind": "cornac_most_pop",
            "name": "CornacMostPop",
            "params": {},
        })

    if (
        not specs
        or len({s["name"] for s in specs}) != len(specs)
    ):
        raise ConfigurationError(
            "Enabled model labels must be nonempty and unique."
        )

    return specs


def _create_model(spec, seed):
    kind = spec["kind"]

    if kind == "demo_popularity":
        return DemoPopularityModel(spec["name"])

    if kind == "cornac_most_pop":
        from .cornac_adapter import CornacMostPopAdapter
        return CornacMostPopAdapter(name=spec["name"], seed=seed)

    address = {
        "popularity": (
            "src.models.popularity",
            "PopularityRecommender",
        ),
        "bpr": (
            "src.models.matrix_factorization",
            "BPRMatrixFactorization",
        ),
        "text_bpr": (
            "src.models.text_bpr",
            "TextBPRMatrixFactorization",
        ),
    }[kind]

    try:
        model_type = getattr(import_module(address[0]), address[1])
    except ImportError as error:
        raise ImportError(
            "The public Model package is required. Integrate src/models from the "
            "Models contribution before selecting a real model."
        ) from error

    if kind == "popularity":
        return model_type()

    config = ModelConfig(**{
        "random_seed": seed,
        **spec["params"],
    })
    return model_type(config)


def _features(spec, train, overrides):
    label = spec["name"]

    if spec["kind"] != "text_bpr":
        if label in overrides:
            raise ConfigurationError(
                "Only TextBPR accepts injected features."
            )
        return None, None

    if label in overrides:
        supplied = overrides[label]
        source = {"type": "injected"}

    elif spec.get("features") is not None:
        supplied = _provider(spec["features"], train)
        source = {
            "type": "provider",
            **spec["features"],
        }

    else:
        raise ConfigurationError(
            f"{label} requires ItemFeatures or a feature provider."
        )

    report = None
    if isinstance(supplied, tuple):
        if (
            len(supplied) != 2
            or not isinstance(supplied[1], Mapping)
        ):
            raise DataValidationError(
                "Feature providers must return (ItemFeatures, report)."
            )
        supplied, report = supplied
        report = _json_copy(dict(report))

    if not isinstance(supplied, ItemFeatures):
        raise DataValidationError(
            "TextBPR requires the shared ItemFeatures DTO."
        )

    if (
        supplied.mapping_fingerprint != train.mapping_fingerprint
        or supplied.train_fingerprint != train.fingerprint
        or supplied.text_fingerprint != train.review_fingerprint
    ):
        raise DataValidationError(
            "Feature mapping/train/text fingerprints must match train."
        )

    ordered_ids = tuple(
        train.item_mapper.to_raw(i)
        for i in range(train.num_items)
    )
    if supplied.item_ids != ordered_ids:
        raise DataValidationError(
            "Feature rows must follow train item-mapping order."
        )

    encoder_name = _text(
        supplied.encoder_name,
        "feature encoder_name",
    )
    encoder, separator, revision = encoder_name.rpartition("@")

    if (
        not separator
        or not encoder.strip()
        or not revision.strip()
        or revision.strip().lower() in ("main", "master", "latest")
    ):
        raise ConfigurationError(
            "Feature encoder_name must include @fixed_revision."
        )

    values = np.asarray(supplied.values)
    if (
        values.ndim != 2
        or values.shape[0] != train.num_items
        or values.shape[1] == 0
        or values.dtype.kind not in "fiu"
        or not np.isfinite(values).all()
    ):
        raise DataValidationError(
            "Feature matrix must be finite and match train items."
        )

    header = {
        "shape": list(values.shape),
        "dtype": "<f8",
        "ordered_item_ids_sha256": _hash(supplied.item_ids),
        "mapping_fingerprint": supplied.mapping_fingerprint,
        "train_fingerprint": supplied.train_fingerprint,
        "text_fingerprint": supplied.text_fingerprint,
        "encoder_name": supplied.encoder_name,
    }

    matrix = np.ascontiguousarray(values, dtype="<f8")
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix))

    if not np.all(
        np.isclose(norms, 0, atol=1e-6)
        | np.isclose(norms, 1, atol=1e-5)
    ):
        raise DataValidationError(
            "Feature rows must have unit norm or be zero."
        )

    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")
    digest.update(memoryview(matrix).cast("B"))

    if report is not None:
        for key in (
            "mapping_fingerprint",
            "train_fingerprint",
            "text_fingerprint",
            "encoder_name",
        ):
            if key in report and report[key] != header[key]:
                raise DataValidationError(
                    f"Feature report {key} does not match ItemFeatures."
                )

    return supplied, {
        **header,
        "checksum_scheme": "sha256_float64_le_C_header_v1",
        "sha256": digest.hexdigest(),
        "source": source,
        "build_report": report,
    }


def _cohort_signature(result):
    return _hash({
        "evaluated": [
            [
                u.user_id,
                u.candidate_count,
                sorted(u.relevant_items),
            ]
            for u in result.per_user
        ],
        "skipped": [
            [u.user_id, u.reason]
            for u in result.skipped_users
        ],
        "metrics": sorted(result.metrics),
    })


def _version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def run_benchmark(
    config_path,
    *,
    dataset_splits=None,
    models: Mapping[str, object] | None = None,
    item_features: Mapping[str, object] | None = None,
    include_cornac=None,
    split=None,
    output_dir=None,
):
    config_file = Path(config_path).expanduser().resolve()
    config = _read_config(config_file)

    seed = _integer(config["seed"], "seed")
    if seed > 2**32 - 1:
        raise ConfigurationError("seed must not exceed 2**32 - 1.")

    run_name = _text(config["run_name"], "run_name")

    protocol_fields = (
        "ks",
        "positive_threshold",
        "exclude_seen",
        "empty_ground_truth_policy",
        "cold_start_policy",
    )
    protocol = EvaluationProtocol(
        **_keys(config["protocol"], protocol_fields, "protocol")
    )

    if (
        not protocol.exclude_seen
        or protocol.empty_ground_truth_policy != "skip"
        or protocol.cold_start_policy != "skip"
    ):
        raise ConfigurationError(
            "The v1 benchmark requires exclude_seen=True and skip policies."
        )

    benchmark = _keys(
        config["benchmark"],
        (
            "include_cornac",
            "split",
            "early_stopping",
            "selection_metric",
            "history",
            "verbose",
        ),
        "benchmark",
    )

    use_cornac = (
        benchmark.get("include_cornac", False)
        if include_cornac is None
        else include_cornac
    )
    early_stopping = benchmark.get("early_stopping", True)
    verbose = benchmark.get("verbose", False)

    if any(
        not isinstance(v, bool)
        for v in (use_cornac, early_stopping, verbose)
    ):
        raise ConfigurationError(
            "Cornac/early_stopping/verbose options must be boolean."
        )

    selected = (
        benchmark.get("split", "test")
        if split is None
        else split
    )
    history = benchmark.get("history", "train")

    if selected not in ("validation", "test"):
        raise ConfigurationError("split must be validation or test.")

    if history not in ("train", "train_validation"):
        raise ConfigurationError(
            "history must be train or train_validation."
        )

    if selected != "test" and history != "train":
        raise ConfigurationError(
            "Validation history can be added only when evaluating test."
        )

    bundle = _load_splits(config["data"], dataset_splits)
    datasets, records = _validate_splits(bundle)
    preparation_report = _json_copy(getattr(bundle, "report", {}))

    if not records[selected]:
        raise EmptyDatasetError(
            f"The selected {selected} split is empty."
        )

    specs = _model_specs(config["models"], use_cornac)
    source = config["data"]["source"]

    if source != "demo" and any(
        s["kind"] == "demo_popularity"
        for s in specs
    ):
        raise ConfigurationError(
            "DemoPopularity is restricted to synthetic demo data."
        )

    if source == "demo" and any(
        s["kind"] == "text_bpr"
        for s in specs
    ):
        raise ConfigurationError(
            "TextBPR requires real train review text."
        )

    for value, label in (
        (models, "models"),
        (item_features, "item_features"),
    ):
        if value is not None and not isinstance(value, Mapping):
            raise ConfigurationError(
                f"{label} must map benchmark labels to objects."
            )

    model_overrides = dict(models or {})
    feature_overrides = dict(item_features or {})
    labels = {s["name"] for s in specs}

    if (set(model_overrides) | set(feature_overrides)) - labels:
        raise ConfigurationError(
            "Injected labels must exist among enabled models."
        )

    evaluator = Evaluator(protocol)
    selection_metric = benchmark.get(
        "selection_metric",
        f"NDCG@{protocol.max_k}",
    )

    if selection_metric not in {m.name for m in evaluator.metrics}:
        raise ConfigurationError(
            "selection_metric must be a configured metric."
        )

    plans = []
    identities = set()

    for spec in specs:
        label = spec["name"]

        ModelConfig(**{
            "random_seed": seed,
            **spec["params"],
        })

        model = (
            model_overrides[label]
            if label in model_overrides
            else _create_model(spec, seed)
        )

        if id(model) in identities:
            raise ConfigurationError(
                "Each benchmark label requires a separate model object."
            )
        identities.add(id(model))

        _text(getattr(model, "name", None), f"{label}.name")

        for method in ("fit", "score_items", "config"):
            if not callable(getattr(model, method, None)):
                raise ConfigurationError(
                    f"{label} must provide {method}()."
                )

        if (
            getattr(model, "model_kind", spec["kind"])
            != spec["kind"]
        ):
            raise ConfigurationError(
                f"{label} model_kind does not match its configuration."
            )

        actual = _json_copy(model.config())
        if not isinstance(actual, dict):
            raise ConfigurationError(
                "model.config() must return an object."
            )

        if label in model_overrides:
            for key, value in spec["params"].items():
                if key not in actual or actual[key] != value:
                    raise ConfigurationError(
                        f"{label} does not match requested parameter {key}."
                    )

        use_validation = (
            spec.get("early_stopping", early_stopping)
            and spec["kind"] in ("bpr", "text_bpr")
        )

        if (
            use_validation
            and getattr(model, "supports_callback", True) is not True
        ):
            raise ConfigurationError(
                f"{label} does not support validation callbacks."
            )

        features, feature_report = _features(
            spec,
            datasets["train"],
            feature_overrides,
        )

        plans.append({
            "spec": spec,
            "model": model,
            "features": features,
            "feature_report": feature_report,
            "use_validation": use_validation,
        })

    output_value = (
        config["output_dir"]
        if output_dir is None
        else output_dir
    )

    if (
        not isinstance(output_value, (str, Path))
        or not str(output_value).strip()
    ):
        raise ConfigurationError(
            "output_dir must be a nonempty path."
        )

    base = Path(output_value).expanduser()
    if not base.is_absolute():
        base = (
            ROOT
            if output_dir is None
            else Path.cwd()
        ) / base

    base = base.resolve()
    if base.exists() and not base.is_dir():
        raise ConfigurationError(
            "output_dir must point to a directory."
        )

    results = []
    reports = []
    cohorts = {}

    for plan in plans:
        spec = plan["spec"]
        label = spec["name"]
        model = plan.pop("model")
        features = plan.pop("features")
        objective_trace = []

        def validation_score(candidate, trace=objective_trace):
            result = evaluator.evaluate(
                candidate,
                datasets["train"],
                datasets["validation"],
            )
            if selection_metric not in result.metrics:
                raise EmptyDatasetError(
                    "Early stopping requires eligible validation users."
                )

            value = float(result.metrics[selection_metric])
            trace.append(value)
            return value

        options = {"verbose": verbose}

        if plan["use_validation"]:
            options["validation_score"] = validation_score

        if features is not None:
            options["item_features"] = features

        started = perf_counter()
        model.fit(datasets["train"], **options)
        fit_seconds = perf_counter() - started

        if plan["use_validation"] and not objective_trace:
            raise TrainingError(
                f"{label} did not call its validation callback."
            )

        validation = evaluator.evaluate(
            model,
            datasets["train"],
            datasets["validation"],
        )

        result = (
            validation
            if selected == "validation"
            else evaluator.evaluate(
                model,
                datasets["train"],
                datasets["test"],
                additional_seen_dataset=(
                    datasets["validation"]
                    if history == "train_validation"
                    else None
                ),
            )
        )

        if not result.per_user:
            raise EmptyDatasetError(
                f"No eligible users remain in {selected}."
            )

        for scored in (validation, result):
            signature = _cohort_signature(scored)
            previous = cohorts.setdefault(
                scored.split_name,
                signature,
            )
            if signature != previous:
                raise DataValidationError(
                    "Models must be evaluated on the same cohort and candidates."
                )

        results.append(
            replace(
                result,
                model_name=label,
                fit_seconds=fit_seconds,
            )
        )

        actual_config = _json_copy(model.config())
        best_epoch = getattr(model, "best_epoch", None)

        if best_epoch is not None:
            best_epoch = _integer(
                best_epoch,
                f"{label}.best_epoch",
                1,
            )

        if plan["use_validation"] and best_epoch is None:
            raise TrainingError(
                f"{label} must expose the selected best_epoch."
            )

        reports.append({
            "label": label,
            "public_model_name": model.name,
            "kind": spec["kind"],
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "requested": spec,
            "actual_config": actual_config,
            "actual_random_seed": actual_config.get("random_seed"),
            "injected_model": label in model_overrides,
            "early_stopping_used": plan["use_validation"],
            "selection_metric": (
                selection_metric
                if plan["use_validation"]
                else None
            ),
            "validation_callback_count": len(objective_trace),
            "validation_objective_trace": objective_trace,
            "best_epoch": best_epoch,
            "training_history": getattr(model, "history", []),
            "validation_summary": summarize_result(validation),
            "features": plan["feature_report"],
        })

    created = datetime.now(timezone.utc)
    safe_name = (
        re.sub(r"[^A-Za-z0-9._-]+", "_", run_name)
        .strip("._-")[:64]
        or "benchmark"
    )
    directory = base / (
        f"{safe_name}_"
        f"{created.strftime('%Y%m%dT%H%M%S%fZ')}_"
        f"{uuid4().hex[:8]}"
    )

    metadata_value = _json_copy({
        "created_at_utc": created.isoformat(),
        "config_file": str(config_file),
        "requested_config": config,
        "effective_split": selected,
        "effective_history": history,
        "effective_include_cornac": any(
            s["kind"] == "cornac_most_pop"
            for s in specs
        ),
        "effective_protocol": protocol.to_dict(),
        "default_model_seed": seed,
        "output_directory": str(directory),
        "versions": {
            "python": platform.python_version(),
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
            "cornac": _version("cornac"),
            "sentence_transformers": _version("sentence-transformers"),
        },
        "data": {
            "source": source,
            "report": preparation_report,
            "num_users": int(datasets["train"].num_users),
            "num_items": int(datasets["train"].num_items),
            "splits": {
                name: {
                    "num_interactions": len(records[name]),
                    "positive_interactions": sum(
                        r.rating == 1.0
                        for r in records[name]
                    ),
                    "mapping_fingerprint": dataset.mapping_fingerprint,
                    "fingerprint": dataset.fingerprint,
                    "review_fingerprint": dataset.review_fingerprint,
                }
                for name, dataset in datasets.items()
            },
        },
        "cohort_signatures": cohorts,
        "models": reports,
    })

    return save_results(
        results,
        directory,
        metadata_value,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark game recommenders using Contracts v1."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "evaluation.json"),
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
    )
    parser.add_argument("--output-dir")

    cornac = parser.add_mutually_exclusive_group()
    cornac.add_argument(
        "--with-cornac",
        dest="include_cornac",
        action="store_true",
    )
    cornac.add_argument(
        "--without-cornac",
        dest="include_cornac",
        action="store_false",
    )
    parser.set_defaults(include_cornac=None)

    args = parser.parse_args(argv)

    try:
        paths = run_benchmark(
            args.config,
            include_cornac=args.include_cornac,
            split=args.split,
            output_dir=args.output_dir,
        )
    except Exception as error:
        print(
            f"Benchmark failed: {error}",
            file=sys.stderr,
        )
        return 1

    for label, path in paths.items():
        print(f"{label}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())