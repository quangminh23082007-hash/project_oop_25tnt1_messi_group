from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter

from .pretrained_report import (
    PretrainedReportError,
    load_recorded_validation,
    write_pretrained_results,
)


ROOT = Path(__file__).resolve().parents[1]
FIELDS = {
    "run_name", "model", "model_dir", "reference_path", "user_ids",
    "top_k", "exclude_played", "allow_cold_start_fallback",
    "extra_seen", "output_dir",
}
MODEL_TYPES = {
    "bpr": "BPRMatrixFactorization",
    "text_bpr": "TextBPRMatrixFactorization",
}


def _path(value):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise PretrainedReportError("Paths must be nonempty strings or Path objects.")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_object(path):
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8-sig"))
        json.dumps(value, allow_nan=False)
    except (UnicodeError, ValueError, TypeError) as error:
        raise PretrainedReportError(f"Invalid JSON in {path}.") from error
    if not isinstance(value, dict):
        raise PretrainedReportError(f"{path} must contain a JSON object.")
    return value, hashlib.sha256(raw).hexdigest()


def _ids(value, name, allow_empty=False):
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise PretrainedReportError(f"{name} must be a list of nonempty raw IDs.")
    if not value and not allow_empty:
        raise PretrainedReportError(f"{name} must not be empty.")
    if len(value) != len(set(value)):
        raise PretrainedReportError(f"{name} must not contain duplicate IDs.")
    return value


def _config(config_path, user_ids, output_dir):
    config_file = _path(config_path)
    config, checksum = _read_object(config_file)
    missing, unknown = FIELDS - set(config), set(config) - FIELDS
    if missing or unknown:
        raise PretrainedReportError(
            f"Config fields missing={sorted(missing)}, unknown={sorted(unknown)}."
        )
    requested = dict(config)
    if user_ids is not None:
        config["user_ids"] = user_ids
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    for name in ("run_name", "model"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise PretrainedReportError(f"{name} must be a nonempty string.")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", config["run_name"]) is None:
        raise PretrainedReportError("run_name must use 1-80 letters, digits, dots, underscores or hyphens.")
    config["model"] = config["model"].replace("-", "_")
    if config["model"] not in MODEL_TYPES:
        raise PretrainedReportError("model must be bpr or text_bpr.")
    top_k = config["top_k"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise PretrainedReportError("top_k must be a positive integer.")
    for name in ("exclude_played", "allow_cold_start_fallback"):
        if not isinstance(config[name], bool):
            raise PretrainedReportError(f"{name} must be boolean.")
    _ids(config["user_ids"], "user_ids")
    _ids(config["extra_seen"], "extra_seen", allow_empty=True)
    for name in ("model_dir", "reference_path", "output_dir"):
        config[name] = str(_path(config[name]))
    destination = Path(config["output_dir"])
    if destination.exists() and not destination.is_dir():
        raise PretrainedReportError("output_dir must point to a directory.")
    return config_file, checksum, requested, config


def _bundle_context(folder, user_ids, selected_model):
    info_path = folder / "model_info.json"
    info, checksum = _read_object(info_path)
    if info.get("format_version") != 3 or info.get("representation") != "compact":
        raise PretrainedReportError("Expected a format-3 compact CPU bundle.")
    if info.get("model_type") != MODEL_TYPES[selected_model]:
        raise PretrainedReportError("Selected model does not match the bundle type.")
    users = _ids(info.get("user_ids"), "bundle.user_ids")
    items = _ids(info.get("item_ids"), "bundle.item_ids")
    catalog = info.get("catalog")
    if not isinstance(catalog, dict) or any(
        not isinstance(row, dict) for row in catalog.values()
    ):
        raise PretrainedReportError("Bundle catalog must map raw IDs to objects.")
    keys = (
        "format_version", "representation", "model_type", "data_fingerprint",
        "weights_sha256", "best_epoch", "config", "data_config",
        "feature_info", "training_info", "data_limitations",
    )
    bundle_metadata = {key: info.get(key) for key in keys}
    bundle_metadata.update(num_users=len(users), num_items=len(items))
    selected_users = set(user_ids)
    known_users = selected_users.intersection(users)
    item_order = {raw_id: index for index, raw_id in enumerate(items)}
    return bundle_metadata, checksum, known_users, item_order


def _load_model(folder, allow_fallback):
    try:
        from src.models.pretrained import PretrainedRecommender
    except ImportError as error:
        raise PretrainedReportError(
            "Integrate the complete src/models package from the models branch first."
        ) from error
    return PretrainedRecommender.load(
        folder,
        allow_cold_start_fallback=allow_fallback,
    )


def _check_recommendations(
    rows, repeated, user_id, known_user, history, item_order, config
):
    from src.contracts.dto import RecommendationResult

    if not isinstance(rows, list) or any(
        not isinstance(row, RecommendationResult) for row in rows
    ):
        raise PretrainedReportError("Model must return shared RecommendationResult rows.")
    ids, scores = [], []
    fallback = not known_user
    for rank, row in enumerate(rows, 1):
        if isinstance(row.rank, bool) or not isinstance(row.rank, Integral) or row.rank != rank:
            raise PretrainedReportError("Recommendation ranks must start at 1 and be contiguous.")
        if not isinstance(row.game_id, str) or row.game_id not in item_order:
            raise PretrainedReportError("Recommendation contains an unknown raw game ID.")
        if isinstance(row.predicted_score, bool) or not isinstance(row.predicted_score, Real):
            raise PretrainedReportError("Recommendation scores must be real numbers.")
        score = float(row.predicted_score)
        if not math.isfinite(score):
            raise PretrainedReportError("Recommendation scores must be finite.")
        if not isinstance(row.is_fallback, bool) or row.is_fallback != fallback:
            raise PretrainedReportError("Incorrect fallback flag.")
        expected_name = "Popularity" if fallback else config["model"]
        if row.model_name != expected_name:
            raise PretrainedReportError("Incorrect model name in recommendation.")
        ids.append(row.game_id)
        scores.append(score)
    extra = set(config["extra_seen"]).intersection(item_order)
    excluded = history | extra if config["exclude_played"] else set()
    candidates = len(item_order) - len(excluded)
    order = sorted(
        range(len(rows)),
        key=lambda index: (-scores[index], item_order[ids[index]]),
    )
    checks = {
        "unique_items": len(ids) == len(set(ids)),
        "finite_scores": True,
        "rank_order": order == list(range(len(rows))),
        "history_excluded": not set(ids).intersection(excluded),
        "fallback_label": True,
        "expected_count": len(rows) == min(config["top_k"], candidates),
        "repeatable": rows == repeated,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise PretrainedReportError(f"Inference checks failed for {user_id!r}: {failed}.")
    output = [{"user_id": user_id, **asdict(row)} for row in rows]
    audit = {
        "user_id": user_id,
        "known_user": known_user,
        "is_fallback": fallback,
        "returned_items": len(rows),
        "candidate_count": candidates,
        "history_count": len(history),
        "history_filter_applied": config["exclude_played"],
        "extra_seen_in_mapping": len(extra),
        "checks": checks,
    }
    return output, audit


def run_pretrained(config_path, *, user_ids=None, output_dir=None):
    config_file, config_sha256, requested, config = _config(
        config_path, user_ids, output_dir
    )
    folder = Path(config["model_dir"])
    bundle, info_sha256, known_users, item_order = _bundle_context(
        folder, config["user_ids"], config["model"]
    )
    unknown = [user for user in config["user_ids"] if user not in known_users]
    if unknown and not config["allow_cold_start_fallback"]:
        raise PretrainedReportError(
            f"Users absent from bundle mapping: {unknown}. "
            "Choose source IDs or enable explicit popularity fallback."
        )
    recorded = load_recorded_validation(
        config["reference_path"],
        selected_model=config["model"],
        bundle_metadata=bundle,
    )
    started = perf_counter()
    model = _load_model(folder, config["allow_cold_start_fallback"])
    load_seconds = perf_counter() - started
    if model.model_name != config["model"] or (
        model.provenance.get("data_fingerprint") != bundle["data_fingerprint"]
    ):
        raise PretrainedReportError("Loaded model does not match recorded provenance.")
    with (folder / "model_info.json").open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != info_sha256:
        raise PretrainedReportError("Bundle metadata changed during loading.")
    recommendations, users = [], []
    started = perf_counter()
    for user_id in config["user_ids"]:
        known = user_id in known_users
        history = {
            game.item_id for game in model.get_user_played_games(user_id)
        } if known else set()
        if not history.issubset(item_order):
            raise PretrainedReportError("History contains items outside the bundle mapping.")
        options = {
            "top_k": config["top_k"],
            "exclude_played": config["exclude_played"],
            "extra_seen": tuple(config["extra_seen"]),
        }
        rows = model.recommend_games(user_id, **options)
        repeated = model.recommend_games(user_id, **options)
        output, audit = _check_recommendations(
            rows, repeated, user_id, known, history, item_order, config
        )
        recommendations.extend(output)
        users.append(audit)
    inference_seconds = perf_counter() - started
    checks = {
        "all_passed": True,
        "users": users,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "recommendation_calls_per_user": 2,
        "recomputed_metrics": False,
    }
    try:
        numpy_version = metadata.version("numpy")
    except metadata.PackageNotFoundError:
        numpy_version = None
    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "pretrained_inference",
        "recomputed_metrics": False,
        "config_file": str(config_file),
        "config_sha256": config_sha256,
        "requested_config": requested,
        "effective_config": config,
        "versions": {"python": platform.python_version(), "numpy": numpy_version},
        "bundle": {
            **bundle,
            "path": str(folder),
            "model_info_sha256": info_sha256,
            "weights_verified_by_loader": True,
        },
        "validation_source": recorded["source"],
    }
    return write_pretrained_results(
        config["output_dir"],
        config["run_name"],
        recommendations=recommendations,
        inference_checks=checks,
        recorded_validation=recorded,
        run_config=run_config,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run CPU recommendations and import original validation reports."
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "evaluation_pretrained.json")
    )
    parser.add_argument("--user", action="append", dest="user_ids")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    try:
        paths = run_pretrained(
            args.config, user_ids=args.user_ids, output_dir=args.output_dir
        )
    except Exception as error:
        print(f"Pretrained run failed: {error}", file=sys.stderr)
        return 1
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
