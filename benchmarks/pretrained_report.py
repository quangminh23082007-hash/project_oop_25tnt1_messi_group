from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PretrainedReportError(ValueError):
    pass


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise PretrainedReportError(f"{name} must be an object.")
    return value


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PretrainedReportError(f"{name} must be an integer >= {minimum}.")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PretrainedReportError(f"{name} must be a nonempty string.")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PretrainedReportError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise PretrainedReportError("Report values must be finite JSON-compatible data.") from exc


def _metric(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PretrainedReportError(f"{name} must be a finite number in [0, 1].")
    try:
        valid = math.isfinite(value) and 0 <= value <= 1
    except OverflowError:
        valid = False
    if not valid:
        raise PretrainedReportError(f"{name} must be a finite number in [0, 1].")
    return float(value)


def _validation(value: Any, name: str, users: int, items: int) -> dict:
    report = _object(value, name)
    evaluated = _integer(report.get("evaluated_users"), f"{name}.evaluated_users")
    eligible = _integer(report.get("eligible_users"), f"{name}.eligible_users")
    catalog = _integer(report.get("candidate_catalog_size"), f"{name}.candidate_catalog_size")
    if not evaluated <= eligible <= users or catalog != items:
        raise PretrainedReportError(f"{name} has inconsistent user/catalog counts.")
    return {
        "recall_at_10": _metric(report.get("recall@10"), f"{name}.recall@10"),
        "ndcg_at_10": _metric(report.get("ndcg@10"), f"{name}.ndcg@10"),
        "evaluated_users": evaluated,
        "eligible_users": eligible,
        "candidate_catalog_size": catalog,
        "candidate_policy": _text(report.get("candidate_policy"), f"{name}.candidate_policy"),
    }


def load_recorded_validation(
    reference_path: str | Path, *, selected_model: str, bundle_metadata: dict
) -> dict:
    if selected_model not in ("bpr", "text_bpr"):
        raise PretrainedReportError("selected_model must be 'bpr' or 'text_bpr'.")
    path = Path(reference_path).expanduser().resolve()
    try:
        raw = path.read_bytes()
        reference = _object(json.loads(raw.decode("utf-8-sig")), "reference")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PretrainedReportError(f"Cannot read validation reference: {path}") from exc
    _json(reference)
    if reference.get("contract_revision") != "1.1":
        raise PretrainedReportError("The reference must declare contract_revision '1.1'.")
    if reference.get("delivery_mode") != "pretrained-existing-run":
        raise PretrainedReportError("The reference must describe a pretrained existing run.")
    models = _object(reference.get("models"), "reference.models")
    bundle = _object(bundle_metadata, "bundle_metadata")
    _json(bundle)
    rows = []
    signatures = []
    popularity_reports = []
    model_types = {"bpr": "BPRMatrixFactorization", "text_bpr": "TextBPRMatrixFactorization"}
    selected_entry = None
    for key, model_type in model_types.items():
        entry = _object(models.get(key), f"models.{key}")
        if entry.get("format_version") != 3 or entry.get("representation") != "compact":
            raise PretrainedReportError(f"models.{key} must describe a format-3 compact bundle.")
        if entry.get("model_type") != model_type:
            raise PretrainedReportError(f"models.{key} has an unexpected model_type.")
        users = _integer(entry.get("num_users"), f"models.{key}.num_users")
        items = _integer(entry.get("num_items"), f"models.{key}.num_items")
        fingerprint = _text(entry.get("data_fingerprint"), f"models.{key}.data_fingerprint")
        weights = _digest(entry.get("weights_sha256"), f"models.{key}.weights_sha256")
        epoch = _integer(entry.get("best_epoch"), f"models.{key}.best_epoch")
        _object(entry.get("config"), f"models.{key}.config")
        data_config = _object(entry.get("data_config"), f"models.{key}.data_config")
        training = _object(entry.get("training_info"), f"models.{key}.training_info")
        seed = _integer(training.get("validation_seed"), f"models.{key}.validation_seed", 0)
        if training.get("validation_k") != 10:
            raise PretrainedReportError(f"models.{key} must report validation at K=10.")
        report = _object(entry.get("validation_report"), f"models.{key}.validation_report")
        if report.get("model") != key.replace("_", "-") or report.get("best_epoch") != epoch:
            raise PretrainedReportError(f"models.{key} validation model/epoch does not match.")
        metrics = _validation(report.get("validation"), f"models.{key}.validation", users, items)
        maximum = _integer(training.get("validation_max_users"), f"models.{key}.validation_max_users")
        if metrics["evaluated_users"] > maximum:
            raise PretrainedReportError(f"models.{key} exceeds its declared validation sample.")
        popularity = _validation(report.get("popularity_validation"), f"models.{key}.popularity", users, items)
        cohort_keys = ("evaluated_users", "eligible_users", "candidate_catalog_size", "candidate_policy")
        if any(metrics[name] != popularity[name] for name in cohort_keys):
            raise PretrainedReportError(f"models.{key} and its Popularity baseline use different cohorts.")
        signatures.append((fingerprint, users, items, data_config, seed, maximum,
                           tuple(metrics[name] for name in cohort_keys)))
        popularity_reports.append(popularity)
        rows.append({"model": key, "split": "validation", "recomputed": False,
                     "primary_model_selected": key == selected_model, "best_epoch": epoch, **metrics,
                     "data_fingerprint": fingerprint, "weights_sha256": weights})
        if key == selected_model:
            selected_entry = entry
            _integer(bundle.get("best_epoch"), "bundle.best_epoch")
            for name in ("format_version", "representation", "model_type", "data_fingerprint",
                         "weights_sha256", "best_epoch", "config", "data_config",
                         "feature_info", "training_info"):
                if bundle.get(name) != entry.get(name):
                    raise PretrainedReportError(f"Selected bundle does not match reference field {name}.")
            for name, ids_name, expected in (("num_users", "user_ids", users), ("num_items", "item_ids", items)):
                count = bundle.get(name)
                if count is None and isinstance(bundle.get(ids_name), list):
                    count = len(bundle[ids_name])
                if _integer(count, f"bundle.{name}") != expected:
                    raise PretrainedReportError(f"Selected bundle does not match reference field {name}.")
    if signatures[0] != signatures[1] or popularity_reports[0] != popularity_reports[1]:
        raise PretrainedReportError("Recorded models do not share a comparable validation protocol/baseline.")
    rows.append({"model": "popularity", "split": "validation", "recomputed": False,
                 "primary_model_selected": False, "best_epoch": None, **popularity_reports[0],
                 "data_fingerprint": selected_entry["data_fingerprint"], "weights_sha256": None})
    return {
        "source": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()},
        "recomputed": False,
        "selected_model": selected_model,
        "rows": rows,
        "provenance": {
            "contract_revision": reference["contract_revision"],
            "delivery_mode": reference["delivery_mode"],
            "data_branch_reference": reference.get("data_branch_reference"),
            "data_config": selected_entry["data_config"],
            "training_info": selected_entry["training_info"],
            "feature_info": selected_entry.get("feature_info"),
            "data_limitations": selected_entry.get("data_limitations", []),
            "note": reference.get("note"),
        },
    }


def _csv(rows: list[dict], fields: tuple[str, ...] = ()) -> str:
    fields = list(dict.fromkeys([*fields, *(key for row in rows for key in row)]))
    if any(not isinstance(key, str) for key in fields):
        raise PretrainedReportError("CSV field names must be strings.")
    output = io.StringIO(newline="")
    if not fields:
        return ""
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _json(value) if isinstance(value, (dict, list, tuple)) else value
                         for key, value in row.items()})
    return output.getvalue()


def write_pretrained_results(
    output_root: str | Path,
    run_name: str,
    *,
    recommendations: list[dict],
    inference_checks: dict,
    recorded_validation: dict,
    run_config: dict,
) -> dict[str, Path]:
    if not isinstance(run_name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_name) is None:
        raise PretrainedReportError("run_name must contain 1-80 letters, digits, dots, underscores or hyphens.")
    if not isinstance(recommendations, list) or any(not isinstance(row, dict) for row in recommendations):
        raise PretrainedReportError("recommendations must be a list of objects.")
    checks = _object(inference_checks, "inference_checks")
    recorded = _object(recorded_validation, "recorded_validation")
    config = _object(run_config, "run_config")
    rows = recorded.get("rows")
    if recorded.get("recomputed") is not False or not isinstance(rows, list) or not rows:
        raise PretrainedReportError("recorded_validation must contain imported, unrecomputed rows.")
    if any(not isinstance(row, dict) or row.get("recomputed") is not False
           or row.get("split") != "validation" for row in rows):
        raise PretrainedReportError("All imported rows must declare validation and recomputed=False.")
    _json(recommendations)
    _json(recorded)
    complete_config = dict(config)
    complete_config["recorded_validation"] = recorded
    payloads = {
        "recommendations": ("recommendations.csv", _csv(recommendations, (
            "user_id", "rank", "game_id", "title", "genres", "predicted_score",
            "price", "developer", "model_name", "is_fallback",
        ))),
        "inference_checks": ("inference_checks.json", _json(checks) + "\n"),
        "original_validation": ("original_validation.csv", _csv(rows)),
        "run_config": ("run_config.json", _json(complete_config) + "\n"),
    }
    try:
        for _, content in payloads.values():
            content.encode("utf-8")
    except UnicodeError as exc:
        raise PretrainedReportError("Report values must be valid UTF-8 text.") from exc
    root = Path(output_root).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = root / f"{run_name}_{stamp}_{uuid.uuid4().hex[:12]}"
    created = []
    folder_created = False
    try:
        root.mkdir(parents=True, exist_ok=True)
        folder.mkdir()
        folder_created = True
        paths = {}
        for key, (filename, content) in payloads.items():
            path = folder / filename
            with path.open("x", encoding="utf-8", newline="") as handle:
                created.append(path)
                handle.write(content)
            paths[key] = path
        return paths
    except OSError:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                pass
        if folder_created:
            try:
                folder.rmdir()
            except OSError:
                pass
        raise
