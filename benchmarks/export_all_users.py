from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from numbers import Real
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("user_index", "user_id", "rank", "game_id", "predicted_score")
CATALOG_COLUMNS = ("game_id", "title", "genres", "price", "developer", "image_url", "categories")


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _json_write(path, value):
    temporary = path.with_name(path.name + ".partial")
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def _lock(path):
    handle = path.open("a+b")
    locked = False
    try:
        if not path.stat().st_size:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _gzip_write(path, columns, emit):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=3) as packed:
            with io.TextIOWrapper(packed, encoding="utf-8", newline="") as text:
                writer = csv.writer(text)
                writer.writerow(columns)
                statistics = emit(writer)
        raw.flush()
        os.fsync(raw.fileno())
    digest = _sha(temporary)
    if path.exists():
        if _sha(path) != digest:
            raise ValueError(f"Uncommitted file differs from recomputed output: {path}")
        temporary.unlink()
    else:
        os.replace(temporary, path)
    return {
        "file": path.name,
        "sha256": digest,
        "bytes": path.stat().st_size,
        **statistics,
    }


def _verify_file(folder, record):
    name = record.get("file")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Invalid file name in manifest.")
    path = folder / name
    if not path.is_file() or path.stat().st_size != record["bytes"] or _sha(path) != record["sha256"]:
        raise ValueError(f"Missing or changed committed output: {path}")


def _summary(state):
    parts = state["parts"]
    signature = state["signature"]
    processed = state["next_user_index"]
    return {
        "status": state["status"],
        "model": signature["model_type"],
        "top_k": signature["top_k"],
        "mapped_users": signature["mapped_users"],
        "mapped_games": signature["mapped_games"],
        "requested_users": signature["end_user_index"],
        "processed_users": processed,
        "users_with_recommendations": sum(p["users_with_recommendations"] for p in parts),
        "users_without_recommendations": sum(p["users_without_recommendations"] for p in parts),
        "recommendation_rows": sum(p["rows"] for p in parts),
        "part_count": len(parts),
        "all_mapped_users_exported": (
            state["status"] == "complete"
            and processed == signature["mapped_users"]
        ),
        "exclude_played": True,
        "recomputed_metrics": False,
        "elapsed_seconds": state["elapsed_seconds"],
    }


def _validate_state(folder, state, signature):
    if state.get("signature") != signature:
        raise ValueError("Bundle, range, K, chunk size or runtime changed. Use the original settings or a new output folder.")
    if state.get("status") not in ("running", "complete") or not isinstance(state.get("parts"), list):
        raise ValueError("Invalid checkpoint state.")
    _verify_file(folder, state["catalog"])
    expected = 0
    for part in state["parts"]:
        stop = min(expected + signature["users_per_part"], signature["end_user_index"])
        name = f"part_{expected:09d}_{stop:09d}.csv.gz"
        if (
            expected >= stop or part.get("start") != expected
            or part.get("stop") != stop or part.get("file") != name
        ):
            raise ValueError("Noncontiguous parts in checkpoint.")
        users = stop - expected
        for key in ("rows", "users_with_recommendations", "users_without_recommendations"):
            if type(part.get(key)) is not int or part[key] < 0:
                raise ValueError("Invalid part statistics.")
        if (
            part["users_with_recommendations"] + part["users_without_recommendations"] != users
            or part["rows"] > users * signature["top_k"]
        ):
            raise ValueError("Inconsistent part statistics.")
        _verify_file(folder, part)
        expected = stop
    if state.get("next_user_index") != expected:
        raise ValueError("Checkpoint does not match committed parts.")
    if state["status"] == "complete" and expected != signature["end_user_index"]:
        raise ValueError("Incomplete export is marked complete.")


def export_all_users(
    model_dir, output_dir, *, top_k=10, users_per_part=10000, limit_users=0
):
    for name, value, minimum in (
        ("top_k", top_k, 1), ("users_per_part", users_per_part, 1), ("limit_users", limit_users, 0)
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    model_dir, output = _resolve(model_dir), _resolve(output_dir)
    info_path = model_dir / "model_info.json"
    raw = info_path.read_bytes()
    info = json.loads(raw.decode("utf-8-sig"))
    info_sha = hashlib.sha256(raw).hexdigest()
    del raw
    if info.get("format_version") != 3 or info.get("representation") != "compact":
        raise ValueError("Expected a format-3 compact CPU bundle.")
    users, items = info["user_ids"], info["item_ids"]
    if (
        not isinstance(users, list) or not users or not isinstance(items, list) or not items
        or any(not isinstance(v, str) or not v.strip() for v in users)
        or any(not isinstance(v, str) or not v.strip() for v in items)
    ):
        raise ValueError("Bundle mappings must contain nonempty raw string IDs.")
    if top_k > len(items):
        raise ValueError("top_k exceeds the mapped catalog.")
    end = min(len(users), limit_users) if limit_users else len(users)
    import numpy as np
    signature = {
        "schema_version": 1,
        "model_info_sha256": info_sha,
        "weights_sha256": info["weights_sha256"],
        "data_fingerprint": info["data_fingerprint"],
        "model_type": info["model_type"],
        "mapped_users": len(users),
        "mapped_games": len(items),
        "end_user_index": end,
        "top_k": top_k,
        "users_per_part": users_per_part,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas_threads": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    del info
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    with _lock(output / "export.lock"):
        if manifest.exists():
            state = json.loads(manifest.read_text(encoding="utf-8"))
            _validate_state(output, state, signature)
            if state["status"] == "complete":
                _json_write(output / "summary.json", _summary(state))
                print(f"Already complete: {manifest}", flush=True)
                return manifest
        else:
            allowed = {"export.lock", "catalog.csv.gz", "catalog.csv.gz.partial", "manifest.json.partial"}
            if any(path.name not in allowed for path in output.iterdir()):
                raise ValueError("Output folder is not a new export folder.")
            state = None
        from src.models.pretrained import PretrainedRecommender
        started = perf_counter()
        model = PretrainedRecommender.load(model_dir)
        if _sha(info_path) != info_sha or model.provenance["data_fingerprint"] != signature["data_fingerprint"]:
            raise ValueError("Bundle metadata changed during loading.")
        if state is None:
            def catalog_rows(writer):
                for game_id in items:
                    game = model.get_game_details(game_id)
                    details = model.get_catalog_metadata(game_id)
                    writer.writerow((
                        game_id, game.title, json.dumps(game.genres, ensure_ascii=False),
                        game.price, game.developer, details.get("image_url"),
                        json.dumps(details.get("categories"), ensure_ascii=False),
                    ))
                return {"games": len(items)}
            catalog = _gzip_write(output / "catalog.csv.gz", CATALOG_COLUMNS, catalog_rows)
            state = {
                "signature": signature,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "running",
                "next_user_index": 0,
                "elapsed_seconds": 0.0,
                "catalog": catalog,
                "parts": [],
            }
            _json_write(manifest, state)
        elapsed_before = state["elapsed_seconds"]
        item_order = {game_id: index for index, game_id in enumerate(items)}
        while state["next_user_index"] < end:
            start = state["next_user_index"]
            stop = min(start + users_per_part, end)
            def emit(writer):
                count, with_results = 0, 0
                for index in range(start, stop):
                    user_id = users[index]
                    rows = model.recommend_games(user_id, top_k=top_k, exclude_played=True)
                    if len(rows) > top_k:
                        raise ValueError("Model returned more than K recommendations.")
                    seen, previous = set(), None
                    for rank, row in enumerate(rows, 1):
                        if (
                            type(row.rank) is not int or row.rank != rank
                            or row.game_id not in item_order or row.game_id in seen
                            or type(row.is_fallback) is not bool or row.is_fallback
                            or row.model_name != model.model_name
                            or isinstance(row.predicted_score, bool)
                            or not isinstance(row.predicted_score, Real)
                            or not math.isfinite(float(row.predicted_score))
                        ):
                            raise ValueError(f"Invalid output for source user index {index}.")
                        key = (-float(row.predicted_score), item_order[row.game_id])
                        if previous is not None and key < previous:
                            raise ValueError("Scores or tie-breaking order are invalid.")
                        previous = key
                        seen.add(row.game_id)
                        writer.writerow((index, user_id, rank, row.game_id, row.predicted_score))
                    count += len(rows)
                    with_results += bool(rows)
                return {
                    "rows": count,
                    "users_with_recommendations": with_results,
                    "users_without_recommendations": stop - start - with_results,
                }
            part = _gzip_write(
                output / f"part_{start:09d}_{stop:09d}.csv.gz", COLUMNS, emit
            )
            state["parts"].append({"start": start, "stop": stop, **part})
            state["next_user_index"] = stop
            state["elapsed_seconds"] = elapsed_before + perf_counter() - started
            if stop == end:
                state["status"] = "complete"
            _json_write(manifest, state)
            print(
                f"Users {stop:,}/{end:,}; rows {_summary(state)['recommendation_rows']:,}; "
                f"elapsed {state['elapsed_seconds'] / 60:.1f} min",
                flush=True,
            )
        summary = _summary(state)
        _json_write(output / "summary.json", summary)
        print(f"Summary: {output / 'summary.json'}", flush=True)
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export top-K for every mapped source user without training.")
    parser.add_argument("--model-dir", default="artifacts/models/bpr/cpu")
    parser.add_argument("--output-dir", default="results/bpr_all_users_top10")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--users-per-part", type=int, default=10000)
    parser.add_argument("--limit-users", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        export_all_users(
            args.model_dir, args.output_dir,
            top_k=args.top_k, users_per_part=args.users_per_part, limit_users=args.limit_users,
        )
    except KeyboardInterrupt:
        print("Stopped. Run the same command to resume from committed parts.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Export failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
