"""Stream Steam interactions into fixed, versioned recommender inputs.

IDs are never inferred from display names. Every input requires an explicit
identity-provenance choice; verification is a caller assertion, not an inference.
Random pair splits are exploratory and make no future-prediction claim.
"""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import gzip
import hashlib
import html
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
import sqlite3
import tempfile

import numpy as np

from .models import positive_int, validate_pairs


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(mapping, arrays):
    digest = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode())
    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name], dtype="<i8")
        digest.update(name.encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


@dataclass
class PreparedData:
    user_ids: list[str]
    item_ids: list[str]
    arrays: dict[str, np.ndarray]
    catalog: dict
    report: dict

    @property
    def n_users(self):
        return len(self.user_ids)

    @property
    def n_items(self):
        return len(self.item_ids)

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        mapping = {"user_ids": self.user_ids, "item_ids": self.item_ids}
        self.report["fingerprint"] = fingerprint(mapping, self.arrays)
        write_json(folder / "mappings.json", mapping)
        write_json(folder / "catalog.json", self.catalog)
        write_json(folder / "report.json", self.report)
        np.savez_compressed(folder / "dataset.npz", **self.arrays)

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        mapping = json.loads((folder / "mappings.json").read_text(encoding="utf-8"))
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        with np.load(folder / "dataset.npz", allow_pickle=False) as handle:
            arrays = {key: handle[key].copy() for key in handle.files}
        if report.get("fingerprint") != fingerprint(mapping, arrays):
            raise ValueError("Prepared data or ID mapping changed; prepare a new dataset")
        result = cls(mapping["user_ids"], mapping["item_ids"], arrays,
                     json.loads((folder / "catalog.json").read_text(encoding="utf-8")), report)
        if len(set(result.user_ids)) != result.n_users or len(set(result.item_ids)) != result.n_items:
            raise ValueError("Duplicate IDs in mappings")
        for name in ("train", "train_seen", "validation", "validation_seen", "test"):
            if name not in arrays:
                raise ValueError(f"Prepared data is missing {name}")
            validate_pairs(arrays[name], result.n_users, result.n_items,
                           allow_empty=name not in ("train", "train_seen"))
        return result


def table_rows(path):
    """Read one CSV, JSONL(.gz), or Parquet file without loading it all."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Expected one input file: {path}")
    name = path.name.lower()
    if name.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Install pyarrow to read Parquet files") from exc
        parquet = pq.ParquetFile(path)
        # Bound Python object memory even when the source has millions of rows.
        for batch in parquet.iter_batches(batch_size=65536):
            yield from batch.to_pylist()
        return
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        if name.endswith((".csv", ".csv.gz")):
            csv.field_size_limit(16 * 1024 * 1024)
            reader = csv.DictReader(handle)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(f"Missing or duplicate CSV headers: {path}")
            for number, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed CSV record near line {number}: {path}")
                yield row
        elif name.endswith((".jsonl", ".jsonl.gz", ".ndjson")):
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {number}: {path}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object on line {number}: {path}")
                yield row
        else:
            raise ValueError("Input must be .parquet, .csv(.gz), or .jsonl(.gz)")


def parse_recommended(value):
    """Parse labels explicitly; bool('False') would incorrectly be True."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Real) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "recommended"):
            return True
        if text in ("0", "false", "not recommended"):
            return False
    raise ValueError(f"Invalid recommended value: {value!r}; use true/false or Recommended/Not Recommended")


def _id(value):
    # Reject floats: large numeric account IDs may already have lost precision.
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value >= 0:
        return str(value)
    raise ValueError("IDs must be nonempty strings or exact nonnegative integers, never floats")


def _user(row, allow_unverified, verified_user_ids=False):
    if "steamid" in row:
        value = _id(row["steamid"])
        if not value.isascii() or not value.isdecimal() or not 0 < int(value) < 2**64:
            raise ValueError("steamid must be an exact positive unsigned 64-bit account ID")
        if "user_id" in row and _id(row["user_id"]) != value:
            raise ValueError("user_id and steamid disagree")
        return value, "asserted_steamid_column" if verified_user_ids else "unverified_user_id"
    if "user_id" not in row:
        raise ValueError("Interactions require user_id (or steamid), game_id and recommended")
    if not allow_unverified and not verified_user_ids:
        raise ValueError("user_id provenance is unverified; display names are not reliable identities. "
                         "Supply a steamid column from an account-ID source, or explicitly use "
                         "--allow-unverified-users for an exploratory experiment")
    return _id(row["user_id"]), "asserted_stable_user_id" if verified_user_ids else "unverified_user_id"


def _timestamp(value, *, required=False):
    if value is None or value == "":
        if required:
            raise ValueError("Chronological preparation requires timestamp_ms for every interaction/review")
        return None
    if isinstance(value, str) and value.strip().isascii() and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, Integral) or not 100_000_000_000 <= value < 10_000_000_000_000:
        raise ValueError("timestamp_ms must be an integer Unix timestamp in milliseconds; seconds are not accepted")
    return int(value)


def clean_review_text(value, max_chars=12000):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Review text must be a string or null")
    text = re.sub(r"<[^>]*>", " ", html.unescape(value))
    return " ".join("".join(c if c.isprintable() else " " for c in text).split())[:max_chars]


def _draw(seed, *parts):
    # JSON framing avoids collisions between IDs that contain separators.
    key = json.dumps([seed, *parts], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64


def _random_split(seed, user, item, train_fraction, validation_fraction):
    draw = _draw(seed, "pair", user, item)
    return "train" if draw < train_fraction else "validation" if draw < train_fraction + validation_fraction else "test"


def _pairs_from_query(db, query):
    count = db.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0]
    return np.fromiter((value for row in db.execute(query) for value in row), dtype=np.int64,
                       count=count * 2).reshape(-1, 2)


def _catalog(games, item_ids, stats):
    wanted = set(item_ids)
    catalog = {}
    if games is not None:
        for row in table_rows(games):
            stats["metadata_rows"] += 1
            if "game_id" not in row:
                raise ValueError("Game metadata requires game_id")
            item = _id(row["game_id"])
            if item not in wanted:
                continue
            if item in catalog:
                raise ValueError(f"Duplicate game_id in metadata: {item}")
            title = row.get("title", row.get("name", item))
            categories = row.get("genres", [])
            if isinstance(categories, str):
                try:
                    categories = json.loads(categories)
                except json.JSONDecodeError:
                    categories = [categories]
            if not isinstance(categories, (list, tuple)):
                categories = []
            image = row.get("image_url")
            catalog[item] = {"title": title.strip() if isinstance(title, str) and title.strip() else item,
                             "categories": [str(x) for x in categories],
                             "image_url": image if isinstance(image, str) else None}
    stats["train_items_missing_metadata"] = len(wanted) - len(catalog)
    for item in item_ids:
        catalog.setdefault(item, {"title": item, "categories": [], "image_url": None})
    return catalog


def prepare(interactions, out, *, games=None, reviews=None, split="random",
            allow_unverified_users=False, verified_user_ids=False, user_fraction=1.0, seed=42,
            min_user=3, min_item=2, train_fraction=0.8, validation_fraction=0.1):
    """Prepare canonical Steam interactions and optional separately keyed text.

    Duplicate interactions with matching labels and timestamps are collapsed;
    a missing text can be filled by an otherwise matching duplicate.
    Conflicting duplicate labels,
    timestamps or nonempty inline text fail instead of silently rewriting history.
    `validation_seen` contains TRAIN + validation observations for test exclusion.
    """
    positive_int(min_user, "min_user")
    positive_int(min_item, "min_item")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if split not in ("random", "chronological"):
        raise ValueError("split must be random or chronological")
    if not isinstance(allow_unverified_users, bool):
        raise ValueError("allow_unverified_users must be boolean")
    if not isinstance(verified_user_ids, bool):
        raise ValueError("verified_user_ids must be boolean")
    if allow_unverified_users == verified_user_ids:
        raise ValueError("Choose exactly one identity policy: --verified-user-ids asserts stable account IDs "
                         "from the source; --allow-unverified-users acknowledges display-name/unknown IDs "
                         "for an exploratory experiment. Hashing names does not verify identity.")
    if not math.isfinite(user_fraction) or not 0 < user_fraction <= 1:
        raise ValueError("user_fraction must be in (0, 1]")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("Train, validation and test fractions must all be > 0")
    destination = Path(out)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Output folder is not empty; choose a new folder")
    destination.mkdir(parents=True, exist_ok=True)
    stats, identity_sources = Counter(), Counter()
    train_end_ms = validation_end_ms = None
    with tempfile.TemporaryDirectory(prefix="prepare_", dir=destination) as temp:
        db = sqlite3.connect(str(Path(temp) / "dedup.sqlite"))
        try:
            db.execute("CREATE TABLE events (u TEXT, i TEXT, p INTEGER, t INTEGER, x TEXT, xt INTEGER, s TEXT, PRIMARY KEY(u,i)) WITHOUT ROWID")
            for row in table_rows(interactions):
                stats["input_interactions"] += 1
                u, identity = _user(row, allow_unverified_users, verified_user_ids)
                if "game_id" not in row or "recommended" not in row:
                    raise ValueError("Interactions require game_id and recommended")
                i = _id(row["game_id"])
                p = int(parse_recommended(row["recommended"]))
                t = _timestamp(row.get("timestamp_ms"), required=split == "chronological")
                identity_sources[identity] += 1
                if _draw(seed, "user", u) >= user_fraction:
                    stats["outside_user_sample"] += 1
                    continue
                x = clean_review_text(row.get("review", row.get("text")))
                phase = _random_split(seed, u, i, train_fraction, validation_fraction) if split == "random" else ""
                try:
                    db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)", (u, i, p, t, x, t if x else None, phase))
                except sqlite3.IntegrityError:
                    previous = db.execute("SELECT p,t,x FROM events WHERE u=? AND i=?", (u, i)).fetchone()
                    if previous[:2] != (p, t) or (previous[2] and x and previous[2] != x):
                        raise ValueError(f"Conflicting duplicate interaction for user {u!r}, game {i!r}; resolve duplicate snapshots before preparation")
                    if x and not previous[2]:
                        db.execute("UPDATE events SET x=?,xt=? WHERE u=? AND i=?", (x, t, u, i))
                    stats["duplicate_rows_removed"] += 1
                stats["accepted_before_dedup"] += 1
                if stats["accepted_before_dedup"] % 100000 == 0:
                    db.commit()
                    print(f"Prepared input: {stats['accepted_before_dedup']:,} sampled rows", flush=True)
            db.commit()
            total = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            stats["unique_user_item_pairs"] = total
            if total < 3:
                raise ValueError("Need at least three distinct sampled interactions")
            if split == "chronological":
                db.execute("CREATE INDEX event_time ON events(t)")
                train_end_ms = db.execute("SELECT t FROM events ORDER BY t LIMIT 1 OFFSET ?", (int(total * train_fraction),)).fetchone()[0]
                validation_end_ms = db.execute("SELECT t FROM events ORDER BY t LIMIT 1 OFFSET ?", (int(total * (train_fraction + validation_fraction)),)).fetchone()[0]
                if train_end_ms >= validation_end_ms:
                    raise ValueError("Chronological cutoffs coincide; use more distinct timestamps or different fractions")
                db.execute("UPDATE events SET s=CASE WHEN t<? THEN 'train' WHEN t<? THEN 'validation' ELSE 'test' END", (train_end_ms, validation_end_ms))
            db.execute("CREATE TABLE core (u TEXT,i TEXT,PRIMARY KEY(u,i)) WITHOUT ROWID")
            db.execute("INSERT INTO core SELECT u,i FROM events WHERE s='train' AND p=1")
            stats["train_positive_before_core"] = db.execute("SELECT COUNT(*) FROM core").fetchone()[0]
            db.execute("CREATE INDEX core_item ON core(i)")
            while True:
                deleted = db.execute("""DELETE FROM core WHERE
                    u IN (SELECT u FROM core GROUP BY u HAVING COUNT(*) < ?) OR
                    i IN (SELECT i FROM core GROUP BY i HAVING COUNT(*) < ?)""", (min_user, min_item)).rowcount
                if not deleted:
                    break
            users = [row[0] for row in db.execute("SELECT DISTINCT u FROM core ORDER BY u")]
            items = [row[0] for row in db.execute("SELECT DISTINCT i FROM core ORDER BY i")]
            if not users or len(items) < 2:
                raise ValueError("Too few train positives after core filtering; increase user_fraction or reduce min counts; BPR needs two items")
            db.execute("CREATE TABLE users (u TEXT PRIMARY KEY,idx INTEGER) WITHOUT ROWID")
            db.execute("CREATE TABLE items (i TEXT PRIMARY KEY,idx INTEGER) WITHOUT ROWID")
            db.executemany("INSERT INTO users VALUES (?,?)", ((u, idx) for idx, u in enumerate(users)))
            db.executemany("INSERT INTO items VALUES (?,?)", ((i, idx) for idx, i in enumerate(items)))
            mapped = "FROM events e JOIN users u ON e.u=u.u JOIN items i ON e.i=i.i"
            arrays = {}
            conditions = {"train": "e.s='train' AND e.p=1", "train_seen": "e.s='train'",
                          "validation": "e.s='validation' AND e.p=1",
                          "validation_seen": "e.s IN ('train','validation')", "test": "e.s='test' AND e.p=1"}
            for name, condition in conditions.items():
                arrays[name] = _pairs_from_query(db, f"SELECT u.idx,i.idx {mapped} WHERE {condition} ORDER BY u.idx,i.idx")
            split_stats = {}
            for phase in ("train", "validation", "test"):
                count, positives = db.execute("SELECT COUNT(*),COALESCE(SUM(p),0) FROM events WHERE s=?", (phase,)).fetchone()
                eligible = len(arrays[phase])
                split_stats[phase] = {"all_reviews": count, "positive_reviews": positives,
                                      "eligible_positive_reviews": eligible,
                                      "excluded_cold_positive_reviews": positives - eligible,
                                      "eligible_positive_fraction": eligible / positives if positives else 0.0,
                                      "eligible_users": int(len(np.unique(arrays[phase][:, 0])))}
            catalog = _catalog(games, items, stats)
            # Discard non-training text before joining any external review source.
            db.execute("UPDATE events SET x='' WHERE s!='train'")
            if reviews is not None:
                for row in table_rows(reviews):
                    stats["input_review_rows"] += 1
                    u, _ = _user(row, allow_unverified_users, verified_user_ids)
                    if "game_id" not in row:
                        raise ValueError("Review rows require game_id")
                    i = _id(row["game_id"])
                    x = clean_review_text(row.get("review", row.get("text")))
                    if not x:
                        stats["empty_review_rows"] += 1
                        continue
                    t = _timestamp(row.get("timestamp_ms"), required=split == "chronological")
                    existing = db.execute(f"SELECT e.x,e.xt {mapped} WHERE e.u=? AND e.i=? AND e.s='train'", (u, i)).fetchone()
                    if existing is None:
                        stats["reviews_not_mapped_training_pairs"] += 1
                        continue
                    if split == "chronological" and t >= train_end_ms:
                        stats["reviews_after_training_cutoff"] += 1
                        continue
                    if existing[0] and existing[0] != x:
                        raise ValueError(f"Conflicting review texts for training pair {u!r}, {i!r}; supply one as-of review snapshot")
                    if existing[0] == x:
                        stats["duplicate_review_rows"] += 1
                        # Matching snapshots use the earliest known availability
                        # time, independent of file ordering (null means unknown).
                        if t is not None and (existing[1] is None or t < existing[1]):
                            db.execute("UPDATE events SET xt=? WHERE u=? AND i=?", (t, u, i))
                        continue
                    db.execute("UPDATE events SET x=?,xt=? WHERE u=? AND i=?", (x, t, u, i))
            text_path = destination / "train_reviews.jsonl"
            stats["train_reviews_with_text"] = 0
            with text_path.open("w", encoding="utf-8") as handle:
                query = f"SELECT u.idx,i.idx,e.i,e.x,e.xt {mapped} WHERE e.s='train' AND e.x!='' ORDER BY e.xt,e.u,e.i"
                for u, i, original, text, timestamp in db.execute(query):
                    record = {"user_idx": u, "item_idx": i, "game_id": original, "text": text, "timestamp_ms": timestamp}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["train_reviews_with_text"] += 1
            stats["eligible_train_reviews_without_text"] = len(arrays["train_seen"]) - stats["train_reviews_with_text"]
        finally:
            db.close()
    limitations = []
    if identity_sources["unverified_user_id"]:
        limitations.append("User identities are unverified and may be display names; collisions can invalidate personalized metrics.")
    if verified_user_ids:
        limitations.append("Stable user-ID provenance was asserted by the caller; this preparation cannot independently prove account identity.")
    if split == "random":
        limitations.append("Deterministic random pair split is exploratory; it does not establish future chronological generalization.")
    if split == "chronological":
        limitations.append("Timestamps must describe the available text snapshot, not merely original creation time of later-edited reviews.")
    if not stats["train_reviews_with_text"]:
        limitations.append("No eligible review text; train ordinary BPR until training-only text is supplied.")
    for phase in ("validation", "test"):
        if not len(arrays[phase]):
            limitations.append(f"No warm positive {phase} targets survived; meaningful {phase} ranking metrics are unavailable.")
    report = {"format_version": 3, "n_users": len(users), "n_items": len(items),
              "counts": dict(stats), "splits": split_stats, "limitations": limitations,
              "identity_sources": dict(identity_sources),
              "config": {"split": split, "seed": seed, "min_user": min_user, "min_item": min_item,
                         "user_fraction": user_fraction, "allow_unverified_users": allow_unverified_users,
                         "verified_user_ids": verified_user_ids,
                         "train_fraction": train_fraction, "validation_fraction": validation_fraction,
                         "train_end_ms": train_end_ms, "validation_end_ms": validation_end_ms},
              "text": {"file": "train_reviews.jsonl", "sha256": file_sha256(destination / "train_reviews.jsonl"),
                       "policy": "only exact mapped TRAIN interaction pairs; includes both recommendation labels",
                       "max_characters_per_review": 12000, "no_holdout_reviews": True},
              "protocol": {"positive": "recommended is explicitly true; no aggregate rating weighting",
                           "dedup": "matching interaction label/timestamp snapshots collapse, filling empty text; conflicting snapshots fail",
                           "review_dedup": "matching text uses earliest known availability timestamp; conflicting text fails",
                           "mapping_and_core": "train positives only",
                           "negative_pool": "train catalog excluding all observed training pairs",
                           "validation_seen": "union of train and validation observations; use for test exclusions",
                           "metadata": "display only; no aggregate ratings/counts as model inputs",
                           "scaling": "raw input and dedup stream through SQLite; mapped arrays/users/items still require RAM"}}
    result = PreparedData(users, items, arrays, catalog, report)
    result.save(destination)
    return result
