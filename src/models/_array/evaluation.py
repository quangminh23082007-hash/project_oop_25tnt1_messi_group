"""Exact full-TRAIN-catalog ranking; binary relevance; macro-average by user."""
from __future__ import annotations
import numpy as np
from .models import pair_sets, positive_int


def top_k(scores, k: int, excluded=()):
    positive_int(k, "k")
    values = np.asarray(scores, dtype=float).copy()
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite one-dimensional array")
    if len(excluded):
        indices = np.array(sorted(excluded), dtype=np.int64)
        if (indices < 0).any() or (indices >= len(values)).any():
            raise ValueError("Excluded item index out of range")
        values[indices] = -np.inf
    candidates = np.flatnonzero(np.isfinite(values))
    # Deterministic tie break by item_idx, not randomized ordering.
    order = np.lexsort((candidates, -values[candidates]))
    return candidates[order[:k]]


def ranking_metrics(ranked, relevant, k):
    positive_int(k, "k")
    relevant = set(relevant)
    if not relevant:
        raise ValueError("A scored user must have at least one relevant item")
    hits = np.array([int(i) in relevant for i in ranked[:k]], dtype=float)
    recall = float(hits.sum() / len(relevant))
    discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
    ideal = float((1.0 / np.log2(np.arange(2, min(k, len(relevant)) + 2))).sum())
    return recall, float((hits * discounts).sum() / ideal)


def evaluate(model, targets, seen, k=10, max_users=0, seed=42):
    positive_int(k, "k")
    if not isinstance(max_users, int) or max_users < 0:
        raise ValueError("max_users must be a nonnegative integer; 0 means all")
    targets = np.asarray(targets)
    seen = np.asarray(seen)
    if targets.size == 0:
        raise ValueError("No eligible positive holdout events; inspect report.json")
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("Targets must be a (N, 2) user/item array")
    if seen.size == 0:
        seen = np.empty((0, 2), dtype=np.int64)
    elif seen.ndim != 2 or seen.shape[1] != 2:
        raise ValueError("Seen history must be a (N, 2) user/item array")
    # Match pair_sets' integer user IDs, but choose the fixed user sample before
    # allocating millions of Python history sets on a full dataset.
    target_users = targets[:, 0].astype(np.int64, copy=False)
    users = np.unique(target_users)
    total_users = len(users)
    if max_users and total_users > max_users:
        users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))
        targets = targets[np.isin(target_users, users)]
    seen_users = seen[:, 0].astype(np.int64, copy=False)
    seen = seen[np.isin(seen_users, users)]
    truth, exclusions = pair_sets(targets), pair_sets(seen)
    values = []
    for u in users:
        if truth[u] & exclusions.get(u, set()):
            raise ValueError("A held-out target is also excluded; check split/deduplication")
        ranked = top_k(model.score_all(u), k, exclusions.get(u, set()))
        values.append(ranking_metrics(ranked, truth[u], k))
    recall, ndcg = np.mean(values, axis=0)
    return {f"recall@{k}": float(recall), f"ndcg@{k}": float(ndcg),
            "evaluated_users": len(users), "eligible_users": total_users,
            "candidate_catalog_size": model.n_items,
            "candidate_policy": "all train items minus observed history"}
