"""NumPy-only model export and serving. The text encoder is an offline dependency."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import BaseRecommender
from .data import file_sha256, write_json
from .evaluation import top_k
from .models import (BPRMatrixFactorization, TextBPRMatrixFactorization,
                     PopularityRecommender, pair_sets, positive_int, validate_pairs)


class CompactRecommender(BaseRecommender):
    """Inference-only P and effective item factors R = Q + X @ W.T."""

    def fit(self, *args, **kwargs):
        raise RuntimeError("The compact export is inference-only; use the training pipeline to refit")

    def score_all(self, user_idx):
        self._check_user(user_idx)
        return self.R @ self.P[user_idx]


def save_bundle(folder, model, popularity, data, *, compact=False):
    model._check_fitted()
    popularity._check_fitted()
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Model output must be a new/empty folder")
    folder.mkdir(parents=True, exist_ok=True)
    text_model = isinstance(model, TextBPRMatrixFactorization)
    weights = {"P": model.P, "popularity": popularity.counts,
               "train_seen": data.arrays["train_seen"]}
    if compact:
        weights["R"] = model.Q + model.X @ model.W.T if text_model else model.Q
    else:
        weights["Q"] = model.Q
        if text_model:
            weights.update(W=model.W, X=model.X)
    np.savez_compressed(folder / "weights.npz", **weights)
    write_json(folder / "model_info.json", {
        "format_version": 3,
        "representation": "compact" if compact else "full",
        "model_type": type(model).__name__, "config": model.config(),
        "user_ids": data.user_ids, "item_ids": data.item_ids, "catalog": data.catalog,
        "data_fingerprint": data.report["fingerprint"],
        "data_config": data.report.get("config", {}),
        "data_limitations": data.report.get("limitations", []),
        "best_epoch": model.best_epoch, "history": model.history,
        "feature_info": getattr(model, "feature_info", None),
        "training_info": getattr(model, "training_info", None),
        "weights_sha256": file_sha256(folder / "weights.npz"),
    })


def load_bundle(folder):
    folder = Path(folder)
    info = json.loads((folder / "model_info.json").read_text(encoding="utf-8"))
    types = {"BPRMatrixFactorization": BPRMatrixFactorization,
             "TextBPRMatrixFactorization": TextBPRMatrixFactorization}
    if info.get("format_version") != 3 or info.get("model_type") not in types:
        raise ValueError("Unsupported bundle: use the new training/export commands")
    if info.get("weights_sha256") != file_sha256(folder / "weights.npz"):
        raise ValueError("Saved weights changed or the download is incomplete")
    representation = info.get("representation")
    if representation not in ("full", "compact"):
        raise ValueError("Unsupported model representation")
    model = (CompactRecommender() if representation == "compact"
             else types[info["model_type"]](**info["config"]))
    model.n_users, model.n_items = len(info["user_ids"]), len(info["item_ids"])
    positive_int(model.n_users, "n_users")
    positive_int(model.n_items, "n_items")
    if len(set(info["user_ids"])) != model.n_users or len(set(info["item_ids"])) != model.n_items:
        raise ValueError("Duplicate original IDs in bundle")
    factors = positive_int(info["config"]["factors"], "factors")
    popularity = PopularityRecommender()
    popularity.n_users, popularity.n_items = model.n_users, model.n_items
    with np.load(folder / "weights.npz", allow_pickle=False) as weights:
        required = ["P", "R"] if representation == "compact" else ["P", "Q"]
        if representation == "full" and isinstance(model, TextBPRMatrixFactorization):
            required += ["W", "X"]
        for name in required:
            value = weights[name].copy()
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite {name} in bundle")
            setattr(model, name, value)
        popularity.counts = weights["popularity"].copy()
        seen = validate_pairs(weights["train_seen"], model.n_users, model.n_items)
    item_factors = model.R if representation == "compact" else model.Q
    if model.P.shape != (model.n_users, factors) or item_factors.shape != (model.n_items, factors):
        raise ValueError("Weights do not match ID mappings/factor count")
    if isinstance(model, TextBPRMatrixFactorization):
        if model.X.ndim != 2 or model.X.shape[0] != model.n_items or model.X.shape[1] == 0:
            raise ValueError("Invalid item feature shape")
        if model.W.shape != (factors, model.X.shape[1]):
            raise ValueError("Invalid text projection shape")
    if (popularity.counts.shape != (model.n_items,) or
            not np.isfinite(popularity.counts).all() or (popularity.counts < 0).any()):
        raise ValueError("Invalid popularity counts")
    model._is_fitted = popularity._is_fitted = True
    model.best_epoch, model.history = info["best_epoch"], info["history"]
    model.feature_info = info.get("feature_info")
    model.training_info = info.get("training_info")
    model.source = "text_bpr" if info["model_type"] == "TextBPRMatrixFactorization" else "bpr"
    return model, popularity, info, seen


class RecommendationService:
    def __init__(self, model, popularity, user_ids, item_ids, seen, catalog):
        self.model, self.popularity = model, popularity
        self.user_map = {u: n for n, u in enumerate(user_ids)}
        self.item_ids = item_ids
        self.item_map = {i: n for n, i in enumerate(item_ids)}
        self.seen, self.catalog = pair_sets(seen), catalog

    @classmethod
    def load(cls, folder):
        model, popularity, info, seen = load_bundle(folder)
        return cls(model, popularity, info["user_ids"], info["item_ids"], seen, info["catalog"])

    def recommend(self, user_id, k=10, extra_seen=(), *, include_seen=False):
        """extra_seen must contain ALL current reviewed/owned game IDs from the app."""
        positive_int(k, "k")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a nonempty original ID string")
        user = self.user_map.get(user_id)
        if user is None:
            scores, source = self.popularity.counts, "popularity_fallback"
        else:
            scores = self.model.score_all(user)
            source = getattr(self.model, "source", "text_bpr" if isinstance(
                self.model, TextBPRMatrixFactorization) else "bpr")
        excluded = set() if include_seen else set(self.seen.get(user, ()))
        excluded.update(self.item_map[i] for i in extra_seen if i in self.item_map)
        ranked = top_k(scores, k, excluded)
        return {"user_id": user_id, "source": source, "items": [
            {"game_id": self.item_ids[int(i)],
             "title": self.catalog.get(self.item_ids[int(i)], {}).get("title", self.item_ids[int(i)]),
             "score": float(scores[i])} for i in ranked]}
