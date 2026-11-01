from collections.abc import Iterable, Sequence
from importlib import import_module
from math import isfinite
from numbers import Integral, Real
from typing import Dict, List

from src.evaluation.protocol import Interaction, validate_interactions


def _load_cornac():
    try:
        return import_module("cornac")
    except ImportError as error:
        raise ImportError(
            "Cornac is required for this benchmark. Install it in the project environment "
            "with: python -m pip install cornac"
        ) from error


class CornacMostPopAdapter:
    def __init__(
        self,
        name: str = "CornacMostPop",
        positive_threshold: float = 1.0,
        seed: int = 42,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string.")
        if isinstance(positive_threshold, bool) or not isinstance(positive_threshold, Real):
            raise TypeError("positive_threshold must be a finite real number.")
        if not isfinite(float(positive_threshold)):
            raise ValueError("positive_threshold must be finite.")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer.")
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be between 0 and 2**32 - 1.")

        self.name = name.strip()
        self.positive_threshold = float(positive_threshold)
        self.seed = int(seed)
        self.is_fitted = False
        self.num_users = 0
        self.num_items = 0
        self._model = None
        self._user_map: Dict[str, int] = {}
        self._item_map: Dict[str, int] = {}

    def fit(
        self,
        train_interactions: Iterable[Interaction],
        num_users: int,
        num_items: int,
    ) -> "CornacMostPopAdapter":
        interactions = validate_interactions(
            train_interactions,
            num_users,
            num_items,
            label="train_interactions",
        )
        if not interactions:
            raise ValueError("Cannot fit CornacMostPop on an empty training set.")

        positives = sorted(
            (record for record in interactions if record.rating >= self.positive_threshold),
            key=lambda record: (record.user_id, record.item_id),
        )
        if not positives:
            raise ValueError("CornacMostPop requires at least one positive training interaction.")

        cornac = _load_cornac()
        rows = [(str(record.user_id), str(record.item_id), 1.0) for record in positives]
        dataset = cornac.data.Dataset.from_uir(rows, seed=self.seed)
        model = cornac.models.MostPop(name=self.name)
        model.fit(dataset)

        self._model = model
        self._user_map = dict(dataset.uid_map)
        self._item_map = dict(dataset.iid_map)
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.is_fitted = True
        return self

    def score_items(self, user_id: int, item_ids: Sequence[int]) -> List[float]:
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Call fit before score_items.")
        if isinstance(user_id, bool) or not isinstance(user_id, Integral):
            raise TypeError("user_id must be an integer.")
        if not 0 <= user_id < self.num_users:
            raise ValueError("user_id is outside the configured user range.")
        if isinstance(item_ids, (str, bytes)) or not isinstance(item_ids, Sequence):
            raise TypeError("item_ids must be an ordered sequence of integer IDs.")

        candidates = []
        for item_id in item_ids:
            if isinstance(item_id, bool) or not isinstance(item_id, Integral):
                raise TypeError("item_ids contains a non-integer ID.")
            if not 0 <= item_id < self.num_items:
                raise ValueError("item_ids contains an ID outside the configured item range.")
            candidates.append(int(item_id))
        if not candidates:
            return []

        cornac_user = self._user_map.get(str(int(user_id)), 0)
        known_scores = self._model.score(cornac_user)
        if len(known_scores) != len(self._item_map):
            raise RuntimeError("Cornac returned scores inconsistent with its item mapping.")

        scores = []
        for item_id in candidates:
            cornac_item = self._item_map.get(str(item_id))
            score = 0.0 if cornac_item is None else known_scores[cornac_item]
            if isinstance(score, bool) or not isinstance(score, Real):
                raise TypeError("Cornac returned a non-numeric score.")
            if not isfinite(float(score)):
                raise ValueError("Cornac returned a non-finite score.")
            scores.append(float(score))
        return scores
