from typing import Any, Dict, FrozenSet, Iterable, List, Optional
import numpy as np

from src.contracts.dto import GameItem, Interaction
from src.contracts.interfaces import BaseEvaluator, BaseRecommender
from src.contracts.dto import EvaluationResult


class StubIdMapper:
    def __init__(self, raw_ids: List[str]):
        self._raw_ids = list(raw_ids)
        self.size = len(raw_ids)
        self._raw_to_idx = {r: i for i, r in enumerate(raw_ids)}

    def contains_raw(self, raw_id: str) -> bool:
        return raw_id in self._raw_to_idx

    def to_idx(self, raw_id: str) -> int:
        return self._raw_to_idx[raw_id]

    def to_raw(self, index: int) -> str:
        return self._raw_ids[index]


class StubDataset:
    def __init__(
        self,
        split_name: str = "train",
        interactions: Optional[List[Interaction]] = None,
        games: Optional[Dict[str, GameItem]] = None,
        num_users: int = 3,
        num_items: int = 5,
        user_ids: Optional[List[str]] = None,
        item_ids: Optional[List[str]] = None,
        mapping_fingerprint: str = "mfp-test-01",
        fingerprint: str = "fp-test-01",
        review_fingerprint: str = "rfp-test-01",
    ):
        self.split_name = split_name
        self.num_users = num_users
        self.num_items = num_items
        self.user_mapper = StubIdMapper(user_ids or [f"u{i}" for i in range(num_users)])
        self.item_mapper = StubIdMapper(item_ids or [f"g{i}" for i in range(num_items)])
        self.mapping_fingerprint = mapping_fingerprint
        self.fingerprint = fingerprint
        self.review_fingerprint = review_fingerprint

        if interactions is None:
            self.interactions = [
                Interaction(user_id=0, item_id=0, rating=1.0, timestamp=100),
                Interaction(user_id=0, item_id=1, rating=0.0, timestamp=101),
                Interaction(user_id=1, item_id=2, rating=1.0, timestamp=102),
                Interaction(user_id=2, item_id=3, rating=0.0, timestamp=103),
            ]
        else:
            self.interactions = interactions

        self.num_interactions = len(self.interactions)

        self._games = games or {
            f"g{i}": GameItem(
                item_id=f"g{i}",
                title=f"Game Title {i}",
                genres=("Action", "RPG") if i % 2 == 0 else ("Strategy",),
                price=float(10.0 * (i + 1)) if i != 4 else None,
                developer=f"DevStudio_{i}",
            )
            for i in range(self.num_items)
        }

    def get_game_by_raw_id(self, raw_game_id: str) -> GameItem:
        if raw_game_id not in self._games:
            raise KeyError(raw_game_id)
        return self._games[raw_game_id]

    def get_user_seen_items(self, user_id: int) -> FrozenSet[int]:
        return frozenset(rec.item_id for rec in self.interactions if rec.user_id == user_id)

    def get_user_positive_items(self, user_id: int) -> FrozenSet[int]:
        return frozenset(
            rec.item_id for rec in self.interactions if rec.user_id == user_id and rec.rating == 1.0
        )

    def iter_reviews(self) -> Iterable[Any]:
        return iter([])


class StubModel(BaseRecommender):
    def __init__(self, name: str = "StubModel"):
        super().__init__(name)

    def fit(self, train_dataset: Any, *, validation_score=None, item_features=None, verbose=False):
        self._num_users = train_dataset.num_users
        self._num_items = train_dataset.num_items
        self._mapping_fingerprint = train_dataset.mapping_fingerprint
        self._train_fingerprint = train_dataset.fingerprint
        self._is_fitted = True
        return self

    def score_items(self, user_id: int, item_ids: Any) -> np.ndarray:
        ids = np.asarray(item_ids, dtype=np.int64)
        return (10.0 - ids).astype(float)

    def config(self) -> dict:
        return {"n_factors": 16}


class StubBPRModel(StubModel):
    pass


class StubPopularityRecommender(BaseRecommender):
    def __init__(self, name: str = "Popularity"):
        super().__init__(name)
        self._counts = np.array([4.0, 3.0, 2.0, 1.0, 0.5])

    def fit(self, train_dataset: Any, *, validation_score=None, item_features=None, verbose=False):
        self._num_users = train_dataset.num_users
        self._num_items = train_dataset.num_items
        self._mapping_fingerprint = train_dataset.mapping_fingerprint
        self._train_fingerprint = train_dataset.fingerprint
        self._is_fitted = True
        return self

    def score_global(self, item_ids: Any) -> np.ndarray:
        ids = np.asarray(item_ids, dtype=np.int64)
        return self._counts[ids]

    def score_items(self, user_id: int, item_ids: Any) -> np.ndarray:
        return self.score_global(item_ids)

    def config(self) -> dict:
        return {}


class StubEvaluator(BaseEvaluator):
    def evaluate(
        self,
        model: BaseRecommender,
        train_dataset: Any,
        evaluation_dataset: Any,
        *,
        additional_seen_dataset: Any = None,
    ) -> EvaluationResult:
        return EvaluationResult(
            model_name=model.name,
            split_name=evaluation_dataset.split_name,
            metrics={"NDCG@10": 0.85, "Recall@10": 0.75},
        )


class BrokenModel(StubModel):
    def fit(self, *a, **kw):
        raise RuntimeError("Training divergence")


class NegativeScoreModel(StubModel):
    def score_items(self, user_id: int, item_ids: Any) -> np.ndarray:
        ids = np.asarray(item_ids, dtype=np.int64)
        return (-50.0 + ids).astype(float)
