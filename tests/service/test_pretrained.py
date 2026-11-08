import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import patch
import numpy as np

from src.contracts.dto import GameItem, RecommendationResult
from src.contracts.exceptions import (
    ColdStartError,
    ConfigurationError,
    InvalidParameterError,
    ModelNotFittedError,
    UnknownItemError,
    UnknownUserError,
)
from src.service.pretrained import PretrainedDatasetProxy, PretrainedServiceFacade
from src.service.recommendation_service import RecommendationService
from tests.service.conftest import (
    StubIdMapper,
    StubDataset,
    StubModel,
    StubPopularityRecommender,
)


class StubRealPretrainedRecommender:
    """Mock simulating src.models.pretrained.PretrainedRecommender (Contracts v1.1)."""

    def __init__(self, allow_cold_start_fallback: bool = False, catalog_lookup: Any = None):
        self._users = {"u0": 0, "u1": 1, "u2": 2}
        self._items = ("g0", "g1", "g2", "g3")
        self._item_map = {item: i for i, item in enumerate(self._items)}
        self._catalog = {
            "g0": {"title": "Game 0", "image_url": "example0.png", "categories": ["Action"]},
            "g1": {"title": "Game 1", "image_url": "example1.png", "categories": ["RPG"]},
            "g2": {"title": "Game 2", "image_url": "example2.png", "categories": ["Strategy"]},
            "g3": {"title": "Game 3", "image_url": "example3.png", "categories": ["Action"]},
        }
        self.model_name = "BPRMatrixFactorization"
        self.allow_cold_start_fallback = allow_cold_start_fallback
        self._catalog_lookup = catalog_lookup
        self.provenance = {
            "protocol": "legacy-kaggle-format-3",
            "data_fingerprint": "original-training",
            "data_config": {"train_fraction": 0.7},
        }
        self._seen = np.array([[0, 0], [0, 1], [1, 2]])
        self._popularity = type("Popularity", (), {"counts": np.array([3.0, 2.0, 1.0, 0.5])})()
        self._model = type(
            "InnerModel",
            (),
            {
                "source": "BPRMatrixFactorization",
                "score_all": lambda s, u: np.array([5.0, 4.0, 20.0, 15.0]) if u == 0 else np.array([1.0, 2.0, 3.0, 4.0]),
            },
        )()

    @classmethod
    def load(cls, folder, *, allow_cold_start_fallback=False, catalog_lookup=None):
        return cls(allow_cold_start_fallback=allow_cold_start_fallback, catalog_lookup=catalog_lookup)

    def _history(self, user):
        lo = np.searchsorted(self._seen[:, 0], user, side="left")
        hi = np.searchsorted(self._seen[:, 0], user, side="right")
        return self._seen[lo:hi, 1]

    def get_game_details(self, raw_game_id):
        if raw_game_id not in self._item_map:
            raise UnknownItemError(f"Game not present in trained mapping: {raw_game_id}")
        if self._catalog_lookup is not None:
            return self._catalog_lookup(raw_game_id)
        row = self._catalog.get(raw_game_id, {})
        return GameItem(
            item_id=raw_game_id,
            title=row.get("title", raw_game_id),
            genres=tuple(row.get("categories", ())),
            price=19.99 if raw_game_id == "g2" else None,
        )

    def get_catalog_metadata(self, raw_game_id):
        self.get_game_details(raw_game_id)
        return dict(self._catalog.get(raw_game_id, {}))

    def get_user_played_games(self, raw_user_id):
        user = self._users.get(raw_user_id)
        if user is None:
            raise UnknownUserError("Source user is absent from the trained mapping")
        return [self.get_game_details(self._items[i]) for i in self._history(user)]

    def recommend_games(
        self,
        raw_user_id: str,
        top_k: int = 10,
        exclude_played: bool = True,
        genre_filter: Any = None,
        max_price: Optional[float] = None,
        *,
        extra_seen: Any = (),
    ) -> List[RecommendationResult]:
        user = self._users.get(raw_user_id)
        fallback = user is None
        if fallback and not self.allow_cold_start_fallback:
            raise UnknownUserError("No trained profile for this source user; enable popularity fallback explicitly")
        scores = self._popularity.counts if fallback else self._model.score_all(user)
        excluded = set()
        if exclude_played:
            if user is not None:
                excluded.update(self._history(user))
            excluded.update(self._item_map[g] for g in extra_seen if g in self._item_map)

        order = np.lexsort((np.arange(len(self._items)), -scores))
        results = []
        for idx in order:
            if idx in excluded:
                continue
            gid = self._items[idx]
            game = self.get_game_details(gid)
            if genre_filter is not None:
                g_targets = {genre_filter.casefold()} if isinstance(genre_filter, str) else {g.casefold() for g in genre_filter}
                if not g_targets.intersection(g.casefold() for g in game.genres):
                    continue
            if max_price is not None and (game.price is None or game.price > max_price):
                continue
            results.append(
                RecommendationResult(
                    rank=len(results) + 1,
                    game_id=game.item_id,
                    title=game.title,
                    genres=game.genres,
                    predicted_score=float(scores[idx]),
                    price=game.price,
                    developer=game.developer,
                    model_name="Popularity" if fallback else self.model_name,
                    is_fallback=fallback,
                )
            )
            if len(results) == top_k:
                break
        return results


class PrimaryPrefersLastItemModel(StubModel):
    def score_items(self, user_id: int, item_ids: Any) -> np.ndarray:
        ids = np.asarray(item_ids, dtype=np.int64)
        return ids.astype(float)


class TestPretrained(unittest.TestCase):
    def setUp(self):
        self.data = StubDataset()
        self.model = StubModel()
        self.fallback_model = StubPopularityRecommender()

    def test_pretrained_service_facade_basic_and_extra_seen(self):
        self.model.fit(self.data)
        user_seen = {0: {0, 1}, 1: {2}, 2: {3}}
        user_pos = {0: {0}, 1: {2}, 2: set()}

        facade = PretrainedServiceFacade(
            model=self.model,
            user_mapper=self.data.user_mapper,
            item_mapper=self.data.item_mapper,
            catalog=self.data._games,
            user_seen_items=user_seen,
            user_positive_items=user_pos,
            fallback_model=self.fallback_model,
            allow_cold_start_fallback=True,
            train_dataset=self.data,
        )

        results = facade.recommend_games("u0", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].game_id, "g2")

        facade.record_user_interaction("u0", "g2", rating=1.0)
        played_games = facade.get_user_played_games("u0")
        played_ids = [g.item_id for g in played_games]
        self.assertIn("g2", played_ids)

        new_results = facade.recommend_games("u0", top_k=1)
        self.assertEqual(len(new_results), 1)
        self.assertEqual(new_results[0].game_id, "g3")

    def test_adapter_baseline_always_uses_global_fallback_scores(self):
        primary = PrimaryPrefersLastItemModel()
        primary.fit(self.data)
        fallback = StubPopularityRecommender()
        fallback.fit(self.data)
        facade = PretrainedServiceFacade(
            model=primary,
            user_mapper=self.data.user_mapper,
            item_mapper=self.data.item_mapper,
            catalog=self.data._games,
            user_seen_items={0: {0, 1}, 1: {2}, 2: {3}},
            user_positive_items={0: {0}, 1: {2}, 2: set()},
            fallback_model=fallback,
            allow_cold_start_fallback=True,
            mapping_fingerprint=self.data.mapping_fingerprint,
            train_fingerprint=self.data.fingerprint,
        )

        baseline = facade.recommend_baseline("u0", top_k=1)
        self.assertEqual(baseline[0].game_id, "g2")
        self.assertTrue(baseline[0].is_fallback)

    def test_adapter_records_interaction_for_new_session_user(self):
        self.model.fit(self.data)
        self.fallback_model.fit(self.data)
        facade = PretrainedServiceFacade(
            model=self.model,
            user_mapper=self.data.user_mapper,
            item_mapper=self.data.item_mapper,
            catalog=self.data._games,
            user_seen_items={0: {0, 1}, 1: {2}, 2: {3}},
            user_positive_items={0: {0}, 1: {2}, 2: set()},
            fallback_model=self.fallback_model,
            allow_cold_start_fallback=True,
            mapping_fingerprint=self.data.mapping_fingerprint,
            train_fingerprint=self.data.fingerprint,
        )

        facade.record_user_interaction("new-user", "g0")
        self.assertEqual([g.item_id for g in facade.get_user_played_games("new-user")], ["g0"])
        results = facade.recommend_games("new-user", top_k=1)
        self.assertEqual(results[0].game_id, "g1")
        self.assertTrue(results[0].is_fallback)

    def test_adapter_fits_unfitted_fallback_only_on_real_train_dataset(self):
        self.model.fit(self.data)
        facade = PretrainedServiceFacade(
            model=self.model,
            user_mapper=self.data.user_mapper,
            item_mapper=self.data.item_mapper,
            catalog=self.data._games,
            user_seen_items={0: {0, 1}, 1: {2}, 2: {3}},
            user_positive_items={0: {0}, 1: {2}, 2: set()},
            fallback_model=StubPopularityRecommender(),
            allow_cold_start_fallback=True,
            train_dataset=self.data,
        )
        self.assertTrue(facade.fallback_model.is_fitted)
        self.assertEqual(facade.recommend_baseline(top_k=1)[0].game_id, "g0")

    def test_adapter_rejects_unfitted_fallback_without_train_dataset(self):
        self.model.fit(self.data)
        with self.assertRaises(ModelNotFittedError):
            PretrainedServiceFacade(
                model=self.model,
                user_mapper=self.data.user_mapper,
                item_mapper=self.data.item_mapper,
                catalog=self.data._games,
                user_seen_items={0: {0, 1}, 1: {2}, 2: {3}},
                fallback_model=StubPopularityRecommender(),
                allow_cold_start_fallback=True,
                mapping_fingerprint=self.data.mapping_fingerprint,
                train_fingerprint=self.data.fingerprint,
            )

    def test_adapter_baseline_skips_missing_catalog_metadata(self):
        self.model.fit(self.data)
        fallback = StubPopularityRecommender()
        fallback.fit(self.data)
        catalog = dict(self.data._games)
        del catalog["g0"]
        facade = PretrainedServiceFacade(
            model=self.model,
            user_mapper=self.data.user_mapper,
            item_mapper=self.data.item_mapper,
            catalog=catalog,
            user_seen_items={0: set(), 1: set(), 2: set()},
            fallback_model=fallback,
            allow_cold_start_fallback=True,
            mapping_fingerprint=self.data.mapping_fingerprint,
            train_fingerprint=self.data.fingerprint,
        )
        results = facade.recommend_baseline(top_k=1, exclude_played=False)
        self.assertEqual(results[0].game_id, "g1")

    def test_adapter_uses_train_dataset_mapper_order(self):
        self.model.fit(self.data)
        swapped_item_mapper = StubIdMapper(["g0", "g1", "g3", "g2", "g4"])
        facade = PretrainedServiceFacade(
            model=self.model,
            user_mapper=self.data.user_mapper,
            item_mapper=swapped_item_mapper,
            catalog=self.data._games,
            user_seen_items={0: {0, 1}, 1: {2}, 2: {3}},
            user_positive_items={0: {0}, 1: {2}, 2: set()},
            train_dataset=self.data,
        )

        results = facade.recommend_games("u0", top_k=1)
        self.assertEqual(results[0].game_id, "g2")

    def test_pretrained_service_facade_cold_start_and_unknown_user(self):
        self.model.fit(self.data)
        self.fallback_model.fit(self.data)
        user_seen = {0: {0, 1}, 1: {2}, 2: {3}}
        user_pos = {0: {0}, 1: {2}, 2: set()}

        facade = PretrainedServiceFacade(
            model=self.model,
            user_mapper=self.data.user_mapper,
            item_mapper=self.data.item_mapper,
            catalog=self.data._games,
            user_seen_items=user_seen,
            user_positive_items=user_pos,
            fallback_model=self.fallback_model,
            allow_cold_start_fallback=False,
            mapping_fingerprint=self.data.mapping_fingerprint,
            train_fingerprint=self.data.fingerprint,
        )

        with self.assertRaises(UnknownUserError):
            facade.recommend_games("unknown_person")

        with self.assertRaises(ColdStartError):
            facade.recommend_games("u2")

        facade.allow_cold_start_fallback = True
        results = facade.recommend_games("u2", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].is_fallback)

    def test_pretrained_service_facade_accepts_native_pretrained_recommender(self):
        recommender = StubRealPretrainedRecommender(allow_cold_start_fallback=False)
        facade = PretrainedServiceFacade(recommender)

        self.assertEqual(facade.model_name, "BPRMatrixFactorization")
        self.assertEqual(facade.provenance["protocol"], "legacy-kaggle-format-3")
        self.assertFalse(facade.allow_cold_start_fallback)

        # Basic personalized recommendation
        results = facade.recommend_games("u0", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].game_id, "g2")
        self.assertEqual(results[1].game_id, "g3")
        self.assertFalse(results[0].is_fallback)

        # Played games inspection
        played = facade.get_user_played_games("u0")
        played_ids = [p.item_id for p in played]
        self.assertEqual(played_ids, ["g0", "g1"])

        # Game details and catalog metadata
        game = facade.get_game_details("g0")
        self.assertEqual(game.item_id, "g0")
        meta = facade.get_catalog_metadata("g0")
        self.assertEqual(meta["image_url"], "example0.png")

        # Unknown user without fallback raises UnknownUserError
        with self.assertRaises(UnknownUserError):
            facade.recommend_games("non_existent_user")

        # Enable cold-start fallback via property setter
        facade.allow_cold_start_fallback = True
        self.assertTrue(facade.allow_cold_start_fallback)
        fallback_results = facade.recommend_games("non_existent_user", top_k=2)
        self.assertEqual(len(fallback_results), 2)
        self.assertTrue(all(r.is_fallback for r in fallback_results))
        self.assertEqual(fallback_results[0].model_name, "Popularity")

        # Session dynamic interaction recording
        facade.record_user_interaction("u0", "g2", rating=1.0)
        played_after = facade.get_user_played_games("u0")
        self.assertIn("g2", [p.item_id for p in played_after])

        # Excludes dynamically recorded game from subsequent recommendations
        rec_after = facade.recommend_games("u0", top_k=1)
        self.assertEqual(rec_after[0].game_id, "g3")

        # UI helper methods
        self.assertEqual(facade.get_available_user_ids(), ("u0", "u1", "u2"))
        self.assertEqual(facade.get_available_game_ids(), ("g0", "g1", "g2", "g3"))
        genres = facade.get_available_genres()
        self.assertIn("Action", genres)
        self.assertIn("RPG", genres)

        # Baseline recommendations (u0 has seen g0, g1, g2 out of 4 items -> returns remaining g3)
        baseline = facade.recommend_baseline("u0", top_k=2)
        self.assertEqual(len(baseline), 1)
        self.assertEqual(baseline[0].game_id, "g3")
        self.assertTrue(baseline[0].is_fallback)

        # A bundle without its source train dataset exposes metadata only.
        with self.assertRaises(ConfigurationError):
            _ = facade.dataset
        self.assertEqual(facade.metadata["data_fingerprint"], "original-training")

    def test_native_uses_explicit_source_train_dataset(self):
        recommender = StubRealPretrainedRecommender()
        recommender.train_dataset = self.data
        facade = PretrainedServiceFacade(recommender)
        self.assertIs(facade.dataset, self.data)

    def test_native_records_interaction_for_new_session_user(self):
        recommender = StubRealPretrainedRecommender(allow_cold_start_fallback=True)
        facade = PretrainedServiceFacade(recommender, allow_cold_start_fallback=True)

        facade.record_user_interaction("new-user", "g2")
        self.assertEqual([g.item_id for g in facade.get_user_played_games("new-user")], ["g2"])
        results = facade.recommend_games("new-user", top_k=1)
        self.assertEqual(results[0].game_id, "g0")

    def test_load_pretrained_factory(self):
        fake_module = type("FakeModule", (), {"PretrainedRecommender": StubRealPretrainedRecommender})
        with patch("src.service.pretrained.import_module", return_value=fake_module):
            facade = PretrainedServiceFacade.load("fake/bundle/path", allow_cold_start_fallback=True)
            self.assertEqual(facade.model_name, "BPRMatrixFactorization")
            self.assertTrue(facade.allow_cold_start_fallback)

            svc_facade = RecommendationService.load_pretrained("fake/bundle/path", allow_cold_start_fallback=True)
            self.assertEqual(svc_facade.model_name, "BPRMatrixFactorization")


if __name__ == "__main__":
    unittest.main()
