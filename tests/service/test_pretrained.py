import unittest

from src.contracts.exceptions import (
    ColdStartError,
    UnknownUserError,
)
from src.service.pretrained import PretrainedServiceFacade
from tests.service.conftest import (
    StubDataset,
    StubModel,
    StubPopularityRecommender,
)


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
        )

        with self.assertRaises(UnknownUserError):
            facade.recommend_games("unknown_person")

        with self.assertRaises(ColdStartError):
            facade.recommend_games("u2")

        facade.allow_cold_start_fallback = True
        results = facade.recommend_games("u2", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].is_fallback)


if __name__ == "__main__":
    unittest.main()
