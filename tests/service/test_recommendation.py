import unittest

from src.contracts.dto import Interaction, RecommendationResult
from src.contracts.exceptions import (
    ColdStartError,
    InvalidParameterError,
    UnknownUserError,
)
from src.service.recommendation_service import RecommendationService
from tests.service.conftest import (
    NegativeScoreModel,
    StubDataset,
    StubModel,
    StubPopularityRecommender,
)


class TestRecommendation(unittest.TestCase):
    def setUp(self):
        self.data = StubDataset()
        self.model = StubModel()
        self.fallback_model = StubPopularityRecommender()

    def test_recommend_games_success(self):
        svc = RecommendationService(self.model)
        svc.train(self.data)
        results = svc.recommend_games("u0", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], RecommendationResult)
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[1].rank, 2)
        self.assertEqual(results[0].game_id, "g2")
        self.assertEqual(results[1].game_id, "g3")
        self.assertFalse(results[0].is_fallback)

    def test_recommend_games_exclude_played_false(self):
        svc = RecommendationService(self.model)
        svc.train(self.data)
        results = svc.recommend_games("u0", top_k=5, exclude_played=False)
        rec_ids = [r.game_id for r in results]
        self.assertIn("g0", rec_ids)
        self.assertIn("g1", rec_ids)

    def test_recommend_games_unknown_user_no_fallback_raises_unknown_user(self):
        svc = RecommendationService(self.model, allow_cold_start_fallback=False)
        svc.train(self.data)
        with self.assertRaises(UnknownUserError):
            svc.recommend_games("non_existent_user")

    def test_recommend_games_cold_start_user_no_fallback_raises_cold_start_error(self):
        svc = RecommendationService(self.model, allow_cold_start_fallback=False)
        svc.train(self.data)
        with self.assertRaises(ColdStartError):
            svc.recommend_games("u2")

    def test_recommend_games_unknown_user_with_fallback(self):
        svc = RecommendationService(
            self.model,
            allow_cold_start_fallback=True,
            fallback_model=self.fallback_model,
        )
        svc.train(self.data)
        results = svc.recommend_games("non_existent_user", top_k=3)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.is_fallback for r in results))

    def test_recommend_games_cold_start_user_with_fallback_excludes_seen_items(self):
        svc = RecommendationService(
            self.model,
            allow_cold_start_fallback=True,
            fallback_model=self.fallback_model,
        )
        svc.train(self.data)
        results = svc.recommend_games("u2", top_k=5)
        rec_ids = [r.game_id for r in results]
        self.assertNotIn("g3", rec_ids)
        self.assertTrue(all(r.is_fallback for r in results))

    def test_recommend_games_genre_filter(self):
        svc = RecommendationService(self.model)
        svc.train(self.data)
        results = svc.recommend_games("u0", genre_filter="Strategy")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].game_id, "g3")

    def test_recommend_games_price_filter(self):
        svc = RecommendationService(self.model)
        svc.train(self.data)
        results = svc.recommend_games("u0", max_price=35.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].game_id, "g2")

    def test_recommend_games_invalid_top_k_raises(self):
        svc = RecommendationService(self.model).train(self.data)
        for bad_k in (0, -5, True, 1.5, "10"):
            with self.subTest(bad_k=bad_k):
                with self.assertRaises(InvalidParameterError):
                    svc.recommend_games("u0", top_k=bad_k)

    def test_recommend_games_invalid_user_id_raises(self):
        svc = RecommendationService(self.model).train(self.data)
        for bad_id in ("", "   ", None, 123):
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(InvalidParameterError):
                    svc.recommend_games(bad_id)

    def test_fallback_model_name_in_recommendation_result(self):
        svc = RecommendationService(
            self.model,
            allow_cold_start_fallback=True,
            fallback_model=self.fallback_model,
        ).train(self.data)
        results = svc.recommend_games("unknown_user", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].is_fallback)
        self.assertEqual(results[0].model_name, "Popularity")

    def test_recommend_games_overfetch_when_candidate_missing_from_catalog(self):
        games_subset = {
            "g0": self.data.get_game_by_raw_id("g0"),
            "g1": self.data.get_game_by_raw_id("g1"),
            "g3": self.data.get_game_by_raw_id("g3"),
            "g4": self.data.get_game_by_raw_id("g4"),
        }
        dataset_with_missing_g2 = StubDataset(games=games_subset)
        svc = RecommendationService(self.model).train(dataset_with_missing_g2)
        results = svc.recommend_games("u0", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].game_id, "g3")
        self.assertEqual(results[1].game_id, "g4")

    def test_recommend_games_user_seen_full_history_returns_empty(self):
        all_seen_interactions = [
            Interaction(user_id=0, item_id=i, rating=1.0, timestamp=100 + i)
            for i in range(5)
        ]
        full_history_data = StubDataset(interactions=all_seen_interactions)
        svc = RecommendationService(self.model).train(full_history_data)
        results = svc.recommend_games("u0", top_k=3, exclude_played=True)
        self.assertEqual(results, [])

    def test_recommend_games_negative_scores_sorting(self):
        neg_model = NegativeScoreModel(name="NegativeScoreModel")
        svc = RecommendationService(neg_model).train(self.data)
        results = svc.recommend_games("u0", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].game_id, "g4")
        self.assertEqual(results[0].predicted_score, -46.0)
        self.assertEqual(results[1].game_id, "g3")
        self.assertEqual(results[1].predicted_score, -47.0)

    def test_recommend_games_filter_still_achieves_top_k(self):
        svc = RecommendationService(self.model).train(self.data)
        results = svc.recommend_games("u0", top_k=2, genre_filter="Action")
        self.assertEqual(len(results), 2)
        rec_ids = [r.game_id for r in results]
        self.assertEqual(rec_ids, ["g2", "g4"])


if __name__ == "__main__":
    unittest.main()
