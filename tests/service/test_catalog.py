import unittest

from src.contracts.dto import GameItem
from src.contracts.exceptions import (
    UnknownItemError,
    UnknownUserError,
)
from src.service.recommendation_service import RecommendationService
from tests.service.conftest import (
    StubDataset,
    StubModel,
)


class TestCatalog(unittest.TestCase):
    def setUp(self):
        self.data = StubDataset()
        self.model = StubModel()

    def test_get_user_played_games(self):
        svc = RecommendationService(self.model).train(self.data)
        played = svc.get_user_played_games("u0")
        self.assertEqual(len(played), 2)
        self.assertIsInstance(played[0], GameItem)
        self.assertEqual([p.item_id for p in played], ["g0", "g1"])

    def test_get_user_played_games_unknown_user_raises(self):
        svc = RecommendationService(self.model).train(self.data)
        with self.assertRaises(UnknownUserError):
            svc.get_user_played_games("unknown_user")

    def test_get_user_played_games_catalog_missing_raises(self):
        bad_data = StubDataset(games={"g1": self.data.get_game_by_raw_id("g1")})
        svc = RecommendationService(self.model).train(bad_data)
        with self.assertRaises(UnknownItemError):
            svc.get_user_played_games("u0")

    def test_get_game_details(self):
        svc = RecommendationService(self.model).train(self.data)
        game = svc.get_game_details("g1")
        self.assertIsInstance(game, GameItem)
        self.assertEqual(game.item_id, "g1")
        self.assertEqual(game.title, "Game Title 1")

    def test_get_game_details_unknown_game_raises(self):
        svc = RecommendationService(self.model).train(self.data)
        with self.assertRaises(UnknownItemError):
            svc.get_game_details("non_existent_game")


    def test_get_user_played_games_invalid_user_id_raises(self):
        from src.contracts.exceptions import InvalidParameterError
        svc = RecommendationService(self.model).train(self.data)
        for bad_id in ("", "   ", None, 123):
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(InvalidParameterError):
                    svc.get_user_played_games(bad_id)

    def test_get_game_details_invalid_game_id_raises(self):
        from src.contracts.exceptions import InvalidParameterError
        svc = RecommendationService(self.model).train(self.data)
        for bad_id in ("", "   ", None, 123):
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(InvalidParameterError):
                    svc.get_game_details(bad_id)


if __name__ == "__main__":
    unittest.main()
