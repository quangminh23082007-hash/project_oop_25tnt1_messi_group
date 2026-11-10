import unittest

from src.service.recommendation_service import RecommendationService
from tests.service.conftest import (
    StubDataset,
    StubModel,
)


class TestFormatting(unittest.TestCase):
    def setUp(self):
        self.data = StubDataset()
        self.model = StubModel()

    def test_format_recommendations_table(self):
        svc = RecommendationService(self.model).train(self.data)
        results = svc.recommend_games("u0", top_k=2)
        table = svc.format_recommendations_table(results)
        self.assertIn("Rank", table)
        self.assertIn("Game ID", table)
        self.assertIn("g2", table)
        self.assertIn("g3", table)
        empty_table = svc.format_recommendations_table([])
        self.assertEqual(empty_table, "No matching game recommendations found.")

    def test_package_exports_helpers(self):
        import src.service as svc_pkg
        self.assertTrue(hasattr(svc_pkg, "RecommendationService"))
        self.assertTrue(hasattr(svc_pkg, "format_recommendations_table"))
        self.assertTrue(hasattr(svc_pkg, "FilterCriteria"))
        self.assertTrue(hasattr(svc_pkg, "parse_filter_criteria"))
        self.assertTrue(hasattr(svc_pkg, "game_passes_filters"))
        self.assertTrue(hasattr(svc_pkg, "to_recommendation_result"))


if __name__ == "__main__":
    unittest.main()
