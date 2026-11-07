import unittest

from src.contracts.exceptions import (
    ConfigurationError,
    DataValidationError,
    InvalidParameterError,
    ModelNotFittedError,
)
from src.service.recommendation_service import RecommendationService
from tests.service.conftest import (
    StubBPRModel,
    StubDataset,
    StubEvaluator,
    StubModel,
    StubPopularityRecommender,
)


class TestInitialization(unittest.TestCase):
    def setUp(self):
        self.data = StubDataset()
        self.model = StubModel()
        self.evaluator = StubEvaluator()
        self.fallback_model = StubPopularityRecommender()

    def test_init_valid(self):
        svc = RecommendationService(
            self.model,
            self.evaluator,
            allow_cold_start_fallback=True,
            fallback_model=self.fallback_model,
        )
        self.assertFalse(svc.is_ready)
        self.assertEqual(svc.model_name, "StubModel")
        self.assertTrue(svc.allow_cold_start_fallback)

    def test_init_invalid_model_raises_invalid_param(self):
        with self.assertRaises(InvalidParameterError):
            RecommendationService("not_a_model")

    def test_init_invalid_evaluator_raises_invalid_param(self):
        with self.assertRaises(InvalidParameterError):
            RecommendationService(self.model, evaluator="not_an_evaluator")

    def test_init_invalid_cold_start_flag_raises_invalid_param(self):
        with self.assertRaises(InvalidParameterError):
            RecommendationService(self.model, allow_cold_start_fallback="True")

    def test_init_rejects_bpr_as_fallback_model(self):
        bpr_model = StubBPRModel(name="BPRMatrixFactorization")
        with self.assertRaises(ConfigurationError):
            RecommendationService(self.model, fallback_model=bpr_model)

    def test_init_allows_fallback_model_with_bpr_in_display_name(self):
        global_model = StubPopularityRecommender(name="BPR comparison fallback")
        svc = RecommendationService(
            self.model,
            allow_cold_start_fallback=True,
            fallback_model=global_model,
        )
        self.assertIs(svc.fallback_model, global_model)

    def test_init_rejects_fallback_without_score_global(self):
        no_score_global_model = StubModel(name="CustomModel")
        with self.assertRaises(InvalidParameterError):
            RecommendationService(self.model, fallback_model=no_score_global_model)

    def test_init_with_pre_fitted_model_and_train_dataset(self):
        self.model.fit(self.data)
        svc = RecommendationService(self.model, train_dataset=self.data)
        self.assertTrue(svc.is_ready)
        self.assertIs(svc.dataset, self.data)

    def test_init_with_unfitted_model_and_train_dataset_raises(self):
        with self.assertRaises(ModelNotFittedError):
            RecommendationService(self.model, train_dataset=self.data)

    def test_init_with_non_train_dataset_raises(self):
        val_data = StubDataset(split_name="validation")
        self.model.fit(self.data)
        with self.assertRaises(DataValidationError):
            RecommendationService(self.model, train_dataset=val_data)

    def test_operations_before_training_raise_model_not_fitted(self):
        svc = RecommendationService(self.model)
        self.assertFalse(svc.is_ready)
        with self.assertRaises(ModelNotFittedError):
            _ = svc.dataset
        with self.assertRaises(ModelNotFittedError):
            svc.recommend_games("u0")
        with self.assertRaises(ModelNotFittedError):
            svc.get_user_played_games("u0")
        with self.assertRaises(ModelNotFittedError):
            svc.get_game_details("g0")
        with self.assertRaises(ModelNotFittedError):
            svc.evaluate_model(StubDataset(split_name="validation"))

    def test_evaluator_property_and_setter(self):
        svc = RecommendationService(self.model)
        self.assertIsNone(svc.evaluator)
        svc.evaluator = self.evaluator
        self.assertIs(svc.evaluator, self.evaluator)
        with self.assertRaises(InvalidParameterError):
            svc.evaluator = "invalid_evaluator"

    def test_model_property(self):
        svc = RecommendationService(self.model)
        self.assertIs(svc.model, self.model)

    def test_fallback_model_property_and_setter(self):
        svc = RecommendationService(self.model)
        self.assertIsNone(svc.fallback_model)
        svc.fallback_model = self.fallback_model
        self.assertIs(svc.fallback_model, self.fallback_model)


    def test_dynamic_allow_cold_start_fallback_toggle(self):
        svc = RecommendationService(self.model, fallback_model=self.fallback_model)
        svc.train(self.data)
        self.assertFalse(svc.allow_cold_start_fallback)

        # Toggle to True
        svc.allow_cold_start_fallback = True
        self.assertTrue(svc.allow_cold_start_fallback)
        self.assertTrue(svc.fallback_model.is_fitted)

        # Invalid toggle
        with self.assertRaises(InvalidParameterError):
            svc.allow_cold_start_fallback = "True"


if __name__ == "__main__":
    unittest.main()
