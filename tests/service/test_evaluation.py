import unittest

from src.contracts.dto import EvaluationResult
from src.contracts.exceptions import (
    DataValidationError,
    EmptyDatasetError,
)
from src.service.recommendation_service import RecommendationService
from tests.service.conftest import (
    StubDataset,
    StubEvaluator,
    StubModel,
)


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.data = StubDataset()
        self.model = StubModel()
        self.evaluator = StubEvaluator()

    def test_evaluate_model_validation_split(self):
        svc = RecommendationService(self.model, evaluator=self.evaluator).train(self.data)
        val_data = StubDataset(split_name="validation")
        result = svc.evaluate_model(val_data)
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.model_name, "StubModel")
        self.assertEqual(result.split_name, "validation")
        self.assertIn("NDCG@10", result.metrics)

    def test_evaluate_model_test_split_with_additional_seen(self):
        svc = RecommendationService(self.model, evaluator=self.evaluator).train(self.data)
        test_data = StubDataset(split_name="test")
        val_seen_data = StubDataset(split_name="validation")
        result = svc.evaluate_model(test_data, additional_seen_dataset=val_seen_data)
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.split_name, "test")

    def test_evaluate_model_additional_seen_on_validation_split_raises(self):
        svc = RecommendationService(self.model, evaluator=self.evaluator).train(self.data)
        val_data = StubDataset(split_name="validation")
        val_seen_data = StubDataset(split_name="validation")
        with self.assertRaises(DataValidationError):
            svc.evaluate_model(val_data, additional_seen_dataset=val_seen_data)

    def test_evaluate_model_mapping_fingerprint_mismatch_raises(self):
        svc = RecommendationService(self.model, evaluator=self.evaluator).train(self.data)
        bad_val_data = StubDataset(split_name="validation", mapping_fingerprint="mismatch_mfp")
        with self.assertRaises(DataValidationError):
            svc.evaluate_model(bad_val_data)

    def test_evaluate_model_dimension_mismatch_raises(self):
        svc = RecommendationService(self.model, evaluator=self.evaluator).train(self.data)
        bad_val_data = StubDataset(split_name="validation", num_items=10)
        with self.assertRaises(DataValidationError):
            svc.evaluate_model(bad_val_data)

    def test_evaluate_model_empty_dataset_raises(self):
        svc = RecommendationService(self.model, evaluator=self.evaluator).train(self.data)
        empty_val_data = StubDataset(split_name="validation", interactions=[])
        with self.assertRaises(EmptyDatasetError):
            svc.evaluate_model(empty_val_data)


    def test_evaluate_model_without_evaluator_raises_configuration_error(self):
        from src.contracts.exceptions import ConfigurationError
        svc = RecommendationService(self.model).train(self.data)
        val_data = StubDataset(split_name="validation")
        with self.assertRaises(ConfigurationError):
            svc.evaluate_model(val_data)

    def test_evaluate_model_additional_seen_invalid_split_raises(self):
        svc = RecommendationService(self.model, evaluator=self.evaluator).train(self.data)
        test_data = StubDataset(split_name="test")
        invalid_seen_data = StubDataset(split_name="test")
        with self.assertRaises(DataValidationError):
            svc.evaluate_model(test_data, additional_seen_dataset=invalid_seen_data)


if __name__ == "__main__":
    unittest.main()
