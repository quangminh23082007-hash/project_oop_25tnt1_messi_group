import unittest

from src.contracts.dto import DatasetSplits
from src.contracts.exceptions import (
    DataValidationError,
    EmptyDatasetError,
    InvalidParameterError,
)
from src.service.recommendation_service import RecommendationService
from tests.service.conftest import (
    BrokenModel,
    StubDataset,
    StubModel,
)


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.data = StubDataset()
        self.model = StubModel()

    def test_train_success(self):
        svc = RecommendationService(self.model)
        ret = svc.train(self.data)
        self.assertIs(ret, svc)
        self.assertTrue(svc.is_ready)
        self.assertIs(svc.dataset, self.data)

    def test_train_empty_dataset_raises_empty_dataset_error(self):
        svc = RecommendationService(self.model)
        empty_data = StubDataset(interactions=[])
        with self.assertRaises(EmptyDatasetError):
            svc.train(empty_data)
        self.assertFalse(svc.is_ready)

    def test_train_non_train_split_raises_data_validation_error(self):
        svc = RecommendationService(self.model)
        val_data = StubDataset(split_name="validation")
        with self.assertRaises(DataValidationError):
            svc.train(val_data)
        self.assertFalse(svc.is_ready)

    def test_train_invalid_validation_score_raises_invalid_param(self):
        svc = RecommendationService(self.model)
        with self.assertRaises(InvalidParameterError):
            svc.train(self.data, validation_score="not_callable")

    def test_train_failure_leaves_service_not_ready(self):
        svc = RecommendationService(BrokenModel())
        with self.assertRaises(RuntimeError):
            svc.train(self.data)
        self.assertFalse(svc.is_ready)

    def test_train_with_dataset_splits(self):
        splits = DatasetSplits(
            train=self.data,
            validation=StubDataset(split_name="validation"),
            test=StubDataset(split_name="test"),
        )
        svc = RecommendationService(self.model)
        svc.train(splits)
        self.assertTrue(svc.is_ready)
        self.assertIs(svc.train_dataset, self.data)

    def test_train_dataset_property_alias(self):
        svc = RecommendationService(self.model).train(self.data)
        self.assertIs(svc.train_dataset, svc.dataset)

    def test_train_with_verbose(self):
        svc = RecommendationService(self.model)
        svc.train(self.data, verbose=True)
        self.assertTrue(svc.is_ready)

    def test_swap_model_with_retrain(self):
        svc = RecommendationService(self.model).train(self.data)
        new_model = StubModel(name="NewRecommender")
        svc.swap_model(new_model, retrain_immediately=True)
        self.assertEqual(svc.model_name, "NewRecommender")
        self.assertTrue(new_model.is_fitted)

    def test_swap_model_without_retrain(self):
        svc = RecommendationService(self.model).train(self.data)
        new_model = StubModel(name="UnfittedRecommender")
        svc.swap_model(new_model, retrain_immediately=False)
        self.assertEqual(svc.model_name, "UnfittedRecommender")
        self.assertFalse(new_model.is_fitted)


    def test_train_failure_raises_specific_training_error(self):
        from src.contracts.exceptions import TrainingError
        svc = RecommendationService(BrokenModel())
        with self.assertRaises(TrainingError):
            svc.train(self.data)
        self.assertFalse(svc.is_ready)


if __name__ == "__main__":
    unittest.main()
