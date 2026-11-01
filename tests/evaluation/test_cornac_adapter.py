from importlib.util import find_spec
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.cornac_adapter import CornacMostPopAdapter
from src.evaluation.protocol import Interaction


class FakeDataset:
    @classmethod
    def from_uir(cls, rows, seed=None):
        users = {}
        items = {}
        for user_id, item_id, _ in rows:
            users.setdefault(user_id, len(users))
            items.setdefault(item_id, len(items))
        return SimpleNamespace(uid_map=users, iid_map=items, rows=rows, seed=seed)


class FakeMostPop:
    def __init__(self, name):
        self.name = name
        self.dataset = None
        self.last_user = None

    def fit(self, dataset):
        self.dataset = dataset
        self.scores = [0.0] * len(dataset.iid_map)
        for _, item_id, _ in dataset.rows:
            self.scores[dataset.iid_map[item_id]] += 1.0
        return self

    def score(self, user_id):
        self.last_user = user_id
        return list(self.scores)


def fake_cornac():
    return SimpleNamespace(
        data=SimpleNamespace(Dataset=FakeDataset),
        models=SimpleNamespace(MostPop=FakeMostPop),
    )


def training_rows():
    return (
        Interaction(0, 6, 1),
        Interaction(1, 2, 1),
        Interaction(1, 6, 1),
        Interaction(2, 4, 0),
        Interaction(2, 1, 0),
    )


class CornacAdapterTests(unittest.TestCase):
    def make_fitted(self, **kwargs):
        model = CornacMostPopAdapter(**kwargs)
        with patch("benchmarks.cornac_adapter._load_cornac", return_value=fake_cornac()):
            model.fit(training_rows(), 4, 8)
        return model

    def test_constructor_is_lazy(self):
        with patch("benchmarks.cornac_adapter._load_cornac") as loader:
            model = CornacMostPopAdapter()
        loader.assert_not_called()
        self.assertFalse(model.is_fitted)
        self.assertEqual((model.num_users, model.num_items), (0, 0))

    def test_mapping_positive_filter_and_unseen_items(self):
        model = self.make_fitted(seed=17)
        self.assertTrue(model.is_fitted)
        self.assertEqual((model.num_users, model.num_items), (4, 8))
        self.assertEqual(model._model.dataset.iid_map, {"6": 0, "2": 1})
        self.assertEqual(model._model.dataset.seed, 17)
        self.assertEqual(model._model.dataset.rows, [("0", "6", 1.0), ("1", "2", 1.0), ("1", "6", 1.0)])
        self.assertEqual(model.score_items(1, [2, 6, 4, 0]), [1.0, 2.0, 0.0, 0.0])

    def test_negative_only_and_absent_positive_users_get_global_popularity(self):
        model = self.make_fitted()
        expected = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]
        self.assertEqual(model.score_items(2, list(range(8))), expected)
        self.assertEqual(model.score_items(3, list(range(8))), expected)
        self.assertEqual(model._model.last_user, 0)

    def test_threshold_is_applied_before_binary_conversion(self):
        model = CornacMostPopAdapter(positive_threshold=4.0)
        rows = (Interaction(0, 6, 5), Interaction(1, 2, 3), Interaction(1, 6, 4))
        with patch("benchmarks.cornac_adapter._load_cornac", return_value=fake_cornac()):
            model.fit(rows, 2, 8)
        self.assertEqual(model.score_items(0, [2, 6]), [0.0, 2.0])
        self.assertEqual(model._model.dataset.rows, [("0", "6", 1.0), ("1", "6", 1.0)])

    def test_missing_dependency_has_actionable_error(self):
        model = CornacMostPopAdapter()
        with patch("benchmarks.cornac_adapter.import_module", side_effect=ModuleNotFoundError("cornac")):
            with self.assertRaisesRegex(ImportError, "pip install cornac"):
                model.fit(training_rows(), 4, 8)
        self.assertFalse(model.is_fitted)

    def test_invalid_constructor_parameters(self):
        for name in ("", "   ", None, 7):
            with self.subTest(name=name), self.assertRaises(ValueError):
                CornacMostPopAdapter(name=name)
        for threshold in (True, "1", None):
            with self.subTest(threshold=threshold), self.assertRaises(TypeError):
                CornacMostPopAdapter(positive_threshold=threshold)
        for threshold in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                CornacMostPopAdapter(positive_threshold=threshold)
        for seed in (True, 1.5, None):
            with self.subTest(seed=seed), self.assertRaises(TypeError):
                CornacMostPopAdapter(seed=seed)
        for seed in (-1, 2**32):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                CornacMostPopAdapter(seed=seed)

    def test_invalid_fit_inputs_do_not_load_cornac(self):
        model = CornacMostPopAdapter()
        cases = (
            ((), 4, 8),
            ((Interaction(0, 0, 0),), 4, 8),
            ((Interaction(0, 0, 1), Interaction(0, 0, 1)), 4, 8),
            ((Interaction(4, 0, 1),), 4, 8),
            ((Interaction(0, 8, 1),), 4, 8),
            (training_rows(), 0, 8),
        )
        with patch("benchmarks.cornac_adapter._load_cornac") as loader:
            for rows, users, items in cases:
                with self.subTest(rows=rows, users=users, items=items), self.assertRaises((TypeError, ValueError)):
                    model.fit(rows, users, items)
        loader.assert_not_called()

    def test_scoring_validates_state_user_and_items(self):
        with self.assertRaises(RuntimeError):
            CornacMostPopAdapter().score_items(0, [0])
        model = self.make_fitted()
        self.assertEqual(model.score_items(0, []), [])
        for user in (True, "0", 0.5):
            with self.subTest(user=user), self.assertRaises(TypeError):
                model.score_items(user, [0])
        for user in (-1, 4):
            with self.subTest(user=user), self.assertRaises(ValueError):
                model.score_items(user, [0])
        for candidates in ("12", {1, 2}, None, [True], [1.5]):
            with self.subTest(candidates=candidates), self.assertRaises(TypeError):
                model.score_items(0, candidates)
        for candidates in ([-1], [8]):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                model.score_items(0, candidates)

    def test_invalid_backend_outputs_are_rejected(self):
        model = self.make_fitted()
        with patch.object(model._model, "score", return_value=[1.0]):
            with self.assertRaises(RuntimeError):
                model.score_items(0, [6])
        for invalid, exception in ((float("nan"), ValueError), (float("inf"), ValueError), (True, TypeError), ("2", TypeError)):
            with self.subTest(invalid=invalid), patch.object(model._model, "score", return_value=[invalid, 1.0]):
                with self.assertRaises(exception):
                    model.score_items(0, [6])

    def test_failed_refit_preserves_previous_fitted_model(self):
        model = self.make_fitted()
        previous_scores = model.score_items(0, [2, 6])
        with patch("benchmarks.cornac_adapter._load_cornac", side_effect=ImportError("Unavailable")):
            with self.assertRaises(ImportError):
                model.fit(training_rows(), 4, 8)
        self.assertTrue(model.is_fitted)
        self.assertEqual(model.score_items(0, [2, 6]), previous_scores)

    @unittest.skipUnless(find_spec("cornac") is not None, "Install Cornac to run the real-library integration test.")
    def test_real_cornac_matches_positive_popularity_counts(self):
        model = CornacMostPopAdapter(seed=17).fit(training_rows(), 4, 8)
        expected = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]
        self.assertEqual(model.score_items(0, list(range(8))), expected)
        self.assertEqual(model.score_items(2, list(range(8))), expected)
        self.assertEqual(model.score_items(3, list(range(8))), expected)


if __name__ == "__main__":
    unittest.main()
