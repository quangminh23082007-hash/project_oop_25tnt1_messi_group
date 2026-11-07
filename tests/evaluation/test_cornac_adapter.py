import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks.cornac_adapter import CornacMostPopAdapter
from benchmarks.legacy_adapter import LegacyPrepared
from conftest import make_legacy_fixture, make_splits
from src.contracts.exceptions import (
    ConfigurationError,
    DataValidationError,
    ModelNotFittedError,
    TrainingError,
)


class FakeDataset:
    def __init__(self, num_users, num_items, uid_map, iid_map, uir_tuple, seed=None):
        self.num_users = num_users
        self.num_items = num_items
        self.uid_map = uid_map
        self.iid_map = iid_map
        self.uir_tuple = uir_tuple
        self.seed = seed

    @classmethod
    def from_uir(cls, rows, seed=None):
        users = {raw: index for index, raw in enumerate(sorted({row[0] for row in rows}, reverse=True))}
        items = {raw: index for index, raw in enumerate(sorted({row[1] for row in rows}, reverse=True))}
        return cls(len(users), len(items), users, items, (
            np.array([users[row[0]] for row in rows], dtype=np.int64),
            np.array([items[row[1]] for row in rows], dtype=np.int64),
            np.ones(len(rows)),
        ), seed=seed)


class FakeMostPop:
    def __init__(self, name):
        self.name = name

    def fit(self, dataset):
        self.dataset = dataset
        self.scores = np.bincount(dataset.uir_tuple[1], minlength=dataset.num_items).astype(float)
        return self

    def score(self, user_id):
        return self.scores.copy()


def fake_cornac():
    return types.SimpleNamespace(
        data=types.SimpleNamespace(Dataset=FakeDataset),
        models=types.SimpleNamespace(MostPop=FakeMostPop),
        __version__="TEST_ONLY_fake_cornac",
    )


class CornacAdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cornac_adapter_")
        self.addCleanup(temporary.cleanup)
        self.fixture = make_legacy_fixture(Path(temporary.name))
        self.prepared = LegacyPrepared.load(self.fixture["prepared_dir"])

    def test_v1_positive_counts_remap_and_preserve_candidate_order(self):
        with patch("benchmarks.cornac_adapter._load_cornac", return_value=fake_cornac()):
            model = CornacMostPopAdapter().fit(make_splits().train)
        np.testing.assert_array_equal(model.score_items(0, [5, 0, 1, 0]), [1, 2, 0, 2])
        self.assertEqual(model.mapping_fingerprint, make_splits().train.mapping_fingerprint)
        self.assertEqual(model.train_fingerprint, make_splits().train.fingerprint)
        self.assertIsNone(model.data_fingerprint)

    def test_v1_unsupported_callback_invalidates_previous_fit(self):
        with patch("benchmarks.cornac_adapter._load_cornac", return_value=fake_cornac()):
            model = CornacMostPopAdapter().fit(make_splits().train)
            with self.assertRaises(ConfigurationError):
                model.fit(make_splits().train, validation_score=lambda candidate: 0.5)
        self.assertFalse(model.is_fitted)
        with self.assertRaises(ModelNotFittedError):
            model.score_items(0, [0])

    def test_invalid_backend_scores_are_rejected(self):
        with patch("benchmarks.cornac_adapter._load_cornac", return_value=fake_cornac()):
            with patch.object(FakeMostPop, "score", return_value=[float("nan")] * 3):
                with self.assertRaises(DataValidationError):
                    CornacMostPopAdapter().fit(make_splits().train)

    def test_legacy_native_constructor_keeps_full_mapping_and_positive_pairs(self):
        with patch("benchmarks.cornac_adapter._load_cornac", return_value=fake_cornac()):
            model = CornacMostPopAdapter().fit_legacy(self.prepared)
        dataset = model._model.dataset
        self.assertEqual(dataset.num_users, self.prepared.num_users)
        self.assertEqual(dataset.num_items, self.prepared.num_items)
        self.assertEqual(dataset.uid_map, {raw: index for index, raw in enumerate(self.prepared.user_ids)})
        np.testing.assert_array_equal(dataset.uir_tuple[0], self.prepared.train[:, 0])
        np.testing.assert_array_equal(dataset.uir_tuple[1], self.prepared.train[:, 1])
        self.assertEqual(len(dataset.uir_tuple[2]), len(self.prepared.train))
        self.assertEqual(model.data_fingerprint, self.prepared.data_fingerprint)
        self.assertIsNone(model.mapping_fingerprint)
        self.assertIsNone(model.train_fingerprint)

    def test_legacy_ranking_filters_all_history_and_breaks_ties_by_original_item(self):
        with patch("benchmarks.cornac_adapter._load_cornac", return_value=fake_cornac()):
            model = CornacMostPopAdapter().fit_legacy(self.prepared)
        context = self.prepared.context("test")
        ranked = model.rank_items(0, top_k=10, seen_items=context.get_seen_items(0))
        self.assertEqual([row.item_idx for row in ranked], [4, 5])
        self.assertEqual([row.score for row in ranked], [1.0, 1.0])

    def test_failed_legacy_refit_invalidates_model_and_provenance(self):
        with patch("benchmarks.cornac_adapter._load_cornac", return_value=fake_cornac()):
            model = CornacMostPopAdapter().fit_legacy(self.prepared)
            with patch.object(FakeMostPop, "fit", side_effect=RuntimeError("TEST_ONLY failed fit")):
                with self.assertRaises(TrainingError):
                    model.fit_legacy(self.prepared)
        self.assertFalse(model.is_fitted)
        self.assertIsNone(model.data_fingerprint)
        with self.assertRaises(ModelNotFittedError):
            model.rank_items(0, top_k=10, seen_items=())

    def test_missing_cornac_cannot_be_silently_substituted(self):
        with patch("benchmarks.cornac_adapter._load_cornac", side_effect=ImportError("TEST_ONLY absent")):
            with self.assertRaises(ImportError):
                CornacMostPopAdapter().fit_legacy(self.prepared)

    def test_real_cornac_fit_and_scores(self):
        if importlib.util.find_spec("cornac") is None:
            self.skipTest("Cornac is not installed; this unit smoke does not replace a real benchmark run.")
        model = CornacMostPopAdapter().fit_legacy(self.prepared)
        self.assertEqual(type(model._model).__module__.split(".")[0], "cornac")
        expected = np.bincount(self.prepared.train[:, 1], minlength=self.prepared.num_items)
        np.testing.assert_array_equal(model.score_items(0, np.arange(self.prepared.num_items)), expected)
        self.assertTrue(model.metadata()["cornac_version"])


if __name__ == "__main__":
    unittest.main()
