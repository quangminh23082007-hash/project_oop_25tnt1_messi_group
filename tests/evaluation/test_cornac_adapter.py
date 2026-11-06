import csv
import json
import tempfile
import unittest
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter

import numpy as np

from conftest import make_splits
from src.contracts.dto import EvaluationResult, ModelConfig
from src.evaluation import EvaluationProtocol, Evaluator
from src.evaluation.result import save_results


MODELS_AVAILABLE = find_spec("src.models") is not None

if MODELS_AVAILABLE:
    from src.models import (
        BPRMatrixFactorization,
        PopularityRecommender,
        TextBPRMatrixFactorization,
    )
    from src.models.text_features import build_item_features


@unittest.skipUnless(
    MODELS_AVAILABLE,
    "src.models is not present in this checkout.",
)
class ModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.splits = make_splits()
        self.evaluator = Evaluator(
            EvaluationProtocol(ks=(1, 3))
        )
        self.model_config = ModelConfig(
            n_factors=3,
            n_epochs=2,
            learning_rate=0.01,
            reg=0.01,
            patience=1,
            random_seed=42,
        )

    def check_test_result(self, model):
        result = self.evaluator.evaluate(
            model,
            self.splits.train,
            self.splits.test,
        )

        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(
            {row.user_id for row in result.per_user},
            {0, 1, 4},
        )
        self.assertEqual(
            {
                row.user_id: row.reason
                for row in result.skipped_users
            },
            {
                2: "no_positive_train",
                3: "empty_ground_truth",
            },
        )

        for row in result.per_user:
            self.assertEqual(
                len(row.recommended_items),
                len(set(row.recommended_items)),
            )
            self.assertTrue(
                set(row.recommended_items).isdisjoint(
                    self.splits.train.get_user_seen_items(
                        row.user_id
                    )
                )
            )
            self.assertTrue(
                all(
                    0.0 <= value <= 1.0
                    for value in row.metrics.values()
                )
            )

        return result

    def validation_callback(self, calls):
        def score(candidate):
            self.assertTrue(candidate.is_fitted)
            self.assertEqual(
                candidate.mapping_fingerprint,
                self.splits.train.mapping_fingerprint,
            )
            self.assertEqual(
                candidate.train_fingerprint,
                self.splits.train.fingerprint,
            )

            result = self.evaluator.evaluate(
                candidate,
                self.splits.train,
                self.splits.validation,
            )
            self.assertIsInstance(
                result,
                EvaluationResult,
            )

            calls.append(result)
            return result.metrics["NDCG@3"]

        return score

    def test_native_models_evaluate_and_export_four_reports(self):
        models = (
            PopularityRecommender(),
            BPRMatrixFactorization(
                config=self.model_config
            ),
        )
        results = []

        for model in models:
            started = perf_counter()
            model.fit(self.splits.train)
            fit_seconds = perf_counter() - started

            results.append(
                replace(
                    self.check_test_result(model),
                    fit_seconds=fit_seconds,
                )
            )

        candidates = tuple(
            reversed(range(self.splits.train.num_items))
        )
        counts = [
            sum(
                row.item_id == item and row.rating == 1.0
                for row in self.splits.train.interactions
            )
            for item in candidates
        ]

        np.testing.assert_allclose(
            models[0].score_items(0, candidates),
            counts,
        )

        metadata = {
            "purpose": "native_model_integration",
            "configs": [
                model.config()
                for model in models
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            paths = save_results(
                results,
                directory,
                metadata,
            )
            self.assertEqual(
                set(paths),
                {"summary", "per_user", "skipped_users", "run_config"},
            )

            for path in paths.values():
                self.assertTrue(Path(path).is_file())

            expected_counts = (
                ("summary", 2),
                ("per_user", 6),
                ("skipped_users", 4),
            )
            for key, expected_rows in expected_counts:
                with paths[key].open(
                    encoding="utf-8",
                    newline="",
                ) as stream:
                    self.assertEqual(
                        len(list(csv.DictReader(stream))),
                        expected_rows,
                    )

            self.assertEqual(
                json.loads(
                    paths["run_config"].read_text(
                        encoding="utf-8"
                    )
                ),
                metadata,
            )

    def test_bpr_callback_receives_current_public_candidate(self):
        model = BPRMatrixFactorization(
            config=self.model_config
        )
        calls = []

        model.fit(
            self.splits.train,
            validation_score=self.validation_callback(
                calls
            ),
        )

        self.assertTrue(calls)
        self.assertGreaterEqual(model.best_epoch, 1)
        self.assertLessEqual(
            model.best_epoch,
            self.model_config.n_epochs,
        )
        self.check_test_result(model)

    def test_text_bpr_uses_official_train_feature_builder(self):
        def encode_batch(texts):
            return np.asarray([
                [1.0, float(len(text) % 5 + 1)]
                for text in texts
            ])

        features, report = build_item_features(
            self.splits.train,
            encode_batch,
            encoder_name="synthetic-test-encoder",
            encoder_revision="test-v1",
            batch_size=2,
        )

        self.assertEqual(report["reviews_used"], 2)
        self.assertTrue(
            any(
                row.rating == 0.0
                for row in self.splits.train.iter_reviews()
            )
        )
        self.assertEqual(
            features.train_fingerprint,
            self.splits.train.fingerprint,
        )
        self.assertEqual(
            features.text_fingerprint,
            self.splits.train.review_fingerprint,
        )

        model = TextBPRMatrixFactorization(
            config=self.model_config
        )
        calls = []

        model.fit(
            self.splits.train,
            item_features=features,
            validation_score=self.validation_callback(
                calls
            ),
        )

        self.assertTrue(calls)
        result = self.check_test_result(model)

        with tempfile.TemporaryDirectory() as directory:
            paths = save_results(
                [result],
                directory,
                {"feature_report": report},
            )
            self.assertTrue(paths["summary"].is_file())


if __name__ == "__main__":
    unittest.main()