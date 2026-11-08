import copy
import csv
import json
import tempfile
import unittest
from dataclasses import replace
from hashlib import sha256
from math import log2
from pathlib import Path
from unittest.mock import patch

import numpy as np

import benchmarks.run_benchmark as runner
from conftest import FixedScoreModel, make_splits
from src.contracts.dto import EvaluationResult, Interaction, ScoredItem
from src.contracts.exceptions import (
    ColdStartError,
    ConfigurationError,
    DataValidationError,
    EmptyDatasetError,
    ModelNotFittedError,
)
from src.contracts.interfaces import BaseEvaluator
from src.evaluation import EvaluationProtocol, Evaluator
from src.evaluation.result import summarize_result
from src.evaluation.protocol import RankingEvaluationData


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.splits = make_splits()
        self.model = FixedScoreModel(self.splits.train)
        self.evaluator = Evaluator(
            EvaluationProtocol(ks=(1, 2))
        )

    def evaluate(self, **overrides):
        arguments = {
            "model": self.model,
            "train_dataset": self.splits.train,
            "evaluation_dataset": self.splits.test,
        }
        arguments.update(overrides)
        return self.evaluator.evaluate(**arguments)

    def test_known_scores_cohort_and_shared_base(self):
        result = self.evaluate()

        self.assertIsInstance(self.evaluator, BaseEvaluator)
        self.assertEqual(
            [u.user_id for u in result.per_user],
            [0, 1, 4],
        )
        self.assertEqual(
            {u.user_id: u.reason for u in result.skipped_users},
            {
                2: "no_positive_train",
                3: "empty_ground_truth",
            },
        )
        self.assertAlmostEqual(
            result.metrics["Recall@1"],
            1 / 6,
        )
        self.assertAlmostEqual(
            result.metrics["Recall@2"],
            1 / 3,
        )
        self.assertAlmostEqual(
            result.metrics["NDCG@2"],
            1 / 3,
        )
        self.assertEqual(
            summarize_result(result)["num_evaluated_users"],
            3,
        )

    def test_seen_filter_keeps_test_answers_and_includes_negative_train(self):
        self.evaluate()
        candidates = dict(self.model.calls)

        self.assertEqual(candidates[0], (2, 3, 4, 5))
        self.assertNotIn(1, candidates[0])
        self.assertNotIn(2, candidates)
        self.assertNotIn(3, candidates)

    def test_additional_history_filters_but_does_not_create_train_profile(self):
        result = self.evaluate(
            additional_seen_dataset=self.splits.validation,
        )

        self.assertEqual(
            dict(self.model.calls)[1],
            (1, 5),
        )
        self.assertEqual(
            result.per_user[1].recommended_items,
            (1, 5),
        )
        self.assertNotIn(2, dict(self.model.calls))

    def test_split_roles_and_overlap_are_checked(self):
        bad = replace(
            self.splits.test,
            interactions=(Interaction(0, 0, 1),),
        )
        with self.assertRaises(DataValidationError):
            self.evaluate(evaluation_dataset=bad)

        with self.assertRaises(DataValidationError):
            self.evaluate(
                evaluation_dataset=self.splits.train,
            )

        with self.assertRaises(DataValidationError):
            self.evaluate(
                evaluation_dataset=self.splits.validation,
                additional_seen_dataset=self.splits.validation,
            )

    def test_fitted_state_fingerprints_and_dimensions(self):
        self.model.is_fitted = False

        with self.assertRaises(ModelNotFittedError):
            self.evaluate()

        self.model.is_fitted = True

        for attribute, value in (
            ("mapping_fingerprint", "wrong"),
            ("train_fingerprint", "wrong"),
            ("num_items", 99),
        ):
            model = FixedScoreModel(self.splits.train)
            setattr(model, attribute, value)

            with self.subTest(attribute=attribute):
                with self.assertRaises(DataValidationError):
                    self.evaluate(model=model)

        self.evaluate(
            train_dataset=replace(self.splits.train),
        )

    def test_ties_negative_scores_and_repeatability(self):
        self.model.scores.clear()
        first = self.evaluate()

        self.assertEqual(
            first.per_user[0].recommended_items,
            (2, 3),
        )
        self.assertEqual(
            first.per_user,
            self.evaluate().per_user,
        )

        self.model.output = (
            lambda user, ids: [-float(item) for item in ids]
        )
        self.assertEqual(
            self.evaluate().per_user[0].recommended_items,
            (2, 3),
        )

    def test_invalid_scores_are_rejected(self):
        factories = (
            lambda ids: [],
            lambda ids: {item: 1.0 for item in ids},
            lambda ids: np.ones((len(ids), 1)),
            lambda ids: [True] + [1.0] * (len(ids) - 1),
            lambda ids: [np.nan] * len(ids),
            lambda ids: [np.inf] * len(ids),
            lambda ids: ["1"] * len(ids),
            lambda ids: set(ids),
        )

        for factory in factories:
            self.model.output = (
                lambda user, ids, factory=factory: factory(ids)
            )
            with self.subTest(factory=factory):
                with self.assertRaises(DataValidationError):
                    self.evaluate()

    def test_empty_and_all_skipped_results(self):
        target = replace(
            self.splits.test,
            interactions=(Interaction(2, 0, 1),),
        )
        result = self.evaluate(
            evaluation_dataset=target,
        )

        self.assertEqual(
            (result.metrics, result.per_user),
            ({}, ()),
        )
        self.assertEqual(len(result.skipped_users), 1)

        result = self.evaluate(
            evaluation_dataset=replace(
                self.splits.test,
                interactions=(),
            ),
        )
        self.assertEqual(
            (
                result.metrics,
                result.per_user,
                result.skipped_users,
            ),
            ({}, (), ()),
        )

        with self.assertRaises(EmptyDatasetError):
            self.evaluate(
                train_dataset=replace(
                    self.splits.train,
                    interactions=(),
                ),
            )

    def test_alternative_policies_and_invalid_metrics(self):
        self.evaluator = Evaluator(
            EvaluationProtocol(
                ks=(1,),
                cold_start_policy="error",
            )
        )
        with self.assertRaises(ColdStartError):
            self.evaluate()

        self.evaluator = Evaluator(
            EvaluationProtocol(
                ks=(1,),
                empty_ground_truth_policy="zero",
            )
        )
        result = self.evaluate()

        user = next(
            u for u in result.per_user
            if u.user_id == 3
        )
        self.assertEqual(user.metrics["Recall@1"], 0.0)

        with self.assertRaises(ConfigurationError):
            Evaluator(
                EvaluationProtocol(),
                metrics=[object()],
            )


class CallbackModel(FixedScoreModel):
    model_kind = "bpr"
    supports_callback = True

    def __init__(self, train):
        super().__init__(train)
        self.name = "BPR"
        self.best_epoch = None
        self.fit_calls = 0

    def config(self):
        return {
            "n_epochs": 2,
            "random_seed": 42,
        }

    def fit(
        self,
        train,
        *,
        validation_score=None,
        item_features=None,
        verbose=False,
    ):
        self.fit_calls += 1

        if validation_score is not None:
            validation_score(self)
            self.best_epoch = 1

        return self


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "run_name": "unit_test",
            "seed": 42,
            "protocol": {
                "ks": [3, 5],
            },
            "data": {
                "source": "demo",
            },
            "models": [
                {
                    "kind": "demo_popularity",
                    "name": "DemoPopularity",
                },
            ],
            "benchmark": {
                "early_stopping": False,
                "selection_metric": "NDCG@5",
            },
            "output_dir": "results",
        }
        self.config_path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "evaluation.json"
        )

    def run_config(self, **kwargs):
        with patch.object(
            runner,
            "_read_config",
            return_value=copy.deepcopy(self.config),
        ):
            return runner.run_benchmark(
                self.config_path,
                **kwargs,
            )

    def test_demo_exports_four_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.run_config(output_dir=directory)

            self.assertEqual(
                set(paths),
                {"summary", "per_user", "skipped_users", "run_config"},
            )

            with paths["summary"].open(
                encoding="utf-8",
                newline="",
            ) as stream:
                row = next(csv.DictReader(stream))

            self.assertEqual(row["num_evaluated_users"], "4")
            self.assertEqual(row["num_skipped_users"], "1")
            self.assertEqual(float(row["Recall@3"]), 0.75)

            metadata = json.loads(
                paths["run_config"].read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["data"]["splits"]["train"]["num_interactions"],
                13,
            )

    def test_validation_run_does_not_score_test(self):
        roles = []
        original = runner.Evaluator.evaluate

        def evaluate(evaluator, model, train, target, **kwargs):
            roles.append(target.split_name)
            return original(
                evaluator,
                model,
                train,
                target,
                **kwargs,
            )

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                runner.Evaluator,
                "evaluate",
                evaluate,
            ):
                self.run_config(
                    output_dir=directory,
                    split="validation",
                )

        self.assertEqual(roles, ["validation"])

    def test_injected_model_callback_labels_and_data_exclusions(self):
        splits = replace(
            make_splits(),
            report={"removed_rows": 999},
        )
        model = CallbackModel(splits.train)

        self.config["data"] = {"source": "provided"}
        self.config["models"] = [
            {
                "kind": "bpr",
                "name": "BPR_fixture",
                "params": {
                    "n_epochs": 2,
                },
            },
        ]
        self.config["benchmark"]["early_stopping"] = True

        with tempfile.TemporaryDirectory() as directory:
            paths = self.run_config(
                output_dir=directory,
                dataset_splits=splits,
                models={"BPR_fixture": model},
            )

            metadata = json.loads(
                paths["run_config"].read_text(encoding="utf-8")
            )
            report = metadata["models"][0]

            self.assertEqual(report["public_model_name"], "BPR")
            self.assertEqual(report["validation_callback_count"], 1)
            self.assertEqual(model.name, "BPR")

            with paths["summary"].open(
                encoding="utf-8",
                newline="",
            ) as stream:
                row = next(csv.DictReader(stream))

            self.assertEqual(row["model_name"], "BPR_fixture")
            self.assertEqual(row["num_skipped_users"], "2")
            self.assertEqual(
                metadata["data"]["report"]["removed_rows"],
                999,
            )

    def test_invalid_config_fails_before_export(self):
        self.config["protocol"]["exclude_seen"] = False

        with patch.object(runner, "save_results") as save:
            with self.assertRaises(ConfigurationError):
                self.run_config()

        save.assert_not_called()


class FixedRankingProvider:
    def __init__(self, data):
        self.name = "RankingFixture"
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.data_fingerprint = data.data_fingerprint
        self.calls = []
        self.output = None
        self.scores = {
            0: {2: 3.0, 3: 2.0, 4: 1.0, 5: 0.0},
            1: {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.0, 5: -1.0},
        }

    def fit(self, *args, **kwargs):
        raise AssertionError("Ranking evaluation must never fit a provider.")

    def rank_items(self, user_id, *, top_k, seen_items):
        self.calls.append((user_id, top_k, seen_items))
        if self.output is not None:
            return self.output(user_id, top_k, seen_items)
        scores = self.scores.get(user_id, {})
        candidates = [
            ScoredItem(item_id, scores.get(item_id, -10.0))
            for item_id in range(self.num_items)
            if item_id not in seen_items
        ]
        return tuple(
            sorted(candidates, key=lambda item: (-item.score, item.item_idx))[:top_k]
        )


class RankingEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.relevant = {
            0: frozenset({2, 4}),
            1: frozenset({3}),
            2: frozenset({5}),
            3: frozenset(),
        }
        self.seen = {
            0: frozenset({0, 1}),
            1: frozenset({0}),
            2: frozenset({2}),
            3: frozenset({3}),
        }
        self.data = RankingEvaluationData(
            split_name="test",
            num_users=4,
            num_items=6,
            user_ids=(0, 1, 2, 3),
            get_relevant_items=lambda user_id: self.relevant[user_id],
            get_seen_items=lambda user_id: self.seen[user_id],
            has_positive_train=lambda user_id: user_id != 2,
            data_fingerprint=sha256(b"ranking-evaluation-fixture-v1").hexdigest(),
        )
        self.provider = FixedRankingProvider(self.data)
        self.evaluator = Evaluator(EvaluationProtocol(ks=(1, 2)))

    def evaluate(self, **overrides):
        return self.evaluator.evaluate_rankings(
            overrides.get("model", self.provider),
            overrides.get("data", self.data),
        )

    def test_known_metrics_shared_result_and_macro(self):
        result = self.evaluate()

        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(tuple(user.user_id for user in result.per_user), (0, 1))
        self.assertEqual(
            {user.user_id: user.reason for user in result.skipped_users},
            {2: "no_positive_train", 3: "empty_ground_truth"},
        )
        self.assertAlmostEqual(result.metrics["Recall@1"], 0.25)
        self.assertAlmostEqual(result.metrics["Recall@2"], 0.25)
        self.assertAlmostEqual(result.metrics["NDCG@1"], 0.5)
        self.assertAlmostEqual(
            result.metrics["NDCG@2"],
            0.5 / (1.0 + 1.0 / log2(3)),
        )
        self.assertEqual(tuple(user.candidate_count for user in result.per_user), (4, 5))
        self.assertEqual(result.fit_seconds, 0.0)
        self.assertGreaterEqual(result.evaluate_seconds, 0.0)

    def test_full_history_passed_and_cold_empty_users_not_scored(self):
        self.evaluate()

        self.assertEqual(
            self.provider.calls,
            [(0, 2, frozenset({0, 1})), (1, 2, frozenset({0}))],
        )

    def test_signed_scores_ties_and_repeatability(self):
        self.provider.scores = {}
        first = self.evaluate()

        self.assertEqual(first.per_user[0].recommended_items, (2, 3))
        self.assertEqual(first.per_user, self.evaluate().per_user)

    def test_invalid_ranking_entries_and_order_are_rejected(self):
        invalid = (
            (ScoredItem(2, 1.0), ScoredItem(2, 0.0)),
            (ScoredItem(0, 1.0), ScoredItem(2, 0.0)),
            (ScoredItem(2, 1.0), ScoredItem(6, 0.0)),
            (ScoredItem(True, 1.0), ScoredItem(2, 0.0)),
            (ScoredItem(2, True), ScoredItem(3, 0.0)),
            (ScoredItem(2, float("nan")), ScoredItem(3, 0.0)),
            (ScoredItem(2, float("inf")), ScoredItem(3, 0.0)),
            (ScoredItem(2, "1"), ScoredItem(3, 0.0)),
            (ScoredItem(3, 1.0), ScoredItem(2, 1.0)),
            (ScoredItem(2, 0.0), ScoredItem(3, 1.0)),
            ((2, 1.0), (3, 0.0)),
        )
        for ranking in invalid:
            with self.subTest(ranking=ranking):
                self.provider.output = lambda *args, ranking=ranking: ranking
                with self.assertRaises(DataValidationError):
                    self.evaluate()

    def test_top_k_must_be_filled_and_ordered(self):
        invalid = (
            (),
            (ScoredItem(2, 1.0),),
            (ScoredItem(2, 1.0), ScoredItem(3, 0.0), ScoredItem(4, -1.0)),
            {2: 1.0, 3: 0.0},
            iter((ScoredItem(2, 1.0), ScoredItem(3, 0.0))),
        )
        for ranking in invalid:
            with self.subTest(ranking=ranking):
                self.provider.output = lambda *args, ranking=ranking: ranking
                with self.assertRaises(DataValidationError):
                    self.evaluate()

    def test_provider_identity_and_dimensions_are_required(self):
        for attribute, value in (
            ("data_fingerprint", sha256(b"other-source").hexdigest()),
            ("num_users", 5),
            ("num_items", True),
            ("name", ""),
            ("rank_items", None),
        ):
            provider = FixedRankingProvider(self.data)
            setattr(provider, attribute, value)
            with self.subTest(attribute=attribute):
                with self.assertRaises(DataValidationError):
                    self.evaluate(model=provider)
            self.assertEqual(provider.calls, [])

    def test_context_cohort_identity_and_callbacks_are_validated(self):
        for changes in (
            {"user_ids": (1, 0)},
            {"user_ids": (0, 0)},
            {"user_ids": (4,)},
            {"user_ids": (True,)},
            {"user_ids": [0, 1]},
            {"split_name": "train"},
            {"num_items": 0},
            {"data_fingerprint": "fp-pretrained"},
            {"data_fingerprint": "A" * 64},
            {"get_seen_items": None},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(DataValidationError):
                    self.evaluate(data=replace(self.data, **changes))
        self.assertEqual(self.provider.calls, [])

    def test_ground_truth_history_types_ranges_and_overlap_are_checked(self):
        for changes in (
            {"get_relevant_items": lambda user: {2}},
            {"get_relevant_items": lambda user: frozenset({6})},
            {"get_relevant_items": lambda user: frozenset({True})},
            {"get_seen_items": lambda user: frozenset({-1})},
            {"get_seen_items": lambda user: frozenset({2})},
            {"has_positive_train": lambda user: 1},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(DataValidationError):
                    self.evaluate(data=replace(self.data, **changes))

    def test_cold_start_error_and_empty_ground_truth_zero(self):
        self.evaluator = Evaluator(
            EvaluationProtocol(ks=(1,), cold_start_policy="error")
        )
        with self.assertRaises(ColdStartError):
            self.evaluate(data=replace(self.data, user_ids=(2,)))
        self.assertEqual(self.provider.calls, [])

        self.evaluator = Evaluator(
            EvaluationProtocol(ks=(1,), empty_ground_truth_policy="zero")
        )
        result = self.evaluate(data=replace(self.data, user_ids=(3,)))
        self.assertEqual(result.per_user[0].metrics, {"Recall@1": 0.0, "NDCG@1": 0.0})

    def test_no_candidates_and_empty_cohort_are_reported_without_scoring(self):
        self.evaluator = Evaluator(
            EvaluationProtocol(ks=(1,), empty_ground_truth_policy="zero")
        )
        data = replace(
            self.data,
            user_ids=(3,),
            get_seen_items=lambda user: frozenset(range(6)),
        )
        result = self.evaluate(data=data)
        self.assertEqual(result.metrics, {})
        self.assertEqual(result.skipped_users[0].reason, "no_candidates")
        self.assertEqual(self.provider.calls, [])

        result = self.evaluate(data=replace(self.data, user_ids=()))
        self.assertEqual((result.metrics, result.per_user, result.skipped_users), ({}, (), ()))

    def test_smaller_catalog_returns_available_items_only(self):
        data = replace(
            self.data,
            user_ids=(0,),
            get_relevant_items=lambda user: frozenset({5}),
            get_seen_items=lambda user: frozenset(range(5)),
        )
        result = self.evaluate(data=data)
        self.assertEqual(result.per_user[0].recommended_items, (5,))
        self.assertEqual(result.per_user[0].candidate_count, 1)
        self.assertEqual(result.metrics["Recall@2"], 1.0)

    def test_new_path_requires_history_filtering(self):
        self.evaluator = Evaluator(
            EvaluationProtocol(ks=(1,), exclude_seen=False)
        )
        with self.assertRaises(ConfigurationError):
            self.evaluate()
        self.assertEqual(self.provider.calls, [])


if __name__ == "__main__":
    unittest.main()
