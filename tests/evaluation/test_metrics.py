import csv
import json
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np

from src.contracts.dto import (
    EvaluationResult,
    Interaction,
    SkippedUser,
    UserEvaluation,
)
from src.contracts.exceptions import (
    ConfigurationError,
    DataValidationError,
    InvalidParameterError,
)
from src.contracts.interfaces import BaseMetric
from src.evaluation import protocol as protocol_module
from src.evaluation import result as result_module
from src.evaluation.metrics import NDCGAtK, RankingMetric, RecallAtK
from src.evaluation.protocol import EvaluationProtocol, validate_interactions
from src.evaluation.result import save_results, summarize_result


def make_result(name="Model", metric="Recall@2"):
    user = UserEvaluation(
        0,
        {metric: 0.5},
        (1,),
        frozenset({1, 2}),
        3,
    )
    return EvaluationResult(
        model_name=name,
        split_name="test",
        metrics=dict(user.metrics),
        per_user=(user,),
        skipped_users=(
            SkippedUser(1, "no_positive_train"),
        ),
    )


class MetricTests(unittest.TestCase):
    def test_shared_base_and_readonly_cutoff(self):
        self.assertTrue(issubclass(RecallAtK, BaseMetric))
        self.assertTrue(issubclass(NDCGAtK, BaseMetric))

        with self.assertRaises(TypeError):
            RankingMetric(2)

        metric = RecallAtK(2)
        self.assertEqual(metric.name, "Recall@2")

        with self.assertRaises(AttributeError):
            metric.k = 3

    def test_formulas_cutoff_and_discount(self):
        self.assertEqual(
            RecallAtK(3).compute({1, 2, 3, 4}, [2, 5, 1, 3]),
            0.5,
        )
        self.assertEqual(
            NDCGAtK(3).compute({2}, [4, 5, 2]),
            0.5,
        )
        self.assertEqual(
            NDCGAtK(3).compute({1, 2}, [2, 1, 3]),
            1.0,
        )
        self.assertEqual(
            RecallAtK(2).compute({1}, [2, 3, 1]),
            0.0,
        )

    def test_short_list_and_empty_inputs(self):
        expected = 1 / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(
            NDCGAtK(2).compute({1, 2}, [1]),
            expected,
        )

        for metric in (RecallAtK(2), NDCGAtK(2)):
            self.assertEqual(metric.compute({1}, []), 0.0)
            self.assertEqual(metric.compute([], [], 0), 0.0)

    def test_numpy_generator_and_input_preservation(self):
        ranking = np.array([3, 1, 2], dtype=np.int64)
        original = ranking.copy()
        relevant = (item for item in [1, 1, 2])

        self.assertEqual(
            RecallAtK(2).compute(relevant, ranking, 4),
            0.5,
        )
        np.testing.assert_array_equal(ranking, original)

    def test_duplicates_only_rejected_in_top_k(self):
        for metric in (RecallAtK(2), NDCGAtK(2)):
            with self.assertRaises(DataValidationError):
                metric.compute({1}, [1, 1, 2])

            self.assertEqual(
                metric.compute({1}, [1, 2, 2]),
                1.0,
            )

    def test_invalid_cutoffs_ids_shapes_and_ranges(self):
        for cutoff in (True, 0, -1, 1.5, "2"):
            with self.subTest(cutoff=cutoff):
                with self.assertRaises(InvalidParameterError):
                    RecallAtK(cutoff)

        for item in (True, -1, 1.0, "1"):
            with self.subTest(item=item):
                with self.assertRaises(DataValidationError):
                    RecallAtK(2).compute({1}, [item])

        with self.assertRaises(DataValidationError):
            RecallAtK(2).compute({1}, np.array([[1]]))

        with self.assertRaises(DataValidationError):
            RecallAtK(2).compute({2}, [1], 2)

        for ranking in ({1, 2}, {1: 1}, "12"):
            with self.subTest(ranking=ranking):
                with self.assertRaises(TypeError):
                    RecallAtK(2).compute({1}, ranking)


class ProtocolTests(unittest.TestCase):
    def test_shared_identity_binary_labels_and_immutability(self):
        self.assertIs(protocol_module.Interaction, Interaction)

        row = Interaction(0, 1, 1)
        with self.assertRaises(FrozenInstanceError):
            row.item_id = 2

        for label in (True, 0.5, 5, math.nan):
            with self.subTest(label=label):
                with self.assertRaises(DataValidationError):
                    Interaction(0, 1, label)

        protocol = EvaluationProtocol(ks=[10, 5])

        self.assertEqual(protocol.ks, (5, 10))
        self.assertEqual(protocol.max_k, 10)
        self.assertTrue(protocol.is_positive(1))
        self.assertFalse(protocol.is_positive(0))
        self.assertEqual(protocol.to_dict()["ks"], [5, 10])

    def test_invalid_protocol_parameters(self):
        cases = (
            {"ks": ()},
            {"ks": (1, 1)},
            {"ks": (True,)},
            {"positive_threshold": 4},
            {"exclude_seen": 1},
            {"cold_start_policy": "zero"},
            {"empty_ground_truth_policy": "bad"},
        )

        for values in cases:
            with self.subTest(values=values):
                with self.assertRaises(ConfigurationError):
                    EvaluationProtocol(**values)

    def test_interaction_validation_order_duplicates_and_ranges(self):
        rows = (
            Interaction(0, 1, 1),
            Interaction(1, 0, 0),
        )

        self.assertEqual(
            validate_interactions(iter(rows), 2, 2),
            rows,
        )
        self.assertEqual(
            validate_interactions([], 2, 2),
            (),
        )

        invalid = (
            rows + (rows[0],),
            (Interaction(2, 0, 1),),
            ((0, 0, 1),),
        )
        for bad in invalid:
            with self.subTest(rows=bad):
                with self.assertRaises(DataValidationError):
                    validate_interactions(bad, 2, 2)

        with self.assertRaises(InvalidParameterError):
            validate_interactions([], True, 2)


class ResultTests(unittest.TestCase):
    def test_shared_records_summary_and_replace(self):
        self.assertIs(
            result_module.EvaluationResult,
            EvaluationResult,
        )
        self.assertIs(
            result_module.UserEvaluation,
            UserEvaluation,
        )
        self.assertIs(
            result_module.SkippedUser,
            SkippedUser,
        )

        result = make_result()
        summary = summarize_result(result)

        self.assertEqual(summary["num_evaluated_users"], 1)
        self.assertEqual(summary["num_skipped_users"], 1)
        self.assertEqual(
            summary["skipped_reasons"],
            {"no_positive_train": 1},
        )

        with self.assertRaises(FrozenInstanceError):
            result.fit_seconds = 1

        self.assertEqual(
            replace(result, fit_seconds=1).fit_seconds,
            1,
        )
        self.assertEqual(result.fit_seconds, 0)

    def test_invalid_results_are_rejected(self):
        result = make_result()
        user = result.per_user[0]

        invalid = (
            replace(result, metrics={"Recall@2": 1.0}),
            replace(result, fit_seconds=math.inf),
            replace(result, per_user=[user]),
            replace(result, per_user=(user, user)),
            replace(
                result,
                skipped_users=(SkippedUser(0, "overlap"),),
            ),
            replace(
                result,
                skipped_users=(SkippedUser(1, ""),),
            ),
            replace(
                result,
                per_user=(
                    replace(user, relevant_items=(1, 2)),
                ),
            ),
            replace(
                result,
                per_user=(
                    replace(user, recommended_items=(1, 1)),
                ),
            ),
            replace(
                result,
                per_user=(
                    replace(user, candidate_count=0),
                ),
            ),
        )

        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(DataValidationError):
                    summarize_result(value)

        user.metrics["Recall@2"] = math.nan
        with self.assertRaises(DataValidationError):
            summarize_result(result)

    def test_four_files_and_all_skipped_without_fabricated_scores(self):
        empty = EvaluationResult(
            "Skipped",
            "test",
            skipped_users=(
                SkippedUser(2, "empty_ground_truth"),
            ),
        )
        self.assertEqual(
            summarize_result(empty)["status"],
            "no_eligible_users",
        )

        with tempfile.TemporaryDirectory() as temporary:
            paths = save_results(
                [make_result("Mô hình"), empty],
                temporary,
                {"seed": 42},
            )
            self.assertEqual(
                set(paths),
                {"summary", "per_user", "skipped_users", "run_config"},
            )

            with paths["summary"].open(
                encoding="utf-8",
                newline="",
            ) as stream:
                rows = list(csv.DictReader(stream))

            self.assertEqual(rows[0]["model_name"], "Mô hình")
            self.assertEqual(rows[1]["Recall@2"], "")

            with paths["per_user"].open(
                encoding="utf-8",
                newline="",
            ) as stream:
                users = list(csv.DictReader(stream))

            self.assertEqual(
                json.loads(users[0]["recommended_items"]),
                [1],
            )
            self.assertEqual(
                json.loads(users[0]["relevant_items"]),
                [1, 2],
            )

            with paths["skipped_users"].open(
                encoding="utf-8",
                newline="",
            ) as stream:
                self.assertEqual(
                    len(list(csv.DictReader(stream))),
                    2,
                )

            self.assertEqual(
                json.loads(
                    paths["run_config"].read_text(encoding="utf-8")
                ),
                {"seed": 42},
            )

    def test_overwrite_schema_and_nonfinite_config_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            sentinel = output / "summary.csv"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                save_results([make_result()], output, {})

            self.assertEqual(
                sentinel.read_text(encoding="utf-8"),
                "keep",
            )

            fresh = output / "fresh"

            with self.assertRaises(DataValidationError):
                save_results(
                    [
                        make_result(),
                        make_result("Other", "NDCG@2"),
                    ],
                    fresh,
                    {},
                )

            with self.assertRaises(DataValidationError):
                save_results(
                    [make_result()],
                    fresh,
                    {"bad": math.nan},
                )

            self.assertFalse(fresh.exists())


if __name__ == "__main__":
    unittest.main()