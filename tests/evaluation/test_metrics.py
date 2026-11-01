import csv
import json
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from src.evaluation.metrics import NDCGAtK, RankingMetric, RecallAtK
from src.evaluation.protocol import EvaluationProtocol, Interaction, validate_interactions
from src.evaluation.result import EvaluationResult, UserEvaluation, save_results


class MetricTests(unittest.TestCase):
    def test_metric_base_is_abstract(self):
        with self.assertRaises(TypeError):
            RankingMetric(3)

    def test_names_and_read_only_cutoff(self):
        self.assertEqual(RecallAtK(3).name, "Recall@3")
        self.assertEqual(NDCGAtK(5).name, "NDCG@5")
        metric = RecallAtK(3)
        with self.assertRaises(AttributeError):
            metric.k = 4

    def test_invalid_cutoffs(self):
        for metric_type in (RecallAtK, NDCGAtK):
            for cutoff in (True, False, 1.5, "3", None, 0, -1):
                with self.subTest(metric=metric_type.__name__, cutoff=cutoff):
                    with self.assertRaises((TypeError, ValueError)):
                        metric_type(cutoff)

    def test_recall_uses_all_relevant_items_as_denominator(self):
        self.assertEqual(RecallAtK(3).compute({1, 2, 3, 4}, [2, 5, 1, 3]), 0.5)

    def test_perfect_ranking(self):
        for metric in (RecallAtK(3), NDCGAtK(3)):
            with self.subTest(metric=metric.name):
                self.assertEqual(metric.compute({1, 2}, [2, 1, 3]), 1.0)

    def test_ndcg_discounts_lower_rank(self):
        self.assertEqual(NDCGAtK(3).compute({2}, [4, 5, 2]), 0.5)
        self.assertGreater(
            NDCGAtK(3).compute({2}, [2, 4, 5]),
            NDCGAtK(3).compute({2}, [4, 5, 2]),
        )

    def test_short_ranking_does_not_shrink_ideal_denominator(self):
        self.assertEqual(RecallAtK(2).compute({1, 2}, [1]), 0.5)
        expected = 1.0 / (1.0 + 1.0 / math.log2(3))
        self.assertAlmostEqual(NDCGAtK(2).compute({1, 2}, [1]), expected)

    def test_empty_and_no_hit_cases(self):
        for metric in (RecallAtK(2), NDCGAtK(2)):
            for ground_truth, ranking in ((set(), [1]), ({1}, []), ({1}, [2, 3])):
                with self.subTest(metric=metric.name, ground_truth=ground_truth, ranking=ranking):
                    self.assertEqual(metric.compute(ground_truth, ranking), 0.0)
            self.assertEqual(metric.compute([], [], 0), 0.0)

    def test_ground_truth_generator_and_duplicates(self):
        ground_truth = (item for item in [1, 1, 2])
        self.assertEqual(RecallAtK(1).compute(ground_truth, [1]), 0.5)

    def test_cutoff_ignores_later_hits(self):
        for metric in (RecallAtK(2), NDCGAtK(2)):
            self.assertEqual(metric.compute({1}, [2, 3, 1]), 0.0)

    def test_duplicate_top_k_items_are_rejected(self):
        for metric in (RecallAtK(2), NDCGAtK(2)):
            with self.assertRaises(ValueError):
                metric.compute({1}, [1, 1, 2])
            self.assertEqual(metric.compute({1}, [1, 2, 2]), 1.0)

    def test_invalid_item_ids_are_rejected(self):
        for item_id in (-1, True, 1.0, "1"):
            with self.subTest(item_id=item_id):
                with self.assertRaises((TypeError, ValueError)):
                    RecallAtK(2).compute([item_id], [1])
                with self.assertRaises((TypeError, ValueError)):
                    RecallAtK(2).compute({1}, [item_id])

    def test_optional_catalog_range_is_checked(self):
        for ground_truth, ranking in (({3}, [1]), ({1}, [3])):
            with self.assertRaises(ValueError):
                RecallAtK(2).compute(ground_truth, ranking, 3)
        for total in (True, 2.5, "3", -1):
            with self.subTest(total=total):
                with self.assertRaises((TypeError, ValueError)):
                    RecallAtK(2).compute({1}, [1], total)

    def test_rankings_must_be_ordered_sequences(self):
        for ranking in ({1, 2}, {1: 1}, "", "12"):
            with self.subTest(ranking=ranking):
                with self.assertRaises(TypeError):
                    RecallAtK(2).compute({1}, ranking)

    def test_inputs_are_not_mutated(self):
        ground_truth = [1, 2, 2]
        ranking = [3, 1, 2]
        RecallAtK(2).compute(ground_truth, ranking)
        self.assertEqual(ground_truth, [1, 2, 2])
        self.assertEqual(ranking, [3, 1, 2])


class ProtocolTests(unittest.TestCase):
    def test_interaction_normalizes_and_is_frozen(self):
        interaction = Interaction(0, 1, 1)
        self.assertIs(type(interaction.rating), float)
        self.assertEqual(interaction.timestamp, 0)
        with self.assertRaises(FrozenInstanceError):
            interaction.item_id = 2

    def test_interaction_rejects_invalid_fields(self):
        for field in ("user_id", "item_id", "timestamp"):
            for value in (-1, True, 1.5, "1"):
                values = {"user_id": 0, "item_id": 1, "rating": 1, "timestamp": 0}
                values[field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises((TypeError, ValueError)):
                        Interaction(**values)
        for rating in (True, "1", math.nan, math.inf, -math.inf):
            with self.subTest(rating=rating):
                with self.assertRaises((TypeError, ValueError)):
                    Interaction(0, 0, rating)

    def test_protocol_sorts_cutoffs_and_serializes(self):
        protocol = EvaluationProtocol(ks=[5, 1, 3], positive_threshold=4)
        self.assertEqual(protocol.ks, (1, 3, 5))
        self.assertEqual(protocol.max_k, 5)
        self.assertEqual(protocol.to_dict()["ks"], [1, 3, 5])
        self.assertTrue(protocol.is_positive(4))
        self.assertFalse(protocol.is_positive(3))
        json.dumps(protocol.to_dict(), allow_nan=False)
        with self.assertRaises(FrozenInstanceError):
            protocol.exclude_seen = False

    def test_protocol_rejects_invalid_cutoffs_and_policies(self):
        for cutoffs in ((), (1, 1), (0,), (-1,), (True,), (1.5,)):
            with self.subTest(cutoffs=cutoffs):
                with self.assertRaises((TypeError, ValueError)):
                    EvaluationProtocol(ks=cutoffs)
        for values in (
            {"exclude_seen": 1},
            {"empty_ground_truth_policy": "include"},
            {"cold_start_policy": "zero"},
            {"positive_threshold": math.nan},
        ):
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    EvaluationProtocol(**values)

    def test_positive_test_rejects_invalid_rating(self):
        with self.assertRaises(ValueError):
            EvaluationProtocol().is_positive(math.nan)

    def test_interaction_validation_preserves_order(self):
        interactions = [Interaction(0, 2, 0), Interaction(1, 0, 1)]
        self.assertEqual(validate_interactions(iter(interactions), 2, 3), tuple(interactions))
        self.assertEqual(validate_interactions([], 2, 3), ())

    def test_interaction_validation_rejects_duplicates_ranges_and_raw_rows(self):
        for interactions in (
            [Interaction(0, 0, 1), Interaction(0, 0, 0)],
            [Interaction(2, 0, 1)],
            [Interaction(0, 3, 1)],
            [(0, 0, 1)],
        ):
            with self.subTest(interactions=interactions):
                with self.assertRaises((TypeError, ValueError)):
                    validate_interactions(interactions, 2, 3)

    def test_interaction_validation_requires_positive_dimensions(self):
        for users, items in ((0, 3), (2, 0), (True, 3), (2, 3.0)):
            with self.subTest(users=users, items=items):
                with self.assertRaises((TypeError, ValueError)):
                    validate_interactions([], users, items)


class ResultTests(unittest.TestCase):
    def make_result(self, name="Demo"):
        user = UserEvaluation(0, {"Recall@2": 0.5, "NDCG@2": 0.6}, (1,), (1, 2), 3)
        return EvaluationResult(name, "test", dict(user.metrics), [user], {"cold_start": 1})

    def test_result_counts(self):
        result = self.make_result()
        self.assertEqual(result.num_evaluated_users, 1)
        self.assertEqual(result.num_skipped_users, 1)

    def test_invalid_user_result_is_rejected(self):
        for score in (-0.1, 1.1, math.nan, math.inf, True):
            with self.subTest(score=score):
                with self.assertRaises((TypeError, ValueError)):
                    UserEvaluation(0, {"Recall@2": score}, (), (), 0)
        with self.assertRaises(ValueError):
            UserEvaluation(0, {"Recall@2": 1.0}, (1, 1), (1,), 2)
        with self.assertRaises(ValueError):
            UserEvaluation(0, {"Recall@2": 1.0}, (1,), (1,), 0)

    def test_saved_files_have_consistent_columns_and_json_lists(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = save_results([self.make_result("Mô hình A"), self.make_result("Model B")], temporary, {"seed": 42})
            self.assertEqual(set(paths), {"summary", "per_user", "run_config"})
            with paths["summary"].open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["model_name"], "Mô hình A")
            self.assertEqual(rows[0]["num_evaluated_users"], "1")
            self.assertEqual(json.loads(rows[0]["skipped_reasons"]), {"cold_start": 1})
            with paths["per_user"].open(encoding="utf-8", newline="") as stream:
                users = list(csv.DictReader(stream))
            self.assertEqual(json.loads(users[0]["recommended_items"]), [1])
            self.assertEqual(json.loads(users[0]["relevant_items"]), [1, 2])
            self.assertEqual(json.loads(paths["run_config"].read_text(encoding="utf-8")), {"seed": 42})

    def test_existing_result_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            sentinel = Path(temporary) / "summary.csv"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                save_results([self.make_result()], temporary, {})
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertFalse((Path(temporary) / "per_user.csv").exists())

    def test_mismatched_schemas_fail_before_writing(self):
        first = self.make_result()
        second = self.make_result("Other")
        second.metrics = {"Recall@1": 1.0}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                save_results([first, second], temporary, {})
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_invalid_timings_scores_and_skip_counts_fail(self):
        for field, value in (
            ("fit_seconds", -1),
            ("evaluate_seconds", math.inf),
            ("metrics", {"Recall@2": math.nan}),
            ("skipped_reasons", {"cold_start": -1}),
        ):
            result = self.make_result()
            setattr(result, field, value)
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    with self.assertRaises((TypeError, ValueError)):
                        save_results([result], temporary, {})

    def test_mutated_per_user_metrics_are_rechecked(self):
        result = self.make_result()
        result.per_user[0].metrics["Recall@2"] = 2.0
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                save_results([result], temporary, {})

    def test_nonfinite_run_config_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                save_results([self.make_result()], temporary, {"bad": math.nan})
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
