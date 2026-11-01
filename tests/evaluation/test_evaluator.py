import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from benchmarks.run_benchmark import DemoPopularityModel, _demo_data, main, run_benchmark
from conftest import FixedScoreModel, make_evaluation_case, make_interactions
from src.evaluation import EvaluationProtocol, Evaluator, Interaction, RecallAtK


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.model, self.train, self.test = make_evaluation_case()
        self.evaluator = Evaluator(EvaluationProtocol(ks=(1, 2)))

    def evaluate(self, **overrides):
        arguments = {
            "model": self.model,
            "train_interactions": self.train,
            "test_interactions": self.test,
            "num_users": 3,
            "num_items": 6,
        }
        arguments.update(overrides)
        return self.evaluator.evaluate(**arguments)

    def test_known_scores_and_macro_average(self):
        result = self.evaluate()
        self.assertEqual(result.num_evaluated_users, 2)
        self.assertEqual(result.num_skipped_users, 1)
        self.assertEqual(result.skipped_reasons["cold_start"], 1)
        self.assertEqual(result.metrics["Recall@1"], 0.5)
        self.assertEqual(result.metrics["Recall@2"], 1.0)
        self.assertEqual(result.metrics["NDCG@1"], 0.5)
        self.assertAlmostEqual(result.metrics["NDCG@2"], 0.8154648767857288)
        self.assertEqual(result.per_user[0].recommended_items, (2, 3))
        self.assertEqual(result.per_user[1].recommended_items, (0, 3))

    def test_seen_filter_includes_negative_reviews_but_not_test(self):
        result = self.evaluate()
        first_user_candidates = dict(self.model.calls)[0]
        self.assertNotIn(0, first_user_candidates)
        self.assertNotIn(1, first_user_candidates)
        self.assertIn(2, first_user_candidates)
        self.assertIn(4, first_user_candidates)
        self.assertEqual(result.per_user[0].candidate_count, 4)

    def test_equal_scores_use_item_id_ascending(self):
        self.model.scores.clear()
        result = self.evaluate()
        self.assertEqual(result.per_user[0].recommended_items, (2, 3))

    def test_additional_history_is_filtered(self):
        additional = (Interaction(0, 3, 0),)
        result = self.evaluate(additional_seen_interactions=additional)
        self.assertNotIn(3, result.per_user[0].recommended_items)
        self.assertEqual(result.per_user[0].candidate_count, 3)

    def test_seen_filter_can_be_disabled(self):
        self.evaluator = Evaluator(EvaluationProtocol(ks=(1,), exclude_seen=False))
        self.model.scores = {(0, 0): 99.0}
        result = self.evaluate()
        self.assertEqual(result.per_user[0].recommended_items, (0,))

    def test_empty_ground_truth_skip_and_zero(self):
        test = make_interactions(((0, 2, 0), (1, 3, 1)))
        result = self.evaluate(test_interactions=test)
        self.assertEqual(result.num_evaluated_users, 1)
        self.assertEqual(result.skipped_reasons["empty_ground_truth"], 1)
        self.evaluator = Evaluator(EvaluationProtocol(ks=(1,), empty_ground_truth_policy="zero"))
        result = self.evaluate(test_interactions=test)
        self.assertEqual(result.num_evaluated_users, 2)
        self.assertEqual(result.per_user[0].metrics["Recall@1"], 0.0)

    def test_cold_start_error_policy(self):
        self.evaluator = Evaluator(EvaluationProtocol(ks=(1,), cold_start_policy="error"))
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_no_eligible_users_and_empty_input(self):
        with self.assertRaises(ValueError):
            self.evaluate(test_interactions=(Interaction(2, 5, 1),))
        with self.assertRaises(ValueError):
            self.evaluate(test_interactions=())
        with self.assertRaises(ValueError):
            self.evaluate(train_interactions=())

    def test_rejects_duplicate_and_overlapping_pairs(self):
        with self.assertRaises(ValueError):
            self.evaluate(test_interactions=self.test + (self.test[0],))
        with self.assertRaises(ValueError):
            self.evaluate(test_interactions=(Interaction(0, 0, 1),))
        with self.assertRaises(ValueError):
            self.evaluate(additional_seen_interactions=(Interaction(0, 2, 0),))
        with self.assertRaises(ValueError):
            self.evaluate(additional_seen_interactions=(Interaction(0, 0, 0),))

    def test_rejects_invalid_dimensions_and_interaction_types(self):
        for dimensions in ((True, 6), (3, 0), (3, 1.5)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises((ValueError, TypeError)):
                    self.evaluate(num_users=dimensions[0], num_items=dimensions[1])
        with self.assertRaises(TypeError):
            self.evaluate(test_interactions=((0, 2, 1),))
        with self.assertRaises(ValueError):
            self.evaluate(test_interactions=(Interaction(0, 6, 1),))

    def test_rejects_unfitted_and_incompatible_model(self):
        self.model.is_fitted = False
        with self.assertRaises(RuntimeError):
            self.evaluate()
        self.model.is_fitted = True
        self.model.num_items = 7
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_rejects_invalid_model_scores(self):
        original = self.model.score_items
        for output in ([], [0.0], [float("nan")] * 4, [float("inf")] * 4, [True] * 4, ["1"] * 4):
            with self.subTest(output=output):
                self.model.score_items = lambda user_id, item_ids, output=output: output
                with self.assertRaises((TypeError, ValueError)):
                    self.evaluate()
        self.model.score_items = original

    def test_negative_finite_scores_are_valid(self):
        self.model.scores = {(0, 2): -1.0, (0, 3): -2.0, (0, 4): -3.0, (0, 5): -4.0}
        result = self.evaluate()
        self.assertEqual(result.per_user[0].recommended_items, (2, 3))

    def test_unordered_score_containers_are_rejected(self):
        for output in (
            {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0},
            {0.0, 1.0, 2.0, 3.0},
            frozenset((0.0, 1.0, 2.0, 3.0)),
        ):
            with self.subTest(output=output):
                self.model.score_items = lambda user_id, item_ids, output=output: output
                with self.assertRaises(TypeError):
                    self.evaluate()

    def test_generator_input_and_repeat_evaluation(self):
        first = self.evaluate(train_interactions=(item for item in self.train))
        second = self.evaluate()
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(self.train[0], Interaction(0, 0, 1))

    def test_rejects_invalid_custom_metric_configuration(self):
        protocol = EvaluationProtocol(ks=(1,))
        for metrics in ([], [RecallAtK(2)], [RecallAtK(1), RecallAtK(1)], [object()]):
            with self.subTest(metrics=metrics):
                with self.assertRaises((ValueError, TypeError)):
                    Evaluator(protocol, metrics=metrics)

    def test_positive_threshold_changes_relevant_items(self):
        self.evaluator = Evaluator(EvaluationProtocol(ks=(1,), positive_threshold=4.0))
        test = (Interaction(0, 2, 5), Interaction(1, 3, 3))
        result = self.evaluate(test_interactions=test)
        self.assertEqual(result.num_evaluated_users, 1)
        self.assertEqual(result.metrics["Recall@1"], 1.0)


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="evaluation_runner_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "configs").mkdir()
        package_root = Path(__file__).resolve().parents[2]
        self.config = json.loads((package_root / "configs" / "evaluation.json").read_text(encoding="utf-8"))

    def write_config(self):
        path = self.root / "configs" / "evaluation.json"
        path.write_text(json.dumps(self.config), encoding="utf-8")
        return path

    def read_summary(self, paths):
        with paths["summary"].open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def prepare_csv_data(self):
        train, validation, test, num_users, num_items = _demo_data()
        for split_name, interactions in (("train", train), ("validation", validation), ("test", test)):
            with (self.root / f"{split_name}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("user_id", "item_id", "rating", "timestamp"))
                for interaction in interactions:
                    writer.writerow((interaction.user_id, interaction.item_id, interaction.rating, interaction.timestamp))
        self.config["data"] = {
            "source": "csv",
            "train_csv": "train.csv",
            "validation_csv": "validation.csv",
            "test_csv": "test.csv",
            "num_users": num_users,
            "num_items": num_items,
        }

    def test_demo_and_validation_runs_are_separate(self):
        paths = run_benchmark(self.write_config())
        rows = self.read_summary(paths)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["num_evaluated_users"], "4")
        self.assertEqual(rows[0]["num_skipped_users"], "1")
        self.assertEqual(rows[0]["Recall@5"], "1.0")
        config = json.loads(paths["run_config"].read_text(encoding="utf-8"))
        self.assertEqual(config["data"]["counts"], {"train": 12, "validation": 5, "test": 9})
        self.assertEqual(len(config["data"]["sha256"]["train"]), 64)
        validation_paths = run_benchmark(self.write_config(), split="validation")
        self.assertNotEqual(paths["summary"].parent, validation_paths["summary"].parent)
        self.assertTrue(paths["summary"].exists())
        self.assertEqual(self.read_summary(validation_paths)[0]["split_name"], "validation")

    def test_csv_mode_matches_demo_scores_and_fingerprints(self):
        demo_paths = run_benchmark(self.write_config())
        demo_row = self.read_summary(demo_paths)[0]
        demo_config = json.loads(demo_paths["run_config"].read_text(encoding="utf-8"))
        self.prepare_csv_data()
        csv_paths = run_benchmark(self.write_config())
        csv_row = self.read_summary(csv_paths)[0]
        csv_config = json.loads(csv_paths["run_config"].read_text(encoding="utf-8"))
        for name in ("Recall@3", "Recall@5", "NDCG@3", "NDCG@5"):
            self.assertEqual(csv_row[name], demo_row[name])
        self.assertEqual(csv_config["data"]["sha256"], demo_config["data"]["sha256"])
        self.assertEqual(csv_config["data"]["paths"]["train"], str((self.root / "train.csv").resolve()))

    def test_unknown_keys_and_invalid_seed_are_rejected(self):
        self.config["unexpected"] = True
        with self.assertRaises(ValueError):
            run_benchmark(self.write_config())
        del self.config["unexpected"]
        self.config["seed"] = True
        with self.assertRaises(TypeError):
            run_benchmark(self.write_config())
        self.config["seed"] = 42
        self.config["protocol"]["typo"] = 5
        with self.assertRaises(ValueError):
            run_benchmark(self.write_config())

    def test_csv_overlap_is_rejected_before_evaluation(self):
        self.prepare_csv_data()
        (self.root / "test.csv").write_text("user_id,item_id,rating\n0,0,1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "occurs in both"):
            run_benchmark(self.write_config())

    def test_malformed_and_nonfinite_csv_are_rejected(self):
        self.prepare_csv_data()
        for contents in (
            "user_id,item_id,rating\n0,1,nan\n",
            "user_id,item_id,rating\n0,1,1,extra\n",
            "user_id,user_id,rating\n0,1,1\n",
        ):
            with self.subTest(contents=contents):
                (self.root / "train.csv").write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError):
                    run_benchmark(self.write_config())

    def test_injected_models_and_duplicate_names(self):
        paths = run_benchmark(self.write_config(), models=[DemoPopularityModel(name="CustomModel")])
        self.assertEqual(self.read_summary(paths)[0]["model_name"], "CustomModel")
        with self.assertRaises(ValueError):
            run_benchmark(self.write_config(), models=[])
        with self.assertRaises(ValueError):
            run_benchmark(self.write_config(), models=[DemoPopularityModel(), DemoPopularityModel()])

    def test_cli_success_and_configuration_error(self):
        config_path = self.write_config()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(("--config", str(config_path), "--without-cornac"))
        self.assertEqual(exit_code, 0)
        self.assertIn("summary:", stdout.getvalue())
        self.config["benchmark"]["split"] = "invalid"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = main(("--config", str(self.write_config())))
        self.assertEqual(exit_code, 1)
        self.assertIn("Benchmark failed:", stderr.getvalue())

    def test_demo_model_training_and_score_validation(self):
        model = DemoPopularityModel()
        with self.assertRaises(RuntimeError):
            model.score_items(0, [0])
        model.fit((Interaction(0, 0, 1), Interaction(0, 1, 0), Interaction(1, 0, 1)), 2, 3)
        self.assertEqual(model.score_items(0, [0, 1, 2]), [2.0, 0.0, 0.0])
        self.assertEqual(model.score_items(1, []), [])
        with self.assertRaises(ValueError):
            model.score_items(2, [0])
        with self.assertRaises(TypeError):
            model.score_items(True, [0])


if __name__ == "__main__":
    unittest.main()
