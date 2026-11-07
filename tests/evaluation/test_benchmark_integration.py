import csv
import importlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from conftest import legacy_fingerprint, make_legacy_fixture
from src.contracts.exceptions import DataValidationError


def available(module):
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def benchmark_config(fixture, output_dir):
    return {
        "run_name": "synthetic_benchmark_test",
        "prepared_dir": str(fixture["prepared_dir"]),
        "models": [
            {"kind": "bpr", "name": "BPR", "model_dir": str(fixture["bundles"]["bpr"])},
            {"kind": "text_bpr", "name": "TextBPR", "model_dir": str(fixture["bundles"]["text_bpr"])},
            {"kind": "cornac_most_pop", "name": "CornacMostPop"},
        ],
        "protocol": {"ks": [1, 3]},
        "benchmark": {"split": "validation", "max_users": 0, "seed": 42},
        "output_dir": str(output_dir),
    }


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class BenchmarkConfigurationTests(unittest.TestCase):
    def test_cornac_is_required_before_loading_saved_models(self):
        runner = importlib.import_module("benchmarks.run_pretrained")
        with tempfile.TemporaryDirectory(prefix="benchmark_config_test_") as directory:
            root = Path(directory)
            fixture = make_legacy_fixture(root)
            config = benchmark_config(fixture, root / "results")
            config["models"] = config["models"][:2]
            path = root / "benchmark.json"
            write_json(path, config)
            with patch("benchmarks.legacy_adapter.SavedModelRanker.load") as load:
                with self.assertRaises(runner.PretrainedReportError):
                    runner.run_benchmark(path)
                load.assert_not_called()
            self.assertFalse((root / "results").exists())


@unittest.skipUnless(
    available("src.models.pretrained") and available("cornac"),
    "Real benchmark integration requires Model v1.1 and Cornac; no weights are downloaded or models retrained.",
)
class BenchmarkIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="fixed_benchmark_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = make_legacy_fixture(self.root)
        self.output = self.root / "results"
        self.config = benchmark_config(self.fixture, self.output)
        self.path = self.root / "benchmark.json"
        self.runner = importlib.import_module("benchmarks.run_pretrained")

    def run_benchmark(self, **overrides):
        write_json(self.path, self.config)
        return self.runner.run_benchmark(self.path, **overrides)

    def assert_no_exports(self):
        self.assertFalse(self.output.exists() and any(self.output.iterdir()))

    def test_real_saved_models_and_cornac_share_cohort_and_compute_fresh_metrics(self):
        with patch(
            "src.models._array.service.CompactRecommender.fit",
            side_effect=AssertionError("Saved model training must never run in a benchmark."),
        ) as fit:
            paths = self.run_benchmark()
            fit.assert_not_called()
        self.assertEqual(set(paths), {"summary", "per_user", "skipped_users", "run_config"})
        self.assertTrue(all(path.is_file() for path in paths.values()))
        summary = {row["model_name"]: row for row in read_csv(paths["summary"])}
        self.assertEqual(set(summary), {"BPR", "TextBPR", "CornacMostPop"})
        expected_ndcg = (1.0 + 1.0 / math.log2(3.0)) / 2.0
        for name in ("BPR", "TextBPR"):
            self.assertAlmostEqual(float(summary[name]["Recall@1"]), 0.5)
            self.assertAlmostEqual(float(summary[name]["NDCG@1"]), 0.5)
            self.assertAlmostEqual(float(summary[name]["Recall@3"]), 1.0)
            self.assertAlmostEqual(float(summary[name]["NDCG@3"]), expected_ndcg)
            self.assertEqual(float(summary[name]["fit_seconds"]), 0.0)
        for metric in ("Recall@1", "NDCG@1", "Recall@3", "NDCG@3"):
            self.assertAlmostEqual(float(summary["CornacMostPop"][metric]), 1.0)
        users = read_csv(paths["per_user"])
        for name in summary:
            selected = [row for row in users if row["model_name"] == name]
            self.assertEqual({int(row["user_id"]) for row in selected}, {0, 1})
            self.assertEqual({int(row["user_id"]): int(row["candidate_count"]) for row in selected}, {0: 3, 1: 2})
            for row in selected:
                recommended = json.loads(row["recommended_items"])
                user_id = int(row["user_id"])
                history = self.fixture["arrays"]["train_seen"]
                excluded = {int(item) for user, item in history if user == user_id}
                self.assertTrue(set(recommended).isdisjoint(excluded))
        self.assertEqual(read_csv(paths["skipped_users"]), [])
        recorded = json.loads(paths["run_config"].read_text(encoding="utf-8"))
        self.assertIs(recorded["recomputed_metrics"], True)
        self.assertEqual(recorded["mode"], "benchmark")
        self.assertEqual(recorded["evaluation"]["history"], "train_seen")
        self.assertEqual(recorded["evaluation"]["selected_user_count"], 2)
        self.assertEqual(recorded["evaluation"]["eligible_user_count"], 2)
        self.assertEqual(recorded["data"]["data_fingerprint"], self.fixture["fingerprint"])
        self.assertFalse((paths["summary"].parent / "original_validation.csv").exists())

    def test_test_split_and_sample_overrides_use_fixed_shared_history(self):
        destination = self.root / "override_results"
        first = self.run_benchmark(split="test", max_users=1, output_dir=destination)
        second = self.run_benchmark(split="test", max_users=1, output_dir=destination)
        self.assertNotEqual(first["summary"].parent, second["summary"].parent)
        first_users = read_csv(first["per_user"])
        second_users = read_csv(second["per_user"])
        self.assertEqual(first_users, second_users)
        self.assertEqual(len(first_users), 3)
        self.assertEqual(len({row["user_id"] for row in first_users}), 1)
        for row in first_users:
            user_id = int(row["user_id"])
            expected_count = 2 if user_id == 0 else 1
            self.assertEqual(int(row["candidate_count"]), expected_count)
            seen = self.fixture["arrays"]["validation_seen"]
            excluded = {int(item) for user, item in seen if user == user_id}
            self.assertTrue(set(json.loads(row["recommended_items"])).isdisjoint(excluded))
        recorded = json.loads(first["run_config"].read_text(encoding="utf-8"))
        self.assertEqual(recorded["evaluation"]["split"], "test")
        self.assertEqual(recorded["evaluation"]["max_users"], 1)
        self.assertEqual(recorded["evaluation"]["history"], "validation_seen")
        self.assertEqual(recorded["evaluation"]["selected_user_count"], 1)
        self.assertEqual(recorded["evaluation"]["eligible_user_count"], 2)

    def test_changed_prepared_fingerprint_is_rejected(self):
        report_path = self.fixture["prepared_dir"] / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["fingerprint"] = "0" * 64
        write_json(report_path, report)
        with self.assertRaises((self.runner.PretrainedReportError, DataValidationError)):
            self.run_benchmark()
        self.assert_no_exports()

    def test_changed_saved_item_order_is_rejected(self):
        info_path = self.fixture["bundles"]["bpr"] / "model_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["item_ids"][0], info["item_ids"][1] = info["item_ids"][1], info["item_ids"][0]
        write_json(info_path, info)
        with self.assertRaises((self.runner.PretrainedReportError, DataValidationError)):
            self.run_benchmark()
        self.assert_no_exports()

    def test_heldout_target_in_history_is_rejected_even_with_valid_fingerprint(self):
        prepared = self.fixture["prepared_dir"]
        arrays = {name: values.copy() for name, values in self.fixture["arrays"].items()}
        arrays["train_seen"] = np.unique(np.vstack([arrays["train_seen"], [[0, 3]]]), axis=0)
        np.savez_compressed(prepared / "dataset.npz", **arrays)
        mapping = json.loads((prepared / "mappings.json").read_text(encoding="utf-8"))
        report = json.loads((prepared / "report.json").read_text(encoding="utf-8"))
        report["fingerprint"] = legacy_fingerprint(mapping, arrays)
        write_json(prepared / "report.json", report)
        with self.assertRaisesRegex(DataValidationError, "overlap"):
            self.run_benchmark()
        self.assert_no_exports()


if __name__ == "__main__":
    unittest.main()
