import copy
import csv
import hashlib
import importlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from benchmarks.pretrained_report import (
    PretrainedReportError,
    load_recorded_validation,
    write_pretrained_results,
)


def metadata(model="bpr"):
    text = model == "text_bpr"
    return {
        "format_version": 3,
        "representation": "compact",
        "model_type": "TextBPRMatrixFactorization" if text else "BPRMatrixFactorization",
        "config": {"factors": 2},
        "user_ids": ["known", "second"],
        "item_ids": ["g0", "g1", "g2", "g3"],
        "catalog": {f"g{i}": {"title": f"TEST_ONLY Game {i}"} for i in range(4)},
        "data_fingerprint": "TEST_ONLY-original-training",
        "data_config": {
            "split": "random",
            "seed": 42,
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
        },
        "data_limitations": ["TEST_ONLY synthetic fixture; recorded scores are fabricated for tests."],
        "best_epoch": 2,
        "history": [],
        "feature_info": {"encoder": {"model": "TEST_ONLY-not-used-at-inference"}} if text else None,
        "training_info": {"validation_k": 10, "validation_max_users": 2, "validation_seed": 42},
        "weights_sha256": "a" * 64,
    }


def reference(info, selected="bpr"):
    models = {}
    for name in ("bpr", "text_bpr"):
        entry = metadata(name)
        if name == selected:
            entry = copy.deepcopy(info)
        for field in ("user_ids", "item_ids", "catalog", "history"):
            entry.pop(field, None)
        entry.update(num_users=2, num_items=4)
        score = {
            "recall@10": 0.5,
            "ndcg@10": 0.25,
            "evaluated_users": 1,
            "eligible_users": 2,
            "candidate_catalog_size": 4,
            "candidate_policy": "all train items minus observed history",
        }
        popularity = dict(score, **{"recall@10": 0.25, "ndcg@10": 0.1})
        entry["validation_report"] = {
            "model": "text-bpr" if name == "text_bpr" else name,
            "best_epoch": 2,
            "validation": score,
            "popularity_validation": popularity,
        }
        models[name] = entry
    return {
        "contract_revision": "1.1",
        "delivery_mode": "pretrained-existing-run",
        "models": models,
        "note": "TEST_ONLY synthetic provenance and fabricated recorded metrics; never a real experiment.",
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pretrained_report_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.info = metadata()
        self.reference_path = self.root / "TEST_ONLY_reference.json"
        write_json(self.reference_path, reference(self.info))

    def recorded(self):
        return load_recorded_validation(
            self.reference_path, selected_model="bpr", bundle_metadata=self.info
        )

    def export(self, **overrides):
        arguments = {
            "recommendations": [],
            "inference_checks": {"all_passed": True, "users": [], "recomputed_metrics": False},
            "recorded_validation": self.recorded(),
            "run_config": {"fixture": "TEST_ONLY", "recomputed_metrics": False},
        }
        arguments.update(overrides)
        return write_pretrained_results(self.root / "results", "TEST_ONLY_run", **arguments)

    def test_reference_keeps_recorded_validation_separate_from_new_inference(self):
        result = self.recorded()
        self.assertIs(result["recomputed"], False)
        self.assertEqual(result["selected_model"], "bpr")
        self.assertEqual(result["source"]["sha256"], hashlib.sha256(self.reference_path.read_bytes()).hexdigest())
        rows = {row["model"]: row for row in result["rows"]}
        self.assertEqual(set(rows), {"bpr", "text_bpr", "popularity"})
        self.assertTrue(rows["bpr"]["primary_model_selected"])
        self.assertFalse(rows["text_bpr"]["primary_model_selected"])
        self.assertFalse(rows["popularity"]["primary_model_selected"])
        self.assertEqual(rows["bpr"]["recall_at_10"], 0.5)
        self.assertTrue(all(row["split"] == "validation" and row["recomputed"] is False for row in rows.values()))

    def test_bundle_provenance_must_match_the_reference(self):
        for field, value in (
            ("weights_sha256", "f" * 64),
            ("data_fingerprint", "TEST_ONLY-different-data"),
            ("best_epoch", 3),
            ("config", {"factors": 3}),
            ("user_ids", ["known"]),
        ):
            with self.subTest(field=field):
                changed = dict(self.info, **{field: value})
                with self.assertRaises(PretrainedReportError):
                    load_recorded_validation(self.reference_path, selected_model="bpr", bundle_metadata=changed)

    def test_reference_rejects_incomparable_model_cohorts(self):
        changed = reference(self.info)
        changed["models"]["text_bpr"]["validation_report"]["validation"]["candidate_catalog_size"] = 5
        write_json(self.reference_path, changed)
        with self.assertRaises(PretrainedReportError):
            self.recorded()

    def test_exports_use_new_directories_and_preserve_existing_results(self):
        first = self.export()
        original = {key: path.read_bytes() for key, path in first.items()}
        second = self.export()
        self.assertEqual(set(first), {"recommendations", "inference_checks", "original_validation", "run_config"})
        self.assertNotEqual(first["run_config"].parent, second["run_config"].parent)
        for key, path in first.items():
            self.assertIsInstance(path, Path)
            self.assertEqual(path.read_bytes(), original[key])
        checks = json.loads(first["inference_checks"].read_text(encoding="utf-8"))
        self.assertIs(checks["recomputed_metrics"], False)
        with first["original_validation"].open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["split"] == "validation" for row in rows))

    def test_nonfinite_export_values_are_rejected(self):
        with self.assertRaises(ValueError):
            self.export(run_config={"invalid": float("nan")})


class PretrainedIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.pretrained = importlib.import_module("src.models.pretrained")
        except ModuleNotFoundError as error:
            if error.name in {"src", "src.models", "src.models.pretrained"}:
                raise unittest.SkipTest("Model v1.1 src.models.pretrained is not installed; no weights are downloaded.") from error
            raise
        cls.numpy = importlib.import_module("numpy")
        cls.runner = importlib.import_module("benchmarks.run_pretrained")
        cls.errors = importlib.import_module("src.contracts.exceptions")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pretrained_integration_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = self.root / "bundles" / "bpr" / "cpu"
        self.reference_path = self.root / "reference.json"
        self.config_path = self.root / "settings" / "run.json"
        self.output = self.root / "results"
        self.build_bundle()

    def build_bundle(self, model="bpr"):
        self.bundle.mkdir(parents=True, exist_ok=True)
        np = self.numpy
        np.savez_compressed(
            self.bundle / "weights.npz",
            P=np.array([[1.0, 0.0], [0.0, 1.0]]),
            R=np.array([[3.0, 0.0], [2.0, 0.0], [1.0, 4.0], [0.0, 3.0]]),
            popularity=np.array([3.0, 2.0, 1.0, 0.0]),
            train_seen=np.array([[0, 0], [0, 1], [1, 2]], dtype=np.int64),
        )
        info = metadata(model)
        info["weights_sha256"] = hashlib.sha256((self.bundle / "weights.npz").read_bytes()).hexdigest()
        write_json(self.bundle / "model_info.json", info)
        write_json(self.reference_path, reference(info, selected=model))
        self.config = {
            "run_name": "TEST_ONLY_inference",
            "model": model,
            "model_dir": "bundles/bpr/cpu",
            "reference_path": "reference.json",
            "user_ids": ["known"],
            "top_k": 10,
            "exclude_played": True,
            "allow_cold_start_fallback": False,
            "extra_seen": [],
            "output_dir": "results",
        }
        write_json(self.config_path, self.config)

    def run_config(self, **overrides):
        write_json(self.config_path, self.config)
        with patch.object(self.runner, "ROOT", self.root):
            return self.runner.run_pretrained(self.config_path, **overrides)

    def recommendations(self, paths):
        with paths["recommendations"].open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def assert_no_exports(self):
        self.assertFalse(self.output.exists() and any(self.output.iterdir()))

    def test_real_loader_runs_once_and_paths_and_overrides_are_respected(self):
        self.config["user_ids"] = ["!"]
        self.config["extra_seen"] = ["g2"]
        destination = self.root / "override_results"
        with patch.object(self.runner, "_load_model", wraps=self.runner._load_model) as load:
            paths = self.run_config(user_ids=["known", "second"], output_dir=destination)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(Path(load.call_args.args[0]).resolve(), self.bundle.resolve())
        self.assertEqual(paths["recommendations"].parent.parent.resolve(), destination.resolve())
        rows = self.recommendations(paths)
        self.assertEqual([row["game_id"] for row in rows if row["user_id"] == "known"], ["g3"])
        self.assertEqual([row["game_id"] for row in rows if row["user_id"] == "second"], ["g3", "g0", "g1"])
        checks = json.loads(paths["inference_checks"].read_text(encoding="utf-8"))
        self.assertTrue(checks["all_passed"])
        self.assertIs(checks["recomputed_metrics"], False)
        self.assertEqual([row["candidate_count"] for row in checks["users"]], [1, 3])

    def test_unknown_user_requires_explicit_fallback(self):
        self.config.update(user_ids=["TEST_ONLY-unknown"], top_k=2, extra_seen=["g0"])
        with self.assertRaises((self.errors.UnknownUserError, PretrainedReportError)):
            self.run_config()
        self.assert_no_exports()
        self.config["allow_cold_start_fallback"] = True
        paths = self.run_config()
        rows = self.recommendations(paths)
        self.assertEqual([row["game_id"] for row in rows], ["g1", "g2"])
        checks = json.loads(paths["inference_checks"].read_text(encoding="utf-8"))
        self.assertFalse(checks["users"][0]["known_user"])
        self.assertTrue(checks["users"][0]["is_fallback"])

    def test_compact_text_bpr_inference_needs_no_encoder_or_training(self):
        self.build_bundle("text_bpr")
        paths = self.run_config()
        self.assertEqual([row["game_id"] for row in self.recommendations(paths)], ["g2", "g3"])
        model = self.pretrained.PretrainedRecommender.load(self.bundle)
        self.assertEqual(model.model_name, "text_bpr")
        self.assertFalse(hasattr(model, "fit"))

    def test_corrupt_weights_fail_before_any_results_are_written(self):
        with (self.bundle / "weights.npz").open("ab") as handle:
            handle.write(b"TEST_ONLY-corruption")
        with self.assertRaises((self.errors.DataValidationError, PretrainedReportError)):
            self.run_config()
        self.assert_no_exports()

    def test_reference_mismatch_fails_before_any_results_are_written(self):
        changed = json.loads(self.reference_path.read_text(encoding="utf-8"))
        changed["models"]["bpr"]["data_fingerprint"] = "TEST_ONLY-other-training-run"
        write_json(self.reference_path, changed)
        with self.assertRaises(PretrainedReportError):
            self.run_config()
        self.assert_no_exports()

    def test_malformed_model_outputs_fail_before_any_results_are_written(self):
        model = self.pretrained.PretrainedRecommender.load(self.bundle)
        valid = model.recommend_games("known", top_k=10)
        malformed = {
            "nonfinite_score": [replace(valid[0], predicted_score=float("nan")), valid[1]],
            "seen_item": [replace(valid[0], game_id="g0"), valid[1]],
            "tie_order": [
                replace(valid[1], rank=1, predicted_score=1.0),
                replace(valid[0], rank=2, predicted_score=1.0),
            ],
            "fallback_flag": [replace(valid[0], is_fallback=True), valid[1]],
        }
        for case, rows in malformed.items():
            with self.subTest(case=case):
                with patch.object(self.pretrained.PretrainedRecommender, "recommend_games", return_value=rows):
                    with self.assertRaises(PretrainedReportError):
                        self.run_config()
                self.assert_no_exports()

    def test_all_candidates_seen_produces_valid_empty_recommendations(self):
        self.config["extra_seen"] = ["g2", "g3"]
        paths = self.run_config()
        with paths["recommendations"].open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertTrue({"user_id", "rank", "game_id", "predicted_score"}.issubset(reader.fieldnames))
            self.assertEqual(list(reader), [])
        checks = json.loads(paths["inference_checks"].read_text(encoding="utf-8"))
        self.assertTrue(checks["all_passed"])
        self.assertIs(checks["recomputed_metrics"], False)
        self.assertEqual(checks["users"][0]["candidate_count"], 0)
        self.assertEqual(checks["users"][0]["returned_items"], 0)
        with paths["original_validation"].open(encoding="utf-8", newline="") as handle:
            original = list(csv.DictReader(handle))
        self.assertEqual(len(original), 3)
        self.assertTrue(all(row["recomputed"].lower() == "false" for row in original))

    def test_malformed_catalog_is_rejected_before_model_loading(self):
        info = json.loads((self.bundle / "model_info.json").read_text(encoding="utf-8"))
        for catalog in (None, {"g0": None}):
            with self.subTest(catalog=catalog):
                write_json(self.bundle / "model_info.json", dict(info, catalog=catalog))
                with patch.object(self.runner, "_load_model") as load:
                    with self.assertRaises(PretrainedReportError):
                        self.run_config()
                    load.assert_not_called()
                self.assert_no_exports()


if __name__ == "__main__":
    unittest.main()
