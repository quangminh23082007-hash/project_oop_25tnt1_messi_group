import csv
import gzip
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks import export_all_users as exporter


class AllUsersExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="all_users_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = self.root / "model"
        self.bundle.mkdir()
        weights = self.bundle / "weights.npz"
        np.savez_compressed(
            weights,
            P=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            R=np.array([[3.0, 0.0], [2.0, 0.0], [1.0, 4.0], [0.0, 3.0]]),
            popularity=np.array([3.0, 2.0, 1.0, 0.0]),
            train_seen=np.array([[0, 0], [0, 1], [1, 2], [2, 0], [2, 1], [2, 2], [2, 3]], dtype=np.int64),
        )
        info = {
            "format_version": 3,
            "representation": "compact",
            "model_type": "BPRMatrixFactorization",
            "config": {"factors": 2},
            "user_ids": ["u0", "u1", "u2"],
            "item_ids": ["g0", "g1", "g2", "g3"],
            "catalog": {f"g{i}": {"title": f"TEST_ONLY Game {i}"} for i in range(4)},
            "data_fingerprint": "TEST_ONLY",
            "data_config": {},
            "best_epoch": 1,
            "history": [],
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        }
        (self.bundle / "model_info.json").write_text(json.dumps(info), encoding="utf-8")
        self.output = self.root / "output"

    def run_export(self, **kwargs):
        return exporter.export_all_users(
            self.bundle, self.output, top_k=2, users_per_part=1, **kwargs
        )

    def state(self):
        return json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))

    def rows(self, state):
        rows = []
        for part in state["parts"]:
            with gzip.open(self.output / part["file"], "rt", encoding="utf-8", newline="") as handle:
                rows.extend(csv.DictReader(handle))
        return rows

    def test_complete_export_tracks_every_user_and_empty_results(self):
        self.run_export()
        state = self.state()
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "complete")
        self.assertEqual(summary["processed_users"], 3)
        self.assertEqual(summary["users_without_recommendations"], 1)
        self.assertEqual(summary["recommendation_rows"], 4)
        self.assertTrue(summary["all_mapped_users_exported"])
        self.assertFalse(summary["recomputed_metrics"])
        rows = self.rows(state)
        self.assertEqual([r["game_id"] for r in rows if r["user_id"] == "u0"], ["g2", "g3"])
        self.assertEqual([r["game_id"] for r in rows if r["user_id"] == "u1"], ["g3", "g0"])
        self.assertEqual(len(state["parts"]), 3)

    def test_completed_run_does_not_load_or_duplicate_model(self):
        from src.models.pretrained import PretrainedRecommender
        self.run_export()
        before = (self.output / "manifest.json").read_bytes()
        with patch.object(PretrainedRecommender, "load", side_effect=AssertionError("Unexpected reload")):
            self.run_export()
        self.assertEqual((self.output / "manifest.json").read_bytes(), before)
        self.assertEqual(len(self.rows(self.state())), 4)

    def test_interruption_resumes_after_committed_users(self):
        from src.models.pretrained import PretrainedRecommender
        original = PretrainedRecommender.recommend_games
        def interrupted(model, user_id, **kwargs):
            if user_id == "u1":
                raise KeyboardInterrupt()
            return original(model, user_id, **kwargs)
        with patch.object(PretrainedRecommender, "recommend_games", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_export()
        self.assertEqual(self.state()["next_user_index"], 1)
        self.run_export()
        self.assertEqual(self.state()["next_user_index"], 3)
        self.assertEqual(len(self.rows(self.state())), 4)

    def test_orphan_part_is_recomputed_and_adopted_without_duplicates(self):
        original = exporter._json_write
        def interrupted(path, value):
            if path.name == "manifest.json" and value.get("next_user_index") == 2:
                raise OSError("TEST_ONLY interrupted checkpoint")
            return original(path, value)
        with patch.object(exporter, "_json_write", interrupted):
            with self.assertRaises(OSError):
                self.run_export()
        self.assertEqual(self.state()["next_user_index"], 1)
        self.run_export()
        self.assertEqual(len(self.rows(self.state())), 4)

    def test_changed_committed_part_is_rejected(self):
        self.run_export()
        part = self.output / self.state()["parts"][0]["file"]
        with part.open("ab") as handle:
            handle.write(b"TEST_ONLY corruption")
        with self.assertRaises(ValueError):
            self.run_export()

    def test_changed_settings_are_rejected(self):
        self.run_export(limit_users=1)
        with self.assertRaises(ValueError):
            self.run_export()
        self.assertFalse(json.loads((self.output / "summary.json").read_text(encoding="utf-8"))["all_mapped_users_exported"])

    def test_invalid_requests_fail_before_output(self):
        with self.assertRaises(ValueError):
            exporter.export_all_users(self.bundle, self.output, top_k=0)
        self.assertFalse(self.output.exists())

    def test_manifest_temporary_file_before_first_commit_can_be_recovered(self):
        original = exporter._json_write
        def interrupted(path, value):
            if path.name == "manifest.json" and not path.exists():
                path.with_name(path.name + ".partial").write_text("TEST_ONLY incomplete", encoding="utf-8")
                raise OSError("TEST_ONLY initial interruption")
            return original(path, value)
        with patch.object(exporter, "_json_write", interrupted):
            with self.assertRaises(OSError):
                self.run_export()
        self.run_export()
        self.assertTrue(json.loads((self.output / "summary.json").read_text(encoding="utf-8"))["all_mapped_users_exported"])


if __name__ == "__main__":
    unittest.main()
