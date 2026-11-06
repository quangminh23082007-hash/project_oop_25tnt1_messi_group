"""Verify portable serving and installation without GPUs, training data or service prototypes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from src.contracts import GameItem
from src.contracts.exceptions import DataValidationError, UnknownUserError
from src.models.pretrained import PretrainedRecommender

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_installer', ROOT / 'scripts/install_model.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class PretrainedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='pretrained_test_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        assert self.root.parent == Path(tempfile.gettempdir()).resolve()
        self.bundle = self.root / 'bundle'
        self.bundle.mkdir()
        np.savez_compressed(self.bundle / 'weights.npz', P=np.array([[1., 0], [0, 1.]]),
                            R=np.array([[3., 0], [2, 0], [1, 4], [0, 3]]),
                            popularity=np.array([3., 2, 1, 0]),
                            train_seen=np.array([[0, 0], [0, 1], [1, 2]]))
        info = dict(format_version=3, representation='compact', model_type='BPRMatrixFactorization',
                    config={'factors': 2}, user_ids=['known', 'second'], item_ids=['g0', 'g1', 'g2', 'g3'],
                    catalog={f'g{i}': {'title': f'Game {i}', 'image_url': 'example.png'} for i in range(4)},
                    data_fingerprint='original-training', data_config={'train_fraction': .7},
                    best_epoch=2, history=[], weights_sha256=installer.sha256_file(self.bundle / 'weights.npz'))
        (self.bundle / 'model_info.json').write_text(json.dumps(info), encoding='utf-8')

    def archive(self, corrupt_metadata=False, corrupt_weights=False):
        path = self.root / 'model.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('bundle/bpr/cpu/weights.npz', b'bad' if corrupt_weights else (self.bundle / 'weights.npz').read_bytes())
            info = (self.bundle / 'model_info.json').read_bytes()
            archive.writestr('bundle/bpr/cpu/model_info.json', b'{}' if corrupt_metadata else info)
            archive.writestr('../../escaped.txt', b'must never extract')
        return path

    def test_personalized_scores_history_and_source_metadata(self):
        model = PretrainedRecommender.load(self.bundle)
        rows = model.recommend_games('known', top_k=10)
        self.assertEqual([r.game_id for r in rows], ['g2', 'g3'])
        self.assertEqual([r.predicted_score for r in rows], [1., 0.])
        self.assertTrue(all(not r.is_fallback for r in rows))
        self.assertEqual([r.game_id for r in model.recommend_games('known', extra_seen=['g2'])], ['g3'])
        self.assertEqual(model.recommend_games('known', top_k=1, exclude_played=False)[0].game_id, 'g0')
        self.assertEqual({r.item_id for r in model.get_user_played_games('known')}, {'g0', 'g1'})
        self.assertEqual(model.provenance['data_fingerprint'], 'original-training')
        self.assertEqual(model.get_catalog_metadata('g0')['image_url'], 'example.png')

    def test_global_fallback_is_explicit_and_filters_app_history(self):
        with self.assertRaises(UnknownUserError):
            PretrainedRecommender.load(self.bundle).recommend_games('new-account')
        model = PretrainedRecommender.load(self.bundle, allow_cold_start_fallback=True)
        rows = model.recommend_games('new-account', top_k=2, extra_seen=['g0'])
        self.assertEqual([r.game_id for r in rows], ['g1', 'g2'])
        self.assertTrue(all(r.is_fallback and r.model_name == 'Popularity' for r in rows))

    def test_catalog_filters_use_original_game_ids_and_unknown_price_is_not_free(self):
        def catalog(gid):
            return GameItem(gid, gid, ('Action',), 5. if gid == 'g3' else 50.)
        model = PretrainedRecommender.load(self.bundle, catalog_lookup=catalog)
        self.assertEqual([r.game_id for r in model.recommend_games('known', max_price=10, genre_filter='action')], ['g3'])
        self.assertEqual(PretrainedRecommender.load(self.bundle).recommend_games('known', max_price=100), [])

    def test_integrity_verification_before_loading(self):
        with (self.bundle / 'weights.npz').open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaises(DataValidationError): PretrainedRecommender.load(self.bundle)

    def test_install_verifies_whole_archive_and_keeps_paths_contained(self):
        archive = self.archive()
        destination = self.root / 'installed'
        installer.install(archive, destination, 'bpr', installer.sha256_file(archive))
        model = PretrainedRecommender.load(destination / 'bpr/cpu')
        self.assertEqual(model.recommend_games('known')[0].game_id, 'g2')
        self.assertFalse((self.root / 'escaped.txt').exists())
        with self.assertRaises(FileExistsError): installer.install(archive, destination, 'bpr')

    def test_changed_metadata_fails_archive_digest_before_install(self):
        original = installer.sha256_file(self.archive())
        changed = self.archive(corrupt_metadata=True)
        with self.assertRaisesRegex(ValueError, 'Archive checksum'):
            installer.install(changed, self.root / 'installed', 'bpr', original)
        self.assertFalse((self.root / 'installed').exists())

    def test_bad_internal_weights_do_not_activate_a_model(self):
        archive = self.archive(corrupt_weights=True)
        with self.assertRaisesRegex(ValueError, 'corrupt weights'):
            installer.install(archive, self.root / 'installed', 'bpr', installer.sha256_file(archive))
        self.assertFalse((self.root / 'installed/bpr').exists())


if __name__ == '__main__':
    unittest.main()
