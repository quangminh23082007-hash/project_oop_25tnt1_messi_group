"""Models-only v1 tests. Fixtures do not import or replace the team's Data layer."""
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import unittest
from unittest.mock import patch
import numpy as np

from src.contracts.dto import Interaction, InteractionRecord, ItemFeatures, ModelConfig, ReviewRecord
from src.contracts.exceptions import (ConfigurationError, DataValidationError, EmptyDatasetError,
                                      InvalidParameterError, ModelNotFittedError, TrainingError)
from src.contracts.interfaces import BaseRecommender
from src.models import (BPRMatrixFactorization, MatrixFactorizationRecommender,
                        PopularityRecommender, TextBPRMatrixFactorization)
from src.models._array.matrix_factorization import BPRMatrixFactorization as ArrayBPR
from src.models.text_features import build_item_features


@dataclass(frozen=True)
class FixtureMapper:
    ids: tuple[str, ...]
    @property
    def size(self): return len(self.ids)
    def to_raw(self, index): return self.ids[index]
    def contains_raw(self, raw_id): return raw_id in self.ids
    def to_idx(self, raw_id): return self.ids.index(raw_id)


@dataclass(frozen=True)
class FixtureDataset:
    interactions: tuple = (Interaction(0, 0, 1), Interaction(0, 1, 0),
                           Interaction(1, 1, 1), Interaction(1, 2, 1), Interaction(2, 3, 0))
    reviews: tuple = (ReviewRecord(0, 0, 1, 0, "fun"), ReviewRecord(0, 1, 0, 0, "bad"),
                      ReviewRecord(1, 1, 1, 0, "bon"))
    split_name: str = "train"
    user_mapper: FixtureMapper = FixtureMapper(("u0", "u1", "u2"))
    item_mapper: FixtureMapper = FixtureMapper(("g0", "g1", "g2", "g3"))
    @property
    def num_users(self): return self.user_mapper.size
    @property
    def num_items(self): return self.item_mapper.size
    @property
    def num_interactions(self): return len(self.interactions)
    @property
    def mapping_fingerprint(self):
        return sha256(repr((self.user_mapper.ids, self.item_mapper.ids)).encode()).hexdigest()
    @property
    def fingerprint(self):
        return sha256(repr((self.split_name, self.mapping_fingerprint, self.interactions)).encode()).hexdigest()
    @property
    def review_fingerprint(self): return sha256(repr(self.reviews).encode()).hexdigest()
    def get_user_seen_items(self, user_id):
        return frozenset(r.item_id for r in self.interactions if r.user_id == user_id)
    def get_user_positive_items(self, user_id):
        return frozenset(r.item_id for r in self.interactions if r.user_id == user_id and r.rating == 1)
    def iter_reviews(self): return iter(self.reviews)


class TeamModelTests(unittest.TestCase):
    def setUp(self):
        self.data = FixtureDataset()
        self.config = ModelConfig(n_factors=3, learning_rate=.02, n_epochs=2, random_seed=7)

    def features(self):
        return ItemFeatures(np.eye(4), self.data.item_mapper.ids, self.data.mapping_fingerprint,
                            self.data.fingerprint, self.data.review_fingerprint, "test-encoder@fixed-v1")

    def test_shared_contract_and_equivalent_bpr_scores(self):
        model = BPRMatrixFactorization(self.config)
        self.assertIsInstance(model, BaseRecommender)
        self.assertIs(MatrixFactorizationRecommender, BPRMatrixFactorization)
        with self.assertRaises(ModelNotFittedError): model.score_items(0, [0])
        self.assertIs(model.fit(self.data), model)
        reference = ArrayBPR(factors=3, learning_rate=.02, reg=.01, epochs=2, seed=7)
        reference.fit(np.array([[0, 0], [1, 1], [1, 2]]), 3, 4,
                      observed=np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 3]]), verbose=False)
        for user in range(3):
            np.testing.assert_array_equal(model.score_items(user, [3, 1, 1, 0]), reference.score_all(user)[[3, 1, 1, 0]])
        self.assertEqual((model.num_users, model.num_items), (3, 4))
        self.assertEqual(model.mapping_fingerprint, self.data.mapping_fingerprint)
        self.assertEqual(model.train_fingerprint, self.data.fingerprint)
        self.assertTrue(model.is_fitted)

    def test_recommend_filters_positive_and_negative_history_with_deterministic_ties(self):
        model = PopularityRecommender().fit(self.data)
        self.assertEqual([r.item_idx for r in model.recommend(0, self.data)], [2, 3])
        self.assertEqual([r.item_idx for r in model.recommend(0, self.data, exclude_seen=False)], [0, 1, 2, 3])
        equivalent_copy = replace(self.data)
        self.assertEqual(model.recommend(0, equivalent_copy), model.recommend(0, self.data))

    def test_recommend_rejects_changed_train_and_mapping(self):
        model = PopularityRecommender().fit(self.data)
        for bad in (replace(self.data, split_name="validation"),
                    replace(self.data, interactions=self.data.interactions[:-1]),
                    replace(self.data, item_mapper=FixtureMapper(tuple(reversed(self.data.item_mapper.ids))))):
            with self.subTest(bad=bad), self.assertRaises(DataValidationError): model.recommend(0, bad)
        for k in (0, -1, 1.5, True):
            with self.assertRaises(InvalidParameterError): model.recommend(0, self.data, k)
        with self.assertRaises(InvalidParameterError): model.recommend(0, self.data, exclude_seen=1)

    def test_popularity_global_scores_and_refit(self):
        model = PopularityRecommender().fit(self.data)
        np.testing.assert_equal(model.score_global([3, 2, 0]), [0, 1, 1])
        model.fit(replace(self.data, interactions=(Interaction(0, 2, 1),)))
        np.testing.assert_equal(model.score_global([0, 1, 2, 3]), [0, 0, 1, 0])
        self.assertEqual(model.score_global([]), [])
        self.assertEqual(model.config(), {})
        with self.assertRaises(ConfigurationError): model.fit(self.data, validation_score=lambda m: 1)
        self.assertFalse(model.is_fitted)
        self.assertFalse(model.core_model._is_fitted)

    def test_bpr_never_samples_any_reviewed_item_as_negative(self):
        model = BPRMatrixFactorization(self.config, epochs=12)
        original = model.core_model._sgd_step
        sampled = []
        def step(u, i, j):
            sampled.append((u, i, j))
            self.assertNotIn(j, self.data.get_user_seen_items(u))
            return original(u, i, j)
        with patch.object(model.core_model, "_sgd_step", side_effect=step): model.fit(self.data)
        self.assertEqual(len(sampled), 36)

    def test_empty_positive_or_no_triples_rejected_without_hanging(self):
        for data in (replace(self.data, interactions=()),
                     replace(self.data, interactions=(Interaction(0, 0, 0),)),
                     replace(self.data, interactions=tuple(Interaction(0, i, int(i == 0)) for i in range(4)))):
            model = BPRMatrixFactorization(self.config)
            with self.assertRaises(EmptyDatasetError): model.fit(data)
            self.assertFalse(model.is_fitted)

    def test_train_role_mapped_types_bounds_and_dedup_are_required(self):
        for data in (replace(self.data, split_name="test"),
                     replace(self.data, interactions=(InteractionRecord("u0", "g0", 1),)),
                     replace(self.data, interactions=(Interaction(3, 0, 1),)),
                     replace(self.data, interactions=self.data.interactions * 2)):
            model = BPRMatrixFactorization(self.config).fit(self.data)
            with self.assertRaises(DataValidationError): model.fit(data)
            self.assertFalse(model.is_fitted)
            self.assertFalse(model.core_model._is_fitted)
            self.assertIsNone(model.train_fingerprint)
            with self.assertRaises(ModelNotFittedError): model.score_items(0, [0])

    def test_callbacks_receive_public_candidate_and_restore_best_for_both_models(self):
        for cls in (BPRMatrixFactorization, TextBPRMatrixFactorization):
            with self.subTest(model=cls.__name__):
                model = cls(self.config, epochs=9, patience=2)
                snapshots = []
                def validate(candidate):
                    self.assertIs(candidate, model)
                    self.assertTrue(candidate.is_fitted)
                    self.assertEqual(candidate.train_fingerprint, self.data.fingerprint)
                    self.assertEqual(candidate.mapping_fingerprint, self.data.mapping_fingerprint)
                    snapshots.append(candidate.score_items(0, [0, 1, 2, 3]))
                    candidate.recommend(0, self.data)
                    return 1.0 / len(snapshots)
                options = {"item_features": self.features()} if cls is TextBPRMatrixFactorization else {}
                model.fit(self.data, validation_score=validate, **options)
                self.assertEqual(len(snapshots), 3)
                self.assertEqual(model.best_epoch, 1)
                np.testing.assert_array_equal(model.score_items(0, [0, 1, 2, 3]), snapshots[0])
                self.assertFalse(np.array_equal(snapshots[0], snapshots[1]))

    def test_callback_failure_invalidates_adapter_and_core(self):
        def failing(candidate): raise RuntimeError("evaluator failed")
        for callback in (failing, lambda m: np.nan, lambda m: np.inf, lambda m: [1.0],
                         lambda m: True, lambda m: "0.5"):
            model = BPRMatrixFactorization(self.config).fit(self.data)
            with self.assertRaises(TrainingError): model.fit(self.data, validation_score=callback)
            self.assertFalse(model.is_fitted)
            self.assertFalse(model.core_model._is_fitted)
            with self.assertRaises(ModelNotFittedError): model.recommend(0, self.data)

    def test_backend_failure_is_a_shared_training_error(self):
        model = BPRMatrixFactorization(self.config)
        with patch.object(model.core_model, "_sgd_step", side_effect=FloatingPointError("diverged")):
            with self.assertRaises(TrainingError): model.fit(self.data)
        self.assertFalse(model.is_fitted)
        self.assertFalse(model.core_model._is_fitted)

    def test_text_features_require_all_provenance_and_mapping_order(self):
        good = self.features()
        for features in (None, np.eye(4), replace(good, mapping_fingerprint="other"),
                         replace(good, train_fingerprint="other"), replace(good, text_fingerprint="other"),
                         replace(good, item_ids=tuple(reversed(good.item_ids)))):
            model = TextBPRMatrixFactorization(self.config)
            with self.assertRaises(ConfigurationError): model.fit(self.data, item_features=features)
            self.assertFalse(model.is_fitted)
        model.fit(self.data, item_features=good)
        self.assertTrue(model.is_fitted)
        with self.assertRaises(ConfigurationError): model.fit(self.data)
        self.assertFalse(model.is_fitted)  # No reuse of stale features on refit.
        with self.assertRaises(ConfigurationError): BPRMatrixFactorization().fit(self.data, item_features=good)
        no_text = replace(self.data, reviews=())
        empty_provenance = replace(good, text_fingerprint=no_text.review_fingerprint)
        with self.assertRaises(EmptyDatasetError):
            model.fit(no_text, item_features=empty_provenance)

    def test_invalid_feature_vectors(self):
        for values in (np.eye(3), np.eye(4) * 2, np.full((4, 2), np.nan), np.ones((4, 2), dtype=complex)):
            with self.assertRaises(DataValidationError): replace(self.features(), values=values)
        source = np.eye(4)
        features = replace(self.features(), values=source)
        source[:] = 0
        np.testing.assert_array_equal(features.values, np.eye(4))
        self.assertFalse(features.values.flags.writeable)

    def test_score_input_order_empty_and_finiteness(self):
        model = PopularityRecommender().fit(self.data)
        for user, items in ((True, [0]), (-1, [0]), (3, [0]), (0, [-1]), (0, [4]),
                            (0, [1.5]), (0, [1, True]), (0, [[1]]), (0, ["1"]), (0, [np.nan])):
            with self.subTest(user=user, items=items), self.assertRaises(InvalidParameterError):
                model.score_items(user, items)
        self.assertEqual(model.score_items(0, []), [])
        for bad in (np.array([1, 2]), np.array([np.nan, 0, 0, 0]), np.ones((4, 1))):
            with patch.object(model.core_model, "score_all", return_value=bad):
                with self.assertRaises(DataValidationError): model.score_items(0, [0, 1])

    def test_config_uses_common_names_and_actual_overrides(self):
        model = TextBPRMatrixFactorization(self.config, factors=5, epochs=4, seed=9, patience=3, text_reg=.07)
        result = json.loads(json.dumps(model.config(), allow_nan=False))
        self.assertEqual(result, dict(n_factors=5, learning_rate=.02, reg=.01, n_epochs=4,
                                      random_seed=9, patience=3, text_reg=.07))
        self.assertNotIn("text_reg", BPRMatrixFactorization().config())
        for options in (dict(factors=0), dict(factors=True), dict(learning_rate=np.nan),
                        dict(learning_rate=True), dict(seed=-1), dict(patience=1.5)):
            with self.assertRaises(ConfigurationError): BPRMatrixFactorization(**options)
        with self.assertRaises(ConfigurationError): model.fit(self.data, validation_score=42)
        for cls in (BPRMatrixFactorization, TextBPRMatrixFactorization, PopularityRecommender):
            with self.assertRaises(ConfigurationError): cls(unsupported_option=1)

    def test_binary_labels_and_source_text_preserved(self):
        value = InteractionRecord("  User A ", " Game ", 0)
        self.assertEqual((value.user_id, value.item_id), ("User A", "Game"))

        review = ReviewRecord(0, 0, 0.0, 0, " Text\nunchanged ")
        self.assertEqual(review.text, " Text\nunchanged ")
        
        for rating in (4.5, np.nan, "1", True):
            with self.assertRaises(DataValidationError): Interaction(0, 1, rating)

    def test_train_feature_builder_and_report(self):
        encoded = []
        def encode(texts):
            encoded.extend(texts)
            return [[0., 1.] if t == "bad" else [1., 0.] for t in texts]
        features, report = build_item_features(self.data, encode, encoder_name="fixture", encoder_revision="v1", batch_size=2)
        self.assertEqual(encoded, ["fun", "bad", "bon"])
        np.testing.assert_allclose(features.values, [[1, 0], [2**-.5, 2**-.5], [0, 0], [0, 0]])
        self.assertEqual((report["reviews_used"], report["items_with_text"], report["zero_vectors"]), (3, 2, 2))
        self.assertEqual(features.text_fingerprint, self.data.review_fingerprint)
        self.assertTrue(TextBPRMatrixFactorization(self.config).fit(self.data, item_features=features).is_fitted)

    def test_feature_builder_rejects_other_split_wrong_record_duplicate_and_missing_text(self):
        def encode(texts): return np.ones((len(texts), 2))
        for data in (replace(self.data, split_name="validation"),
                     replace(self.data, reviews=(ReviewRecord(0, 2, 1, 0, "held out"),)),
                     replace(self.data, reviews=(ReviewRecord(0, 1, 1, 0, "wrong label"),)),
                     replace(self.data, reviews=(ReviewRecord(0, 0, 1, 1, "wrong time"),)),
                     replace(self.data, reviews=self.data.reviews * 2)):
            with self.assertRaises(DataValidationError):
                build_item_features(data, encode, encoder_name="fixture", encoder_revision="v1")
        with self.assertRaises(EmptyDatasetError):
            build_item_features(replace(self.data, reviews=()), encode, encoder_name="fixture", encoder_revision="v1")
        with self.assertRaises(ConfigurationError):
            build_item_features(self.data, encode, encoder_name="fixture", encoder_revision="main")
        for encode_bad in (lambda t: np.ones((len(t) + 1, 2)), lambda t: np.zeros((len(t), 2))):
            with self.assertRaises(DataValidationError):
                build_item_features(self.data, encode_bad, encoder_name="fixture", encoder_revision="v1")


if __name__ == "__main__":
    unittest.main()
