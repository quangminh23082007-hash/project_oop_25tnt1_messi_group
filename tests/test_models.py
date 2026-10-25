"""Model-only tests: validation, gradients, training and scoring."""
import unittest

import numpy as np

from matrix_factorization import BPRMatrixFactorization, bpr_loss_and_gradients
from popularity import PopularityRecommender
from text_bpr import TextBPRMatrixFactorization, text_bpr_loss_and_gradients


class ModelTests(unittest.TestCase):
    def test_popularity_distinct_users_and_refit(self):
        model = PopularityRecommender().fit([(0, 0), (0, 0), (1, 0), (1, 1)], 2, 3)
        np.testing.assert_array_equal(model.score_all(0), [2, 1, 0])
        model.fit([(0, 1)], 2, 3)
        np.testing.assert_array_equal(model.score_all(1), [0, 1, 0])

    def test_invalid_inputs_and_before_fit(self):
        for model in (PopularityRecommender(), BPRMatrixFactorization()):
            with self.assertRaises(RuntimeError):
                model.score(0, 0)
            with self.assertRaises(ValueError):
                model.fit([], 2, 2)
            with self.assertRaises(ValueError):
                model.fit([(0, 2)], 2, 2)
            with self.assertRaises(ValueError):
                model.fit([(0.0, 1.0)], 2, 2)
        model = PopularityRecommender().fit([(0, 0)], 1, 1)
        for u, i in ((-1, 0), (0, -1), (1, 0), (0, 1), (True, 0)):
            with self.assertRaises(ValueError):
                model.score(u, i)

    def test_invalid_config(self):
        for kwargs in ({"factors": 0}, {"epochs": -1}, {"reg": -1},
                       {"learning_rate": 0}, {"learning_rate": float("nan")},
                       {"seed": -1}, {"factors": 2.5}):
            with self.assertRaises(ValueError):
                BPRMatrixFactorization(**kwargs)

    def test_analytical_gradient_matches_finite_difference(self):
        values = [np.array([0.2, -0.4, 0.1]), np.array([-0.3, 0.1, 0.5]), np.array([0.7, -0.2, 0.3])]
        _, analytic = bpr_loss_and_gradients(*values, 0.03)
        eps = 1e-6
        for index in range(3):
            numerical = np.zeros(3)
            for dimension in range(3):
                plus, minus = [v.copy() for v in values], [v.copy() for v in values]
                plus[index][dimension] += eps
                minus[index][dimension] -= eps
                numerical[dimension] = (bpr_loss_and_gradients(*plus, 0.03)[0] - bpr_loss_and_gradients(*minus, 0.03)[0]) / (2 * eps)
            np.testing.assert_allclose(analytic[index], numerical, atol=1e-8)

    def test_small_gradient_step_decreases_objective(self):
        values = [np.array([0.2, 0.3]), np.array([0.6, -0.2]), np.array([0.1, 0.5])]
        before, gradients = bpr_loss_and_gradients(*values, 0.01)
        after = bpr_loss_and_gradients(*(v - 0.001 * g for v, g in zip(values, gradients)), 0.01)[0]
        self.assertLess(after, before)

    def test_extreme_margin_is_numerically_stable(self):
        for sign in (-1, 1):
            loss, grads = bpr_loss_and_gradients(np.array([1000.]), np.array([sign * 1000.]), np.array([0.]), 0.01)
            self.assertTrue(np.isfinite(loss))
            self.assertTrue(all(np.isfinite(g).all() for g in grads))

    def test_no_negative_item_raises(self):
        with self.assertRaisesRegex(ValueError, "No valid BPR triples"):
            BPRMatrixFactorization().fit([(0, 0)], 1, 2, observed=[(0, 0), (0, 1)], verbose=False)

    def test_same_seed_and_best_checkpoint_restoration(self):
        pairs = [(0, 0), (1, 1)]
        first = BPRMatrixFactorization(factors=3, epochs=1, seed=9).fit(pairs, 2, 3, verbose=False)
        sequence = iter([0.9, 0.1])
        selected = BPRMatrixFactorization(factors=3, epochs=5, seed=9, patience=1).fit(
            pairs, 2, 3, validation_score=lambda _: next(sequence), verbose=False)
        self.assertEqual(selected.best_epoch, 1)
        np.testing.assert_array_equal(first.P, selected.P)
        np.testing.assert_array_equal(first.Q, selected.Q)


class TextModelTests(unittest.TestCase):
    def test_four_gradients_and_loss_decrease(self):
        rng = np.random.default_rng(9)
        values = [rng.normal(size=3) for _ in range(3)] + [rng.normal(size=(3, 4))]
        xi, xj = rng.normal(size=(2, 4))
        loss, gradients = text_bpr_loss_and_gradients(*values, xi, xj, .03, .01)
        eps = 1e-6
        for n, value in enumerate(values):
            numerical = np.zeros_like(value)
            for index in np.ndindex(value.shape):
                plus, minus = [v.copy() for v in values], [v.copy() for v in values]
                plus[n][index] += eps
                minus[n][index] -= eps
                numerical[index] = (text_bpr_loss_and_gradients(*plus, xi, xj, .03, .01)[0]
                    - text_bpr_loss_and_gradients(*minus, xi, xj, .03, .01)[0]) / (2 * eps)
            np.testing.assert_allclose(gradients[n], numerical, atol=1e-8)
        updated = [v - .001 * g for v, g in zip(values, gradients)]
        self.assertLess(text_bpr_loss_and_gradients(*updated, xi, xj, .03, .01)[0], loss)

    def test_zero_features_exactly_match_bpr(self):
        pairs = [(0, 0), (1, 1), (2, 2)]
        b = BPRMatrixFactorization(factors=3, epochs=5).fit(pairs, 3, 4, verbose=False)
        t = TextBPRMatrixFactorization(factors=3, epochs=5).fit(pairs, 3, 4, item_features=np.zeros((4, 5)), verbose=False)
        np.testing.assert_array_equal(b.P, t.P)
        np.testing.assert_array_equal(b.Q, t.Q)
        np.testing.assert_array_equal(b.score_all(0), t.score_all(0))

    def test_projection_checkpoint_and_cache(self):
        pairs = [(0, 0), (1, 1)]
        first = TextBPRMatrixFactorization(factors=3, epochs=1).fit(pairs, 2, 3, item_features=np.eye(3), verbose=False)
        seq = iter([.9, .1])
        def val(m):
            m.score_all(0)
            return next(seq)
        chosen = TextBPRMatrixFactorization(factors=3, epochs=5, patience=1).fit(
            pairs, 2, 3, item_features=np.eye(3), validation_score=val, verbose=False)
        for k in ['P', 'Q', 'W']:
            np.testing.assert_array_equal(getattr(first, k), getattr(chosen, k))
        self.assertGreater(np.linalg.norm(chosen.W), 0)
        np.testing.assert_allclose(chosen.score_all(0), [chosen.score(0, i) for i in range(3)], atol=1e-15)


if __name__ == "__main__":
    unittest.main()
