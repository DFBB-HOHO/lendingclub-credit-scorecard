from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from credit_scorecard.binning import WOEBinner
from credit_scorecard.metrics import population_stability_index
from credit_scorecard.modeling import recalibrate_intercept
from credit_scorecard.scoring import ScoreScaler
from sklearn.linear_model import LogisticRegression


class ScorecardTests(unittest.TestCase):
    def test_probability_score_round_trip(self) -> None:
        scaler = ScoreScaler(base_score=600, pdo=50, base_odds_good_to_bad=20)
        probability = np.array([0.01, 0.05, 0.20, 0.50, 0.90])
        reconstructed = scaler.score_to_probability(scaler.probability_to_score(probability))
        np.testing.assert_allclose(reconstructed, probability, atol=1e-10)

    def test_lower_risk_has_higher_score(self) -> None:
        scaler = ScoreScaler()
        scores = scaler.probability_to_score(np.array([0.05, 0.20, 0.40]))
        self.assertTrue(np.all(np.diff(scores) < 0))

    def test_numeric_bins_are_monotonic(self) -> None:
        rng = np.random.default_rng(42)
        x = np.linspace(0, 100, 3000)
        probability = 1 / (1 + np.exp(-(x - 50) / 10))
        y = pd.Series(rng.binomial(1, probability))
        frame = pd.DataFrame({"x": x})
        binner = WOEBinner(["x"], [], max_prebins=10).fit(frame, y)
        table = binner.readable_bin_table()
        rates = table.loc[table["bin"] != "__MISSING__", "bad_rate"].to_numpy()
        self.assertTrue(np.all(np.diff(rates) >= -1e-12) or np.all(np.diff(rates) <= 1e-12))

    def test_unseen_category_maps_without_nan(self) -> None:
        frame = pd.DataFrame({"purpose": ["A"] * 90 + ["B"] * 10})
        target = pd.Series([0] * 70 + [1] * 20 + [0] * 5 + [1] * 5)
        binner = WOEBinner([], ["purpose"], rare_category_fraction=0.15).fit(frame, target)
        transformed = binner.transform(pd.DataFrame({"purpose": ["A", "C", None]}))
        self.assertFalse(transformed.isna().any().any())

    def test_psi_identical_population_is_zero(self) -> None:
        values = np.arange(1000)
        psi, _ = population_stability_index(values, values)
        self.assertAlmostEqual(psi, 0.0, places=12)

    def test_intercept_recalibration_matches_portfolio_rate(self) -> None:
        x = pd.DataFrame({"x": np.linspace(-2, 2, 200)})
        y_train = pd.Series((x["x"] > 0.8).astype(int))
        model = LogisticRegression().fit(x, y_train)
        validation_y = pd.Series(([0] * 120) + ([1] * 80))
        before_order = np.argsort(model.predict_proba(x)[:, 1])
        recalibrate_intercept(model, x, validation_y)
        probability = model.predict_proba(x)[:, 1]
        self.assertAlmostEqual(float(probability.mean()), 0.4, places=8)
        np.testing.assert_array_equal(before_order, np.argsort(probability))


if __name__ == "__main__":
    unittest.main()
