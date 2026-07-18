"""Convert logistic default probabilities into a conventional points scale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ScoreScaler:
    base_score: float = 600
    pdo: float = 50
    base_odds_good_to_bad: float = 20

    @property
    def factor(self) -> float:
        return float(self.pdo / np.log(2))

    @property
    def offset(self) -> float:
        return float(self.base_score - self.factor * np.log(self.base_odds_good_to_bad))

    def probability_to_score(self, probability_bad: np.ndarray) -> np.ndarray:
        probability = np.clip(np.asarray(probability_bad, dtype=float), 1e-8, 1 - 1e-8)
        odds_good_to_bad = (1 - probability) / probability
        return self.offset + self.factor * np.log(odds_good_to_bad)

    def score_to_probability(self, score: np.ndarray) -> np.ndarray:
        log_odds_good_to_bad = (np.asarray(score, dtype=float) - self.offset) / self.factor
        return 1 / (1 + np.exp(log_odds_good_to_bad))

    def scorecard_table(
        self,
        readable_bin_table: pd.DataFrame,
        coefficients: pd.DataFrame,
        intercept: float,
        selected_features: list[str],
    ) -> tuple[float, pd.DataFrame]:
        coefficient_map = coefficients.set_index("feature")["coefficient"].to_dict()
        table = readable_bin_table[readable_bin_table["feature"].isin(selected_features)].copy()
        table["coefficient"] = table["feature"].map(coefficient_map)
        table["bin_points"] = -self.factor * table["coefficient"] * table["woe"]
        base_points = float(self.offset - self.factor * intercept)
        columns = [
            "feature",
            "feature_type",
            "bin_description",
            "count",
            "bad",
            "bad_rate",
            "woe",
            "total_iv",
            "coefficient",
            "bin_points",
        ]
        return base_points, table[columns]

