"""IV/correlation feature selection and logistic-regression tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


def select_features(
    woe_frame: pd.DataFrame,
    iv: dict[str, float],
    min_iv: float = 0.02,
    max_iv: float = 0.5,
    max_abs_correlation: float = 0.7,
    max_features: int | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Filter weak/suspicious IV, then prune redundant WOE features."""
    decisions: list[dict] = []
    eligible: list[str] = []
    for feature, value in sorted(iv.items(), key=lambda item: item[1], reverse=True):
        if value < min_iv:
            decisions.append({"feature": feature, "iv": value, "decision": "drop", "reason": "IV below threshold"})
        elif value > max_iv:
            decisions.append({"feature": feature, "iv": value, "decision": "drop", "reason": "IV above leakage-review threshold"})
        else:
            eligible.append(feature)

    selected: list[str] = []
    correlation = woe_frame[eligible].corr().abs() if eligible else pd.DataFrame()
    for feature in eligible:
        if max_features is not None and len(selected) >= max_features:
            decisions.append(
                {"feature": feature, "iv": iv[feature], "decision": "drop", "reason": f"outside top {max_features} after redundancy pruning"}
            )
            continue
        conflict = next(
            (
                kept
                for kept in selected
                if np.isfinite(correlation.loc[feature, kept])
                and correlation.loc[feature, kept] > max_abs_correlation
            ),
            None,
        )
        if conflict is None:
            selected.append(feature)
            decisions.append({"feature": feature, "iv": iv[feature], "decision": "keep", "reason": "passed IV and correlation filters"})
        else:
            decisions.append(
                {
                    "feature": feature,
                    "iv": iv[feature],
                    "decision": "drop",
                    "reason": f"|correlation| > {max_abs_correlation:.2f} with {conflict}",
                }
            )

    if not selected:
        raise ValueError("Feature selection removed every candidate.")
    return selected, pd.DataFrame(decisions).sort_values(["decision", "iv"], ascending=[False, False])


def tune_logistic_regression(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    validation_x: pd.DataFrame,
    validation_y: pd.Series,
    candidate_c: list[float],
    max_iter: int = 1000,
    random_state: int = 42,
) -> tuple[LogisticRegression, pd.DataFrame]:
    """Choose regularisation strength on validation log loss, not OOT results."""
    rows = []
    models: dict[float, LogisticRegression] = {}
    for c_value in candidate_c:
        model = LogisticRegression(
            C=float(c_value),
            solver="lbfgs",
            max_iter=max_iter,
            random_state=random_state,
        )
        model.fit(train_x, train_y)
        probability = model.predict_proba(validation_x)[:, 1]
        rows.append(
            {
                "C": float(c_value),
                "validation_log_loss": float(log_loss(validation_y, probability, labels=[0, 1])),
                "validation_auc": float(roc_auc_score(validation_y, probability)),
                "iterations": int(model.n_iter_[0]),
            }
        )
        models[float(c_value)] = model

    tuning = pd.DataFrame(rows).sort_values(["validation_log_loss", "C"], ascending=[True, True])
    best_c = float(tuning.iloc[0]["C"])
    return models[best_c], tuning


def recalibrate_intercept(
    model: LogisticRegression,
    validation_x: pd.DataFrame,
    validation_y: pd.Series,
    tolerance: float = 1e-10,
) -> float:
    """Shift only the intercept so validation mean PD matches observed bad rate.

    A constant logit shift preserves rank ordering, AUC, KS and coefficient
    relationships while correcting portfolio-level prior-probability drift.
    """
    logits = model.decision_function(validation_x)
    target_rate = float(validation_y.mean())
    lower, upper = -10.0, 10.0
    for _ in range(100):
        midpoint = (lower + upper) / 2
        mean_probability = float(np.mean(1 / (1 + np.exp(-(logits + midpoint)))))
        if mean_probability < target_rate:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower < tolerance:
            break
    shift = float((lower + upper) / 2)
    model.intercept_ = model.intercept_ + shift
    return shift


def coefficient_table(model: LogisticRegression, features: list[str]) -> pd.DataFrame:
    coefficients = pd.DataFrame({"feature": features, "coefficient": model.coef_[0]})
    coefficients["odds_multiplier_per_1_woe"] = np.exp(coefficients["coefficient"])
    return coefficients.sort_values("coefficient")
