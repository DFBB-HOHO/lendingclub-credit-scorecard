"""Risk-model discrimination, lift, stability and threshold metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve


def binary_metrics(target: pd.Series | np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    fpr, tpr, thresholds = roc_curve(y, p)
    differences = tpr - fpr
    best = int(np.argmax(differences))
    return {
        "observations": int(len(y)),
        "bad_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)),
        "gini": float(2 * roc_auc_score(y, p) - 1),
        "ks": float(differences[best]),
        "ks_probability_threshold": float(thresholds[best]),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_predicted_pd": float(p.mean()),
    }


def population_stability_index(
    expected: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> tuple[float, pd.DataFrame]:
    expected_values = np.asarray(expected, dtype=float)
    actual_values = np.asarray(actual, dtype=float)
    finite_expected = expected_values[np.isfinite(expected_values)]
    cut_points = np.unique(np.quantile(finite_expected, np.linspace(0, 1, bins + 1)))
    if len(cut_points) < 3:
        cut_points = np.array([finite_expected.min(), finite_expected.max()])
    edges = np.concatenate(([-np.inf], cut_points[1:-1], [np.inf]))
    expected_counts, _ = np.histogram(expected_values, bins=edges)
    actual_counts, _ = np.histogram(actual_values, bins=edges)
    expected_share = np.clip(expected_counts / max(expected_counts.sum(), 1), epsilon, None)
    actual_share = np.clip(actual_counts / max(actual_counts.sum(), 1), epsilon, None)
    components = (actual_share - expected_share) * np.log(actual_share / expected_share)
    table = pd.DataFrame(
        {
            "lower": edges[:-1],
            "upper": edges[1:],
            "expected_share": expected_share,
            "actual_share": actual_share,
            "psi_component": components,
        }
    )
    return float(components.sum()), table


def categorical_psi(expected_labels: pd.Series, actual_labels: pd.Series, epsilon: float = 1e-6) -> float:
    categories = sorted(set(expected_labels.astype(str)) | set(actual_labels.astype(str)))
    expected_share = expected_labels.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0.0)
    actual_share = actual_labels.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0.0)
    expected_share = expected_share.clip(lower=epsilon)
    actual_share = actual_share.clip(lower=epsilon)
    return float(((actual_share - expected_share) * np.log(actual_share / expected_share)).sum())


def feature_psi_table(expected_bins: pd.DataFrame, actual_bins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in expected_bins.columns:
        value = categorical_psi(expected_bins[feature], actual_bins[feature])
        status = "stable" if value < 0.10 else "watch" if value < 0.25 else "unstable"
        rows.append({"feature": feature, "psi": value, "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def decile_lift_table(target: pd.Series, probability: np.ndarray, groups: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"target": np.asarray(target, dtype=int), "probability": probability})
    ranks = frame["probability"].rank(method="first", ascending=False)
    frame["risk_decile"] = pd.qcut(ranks, q=groups, labels=range(1, groups + 1))
    overall_bad_rate = frame["target"].mean()
    out = (
        frame.groupby("risk_decile", observed=False)
        .agg(observations=("target", "size"), defaults=("target", "sum"), bad_rate=("target", "mean"), mean_pd=("probability", "mean"))
        .reset_index()
    )
    out["lift"] = out["bad_rate"] / overall_bad_rate
    out["cumulative_default_capture"] = out["defaults"].cumsum() / out["defaults"].sum()
    return out


def approval_strategy_table(target: pd.Series, score: np.ndarray) -> pd.DataFrame:
    """Scenario table on already-granted loans; not a causal approval simulation."""
    y = np.asarray(target, dtype=int)
    scores = np.asarray(score, dtype=float)
    rows = []
    for target_approval in np.arange(0.1, 1.0, 0.1):
        cutoff = float(np.quantile(scores, 1 - target_approval))
        approved = scores >= cutoff
        rows.append(
            {
                "target_approval_rate": float(target_approval),
                "score_cutoff": cutoff,
                "observed_approval_rate": float(approved.mean()),
                "approved_bad_rate": float(y[approved].mean()),
                "default_capture_in_approved": float(y[approved].sum() / max(y.sum(), 1)),
                "approved_observations": int(approved.sum()),
            }
        )
    return pd.DataFrame(rows)

