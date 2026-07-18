"""Plots and a compact Markdown model-development report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


COLORS = {"navy": "#18324A", "teal": "#2A7F8E", "gold": "#C89B3C", "red": "#B8574F", "grey": "#6D7680"}


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    def cell(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, pd.Timestamp):
            value = value.strftime("%Y-%m-%d")
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [cell(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def plot_roc_ks(target: pd.Series, probability: np.ndarray, path: Path) -> None:
    fpr, tpr, _ = roc_curve(target, probability)
    ks_index = int(np.argmax(tpr - fpr))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(fpr, tpr, color=COLORS["teal"], linewidth=2.2, label="ROC curve")
    ax.plot([0, 1], [0, 1], color=COLORS["grey"], linestyle="--", linewidth=1)
    ax.vlines(fpr[ks_index], fpr[ks_index], tpr[ks_index], color=COLORS["red"], linestyle="--", label=f"KS = {tpr[ks_index]-fpr[ks_index]:.3f}")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="OOT discrimination")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_score_distribution(target: pd.Series, score: np.ndarray, path: Path) -> None:
    y = np.asarray(target, dtype=int)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    bins = np.linspace(np.nanpercentile(score, 0.5), np.nanpercentile(score, 99.5), 35)
    ax.hist(score[y == 0], bins=bins, density=True, alpha=0.55, color=COLORS["teal"], label="Fully paid")
    ax.hist(score[y == 1], bins=bins, density=True, alpha=0.55, color=COLORS["red"], label="Default")
    ax.set(xlabel="Score (higher = lower risk)", ylabel="Density", title="OOT score distribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_vintage(vintage: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.4, 5.0))
    ax1.bar(vintage["issue_year"], vintage["observations"], color=COLORS["navy"], alpha=0.75)
    ax1.set(xlabel="Issue year", ylabel="Observed final-status loans", title="Vintage volume and observed bad rate")
    ax2 = ax1.twinx()
    ax2.plot(vintage["issue_year"], vintage["bad_rate"], color=COLORS["gold"], marker="o", linewidth=2)
    ax2.set_ylabel("Observed bad rate")
    ax2.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_iv(iv_table: pd.DataFrame, path: Path) -> None:
    ordered = iv_table.sort_values("iv")
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.barh(ordered["feature"], ordered["iv"], color=COLORS["teal"])
    ax.axvline(0.02, color=COLORS["grey"], linestyle="--", linewidth=1, label="IV = 0.02")
    ax.set(xlabel="Information Value", ylabel="", title="Development-sample feature strength")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown_report(
    path: Path,
    metrics: dict,
    split_table: pd.DataFrame,
    iv_table: pd.DataFrame,
    selected_features: list[str],
    score_psi: float,
    feature_psi: pd.DataFrame,
    base_points: float,
) -> None:
    oot = metrics["oot"]
    validation = metrics["validation"]
    psi_status = "stable" if score_psi < 0.10 else "watch" if score_psi < 0.25 else "unstable"
    splits = split_table.copy()
    splits["bad_rate"] = splits["bad_rate"].map(lambda value: f"{value:.2%}" if pd.notna(value) else "—")
    iv_display = iv_table.copy()
    iv_display["iv"] = iv_display["iv"].map(lambda value: f"{value:.4f}")
    feature_psi_display = feature_psi.copy()
    feature_psi_display["psi"] = feature_psi_display["psi"].map(lambda value: f"{value:.4f}")
    content = f"""# LendingClub application scorecard model report

## Executive result

This project develops a leakage-aware application scorecard on LendingClub loans. The champion model uses monotonic WOE binning and logistic regression, with 2014–2015 as the development window, 2016 held out for model selection and 2017 reserved as an out-of-time (OOT) test. On OOT data it achieved **AUC {oot['auc']:.3f}** and **KS {oot['ks']:.3f}**. The development-to-OOT score PSI is **{score_psi:.3f} ({psi_status})**.

The score is scaled to 600 points at 20:1 good-to-bad odds with PDO 50. The model base points are {base_points:.2f}; a higher total score means lower predicted default risk.

## Why this is not a generic classroom scorecard

- Only application-time variables enter the model. LendingClub-assigned grade, sub-grade and interest rate are excluded as prior underwriting outputs.
- Random splitting is avoided. The 2016 validation and 2017 OOT samples reproduce the temporal drift faced by a live risk model.
- 2007–2013 legacy vintages remain in diagnostics but are excluded from fitting because fine-grained bureau-field coverage and platform policy were not comparable with later vintages.
- A validation-period intercept recalibration corrects portfolio-level PD drift while preserving rank ordering, coefficients, AUC and KS.
- 2018 observations are excluded after a vintage check reveals a sharply falling observed bad rate, consistent with incomplete outcome maturation/right censoring.
- Geography and free text are excluded because their apparent lift does not justify proxy-discrimination, instability and governance risk in a compact application scorecard.
- The report separates risk ranking from policy. Approval-rate scenarios are descriptive counterfactuals on already-granted loans, not claims about a true rejected-applicant population.

## Sample design

    {_markdown_table(splits)}

## Feature screening

Selected WOE features: **{', '.join(selected_features)}**.

    {_markdown_table(iv_display)}

## Performance

| Sample | AUC | KS | Bad rate | Mean predicted PD | Brier score |
|---|---:|---:|---:|---:|---:|
| Development | {metrics['train']['auc']:.3f} | {metrics['train']['ks']:.3f} | {metrics['train']['bad_rate']:.2%} | {metrics['train']['mean_predicted_pd']:.2%} | {metrics['train']['brier_score']:.4f} |
| Validation (2016) | {validation['auc']:.3f} | {validation['ks']:.3f} | {validation['bad_rate']:.2%} | {validation['mean_predicted_pd']:.2%} | {validation['brier_score']:.4f} |
| OOT (2017) | {oot['auc']:.3f} | {oot['ks']:.3f} | {oot['bad_rate']:.2%} | {oot['mean_predicted_pd']:.2%} | {oot['brier_score']:.4f} |

## Stability monitoring

    {_markdown_table(feature_psi_display)}

## Limitations and governance

The dataset contains outcomes only for loans LendingClub granted, so the model estimates default ordering inside the historical approved population. It cannot by itself learn risk for rejected applicants or prove the profitability of a new approval cutoff. Reject inference, application fraud labels, bureau freshness, income verification, LGD/EAD, operating costs and fairness testing would be required before production use. The public US P2P sample also should not be presented as a direct model for a Chinese digital bank; its value is the reproducible modelling and governance workflow.

## Reproduction

Run `python scripts/download_data.py`, then `python scripts/train_scorecard.py`. Generated artefacts include binning rules, IV decisions, coefficients, scorecard points, performance metrics, PSI tables, decile lift and approval-strategy scenarios.
"""
    path.write_text(content, encoding="utf-8")
