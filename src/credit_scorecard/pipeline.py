"""End-to-end model-development pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .binning import WOEBinner
from .data import (
    BASE_NUMERIC,
    NumericWinsorizer,
    basic_clean,
    load_model_data,
    split_summary,
    time_split,
    vintage_summary,
)
from .metrics import (
    approval_strategy_table,
    binary_metrics,
    decile_lift_table,
    feature_psi_table,
    population_stability_index,
)
from .modeling import coefficient_table, recalibrate_intercept, select_features, tune_logistic_regression
from .reporting import plot_iv, plot_roc_ks, plot_score_distribution, plot_vintage, write_markdown_report
from .scoring import ScoreScaler


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def run_pipeline(config_path: str | Path, data_path: str | Path, project_root: str | Path) -> dict:
    root = Path(project_root)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    artifact_dir = root / "artifacts"
    figure_dir = root / "reports" / "figures"
    report_dir = root / "reports"
    for directory in (artifact_dir, figure_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw = load_model_data(str(data_path))
    clean, audit = basic_clean(raw)
    vintage = vintage_summary(clean)
    samples, excluded = time_split(clean, config["split"])
    splits = split_summary(samples, excluded)

    features = config["features"]
    winsorizer = NumericWinsorizer(
        columns=features["continuous"],
        lower_quantile=config["cleaning"]["winsor_lower"],
        upper_quantile=config["cleaning"]["winsor_upper"],
    )
    transformed_samples: dict[str, pd.DataFrame] = {}
    transformed_samples["train"] = winsorizer.fit_transform(samples["train"])
    transformed_samples["validation"] = winsorizer.transform(samples["validation"])
    transformed_samples["oot"] = winsorizer.transform(samples["oot"])

    binner = WOEBinner(
        continuous_features=features["continuous"],
        categorical_features=features["categorical"],
        max_prebins=config["binning"]["max_prebins"],
        min_bin_fraction=config["binning"]["min_bin_fraction"],
        rare_category_fraction=config["binning"]["rare_category_fraction"],
        smoothing=config["binning"]["smoothing"],
    )
    train_woe = binner.fit_transform(transformed_samples["train"], samples["train"]["Default"])
    validation_woe = binner.transform(transformed_samples["validation"])
    oot_woe = binner.transform(transformed_samples["oot"])

    selection = config["selection"]
    selected, feature_decisions = select_features(
        train_woe,
        binner.iv_ or {},
        min_iv=selection["min_iv"],
        max_iv=selection["max_iv"],
        max_abs_correlation=selection["max_abs_correlation"],
        max_features=selection.get("max_features"),
    )
    model_config = config["model"]
    model, tuning = tune_logistic_regression(
        train_woe[selected],
        samples["train"]["Default"],
        validation_woe[selected],
        samples["validation"]["Default"],
        candidate_c=model_config["candidate_c"],
        max_iter=model_config["max_iter"],
        random_state=model_config["random_state"],
    )
    calibration_shift = recalibrate_intercept(
        model,
        validation_woe[selected],
        samples["validation"]["Default"],
    )

    woe_by_sample = {"train": train_woe, "validation": validation_woe, "oot": oot_woe}
    probabilities: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    for name, woe in woe_by_sample.items():
        probability = model.predict_proba(woe[selected])[:, 1]
        probabilities[name] = probability
        metrics[name] = binary_metrics(samples[name]["Default"], probability)

    scaler = ScoreScaler(
        base_score=config["score"]["base_score"],
        pdo=config["score"]["pdo"],
        base_odds_good_to_bad=config["score"]["base_odds_good_to_bad"],
    )
    scores = {name: scaler.probability_to_score(probability) for name, probability in probabilities.items()}
    score_psi, score_psi_bins = population_stability_index(scores["train"], scores["oot"], bins=10)
    metrics["stability"] = {"development_to_oot_score_psi": score_psi}

    train_bins = binner.bin_labels(transformed_samples["train"])[selected]
    oot_bins = binner.bin_labels(transformed_samples["oot"])[selected]
    feature_psi = feature_psi_table(train_bins, oot_bins)

    iv_table = (
        pd.DataFrame([{"feature": key, "iv": value} for key, value in (binner.iv_ or {}).items()])
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )
    coefficients = coefficient_table(model, selected)
    base_points, scorecard = scaler.scorecard_table(
        binner.readable_bin_table(), coefficients, float(model.intercept_[0]), selected
    )

    # Exact reconciliation between probability scaling and points decomposition.
    points_score = np.full(len(train_woe), base_points, dtype=float)
    coefficient_map = coefficients.set_index("feature")["coefficient"].to_dict()
    for feature in selected:
        points_score += -scaler.factor * coefficient_map[feature] * train_woe[feature].to_numpy()
    max_reconciliation_error = float(np.max(np.abs(points_score - scores["train"])))
    if max_reconciliation_error > 1e-6:
        raise AssertionError(f"Score decomposition mismatch: {max_reconciliation_error}")
    metrics["score_scaling"] = {
        "base_score": scaler.base_score,
        "pdo": scaler.pdo,
        "base_odds_good_to_bad": scaler.base_odds_good_to_bad,
        "factor": scaler.factor,
        "offset": scaler.offset,
        "base_points": base_points,
        "max_reconciliation_error": max_reconciliation_error,
        "validation_intercept_calibration_shift": calibration_shift,
    }

    strategy = approval_strategy_table(samples["oot"]["Default"], scores["oot"])
    lift = decile_lift_table(samples["oot"]["Default"], probabilities["oot"])
    score_summary = pd.DataFrame(
        [
            {
                "sample": name,
                "score_mean": float(np.mean(value)),
                "score_std": float(np.std(value)),
                "score_p05": float(np.quantile(value, 0.05)),
                "score_p50": float(np.quantile(value, 0.50)),
                "score_p95": float(np.quantile(value, 0.95)),
            }
            for name, value in scores.items()
        ]
    )

    _write_json(artifact_dir / "cleaning_audit.json", audit)
    _write_json(artifact_dir / "metrics.json", metrics)
    _write_json(artifact_dir / "binning_rules.json", binner.to_dict())
    _write_json(artifact_dir / "winsorization_rules.json", winsorizer.to_dict())
    splits.to_csv(artifact_dir / "sample_split.csv", index=False)
    vintage.to_csv(artifact_dir / "vintage_summary.csv", index=False)
    iv_table.to_csv(artifact_dir / "iv_summary.csv", index=False)
    feature_decisions.to_csv(artifact_dir / "feature_decisions.csv", index=False)
    binner.readable_bin_table().to_csv(artifact_dir / "binning_table.csv", index=False)
    coefficients.to_csv(artifact_dir / "coefficients.csv", index=False)
    tuning.to_csv(artifact_dir / "hyperparameter_tuning.csv", index=False)
    scorecard.to_csv(artifact_dir / "scorecard_points.csv", index=False)
    score_psi_bins.to_csv(artifact_dir / "score_psi_bins.csv", index=False)
    feature_psi.to_csv(artifact_dir / "feature_psi.csv", index=False)
    strategy.to_csv(artifact_dir / "approval_strategy_scenarios.csv", index=False)
    lift.to_csv(artifact_dir / "oot_decile_lift.csv", index=False)
    score_summary.to_csv(artifact_dir / "score_summary.csv", index=False)
    joblib.dump(
        {
            "config": config,
            "winsorizer": winsorizer,
            "binner": binner,
            "selected_features": selected,
            "model": model,
            "score_scaler": scaler,
        },
        artifact_dir / "model_package.joblib",
    )

    plot_roc_ks(samples["oot"]["Default"], probabilities["oot"], figure_dir / "oot_roc_ks.png")
    plot_score_distribution(samples["oot"]["Default"], scores["oot"], figure_dir / "oot_score_distribution.png")
    plot_vintage(vintage, figure_dir / "vintage_diagnostics.png")
    plot_iv(iv_table, figure_dir / "feature_iv.png")
    write_markdown_report(
        report_dir / "model_development_report.md",
        metrics,
        splits,
        iv_table,
        selected,
        score_psi,
        feature_psi,
        base_points,
    )

    return {
        "selected_features": selected,
        "metrics": metrics,
        "score_psi": score_psi,
        "base_points": base_points,
        "output_directory": str(artifact_dir),
    }
