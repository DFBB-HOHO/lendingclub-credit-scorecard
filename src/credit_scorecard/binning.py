"""Interpretable WOE binning with train-only monotonic numeric cut points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


MISSING = "__MISSING__"
OTHER = "__OTHER__"


def _pava_blocks(rates: np.ndarray, weights: np.ndarray, increasing: bool) -> tuple[list[dict], float]:
    """Pool adjacent violators and return contiguous monotonic blocks."""
    blocks: list[dict[str, float | int]] = []
    for idx, (rate, weight) in enumerate(zip(rates, weights)):
        blocks.append({"start": idx, "end": idx, "weight": float(weight), "weighted_rate": float(rate * weight)})
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_rate = float(left["weighted_rate"]) / float(left["weight"])
            right_rate = float(right["weighted_rate"]) / float(right["weight"])
            violation = left_rate > right_rate if increasing else left_rate < right_rate
            if not violation:
                break
            merged = {
                "start": int(left["start"]),
                "end": int(right["end"]),
                "weight": float(left["weight"]) + float(right["weight"]),
                "weighted_rate": float(left["weighted_rate"]) + float(right["weighted_rate"]),
            }
            blocks[-2:] = [merged]

    sse = 0.0
    for block in blocks:
        block_rate = float(block["weighted_rate"]) / float(block["weight"])
        start, end = int(block["start"]), int(block["end"])
        sse += float(np.sum(weights[start : end + 1] * (rates[start : end + 1] - block_rate) ** 2))
    return blocks, sse


def _format_edge(value: float) -> str:
    if np.isneginf(value):
        return "-inf"
    if np.isposinf(value):
        return "inf"
    return f"{value:.6g}"


def _woe_table(labels: pd.Series, target: pd.Series, smoothing: float) -> pd.DataFrame:
    tmp = pd.DataFrame({"bin": labels.astype("string"), "target": target.astype(int)})
    grouped = tmp.groupby("bin", dropna=False, observed=False)["target"].agg(["count", "sum"])
    grouped = grouped.rename(columns={"sum": "bad"}).reset_index()
    grouped["good"] = grouped["count"] - grouped["bad"]
    bins = len(grouped)
    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()
    grouped["dist_good"] = (grouped["good"] + smoothing) / (total_good + smoothing * bins)
    grouped["dist_bad"] = (grouped["bad"] + smoothing) / (total_bad + smoothing * bins)
    grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])
    grouped["iv_component"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]
    grouped["bad_rate"] = grouped["bad"] / grouped["count"]
    return grouped


@dataclass
class WOEBinner:
    continuous_features: list[str]
    categorical_features: list[str]
    max_prebins: int = 10
    min_bin_fraction: float = 0.03
    rare_category_fraction: float = 0.01
    smoothing: float = 0.5

    rules_: dict[str, dict[str, Any]] | None = None
    bin_table_: pd.DataFrame | None = None
    iv_: dict[str, float] | None = None

    def _fit_numeric(self, values: pd.Series, target: pd.Series) -> tuple[dict, pd.DataFrame]:
        numeric = pd.to_numeric(values, errors="coerce")
        nonmissing = numeric.notna()
        x = numeric[nonmissing]
        y = target[nonmissing]

        if x.nunique() < 2:
            edges = np.array([-np.inf, np.inf], dtype=float)
            direction = "flat"
        else:
            quantiles = np.linspace(0, 1, self.max_prebins + 1)[1:-1]
            cut_points = np.unique(x.quantile(quantiles).to_numpy(dtype=float))
            edges = np.concatenate(([-np.inf], cut_points, [np.inf]))
            bin_ids = pd.cut(x, bins=edges, labels=False, include_lowest=True, duplicates="drop")
            stats = (
                pd.DataFrame({"bin": bin_ids, "target": y})
                .groupby("bin", observed=False)["target"]
                .agg(["count", "mean"])
                .reset_index()
            )
            rates = stats["mean"].to_numpy(dtype=float)
            weights = stats["count"].to_numpy(dtype=float)
            inc_blocks, inc_sse = _pava_blocks(rates, weights, increasing=True)
            dec_blocks, dec_sse = _pava_blocks(rates, weights, increasing=False)
            if inc_sse <= dec_sse:
                blocks, direction = inc_blocks, "increasing_bad_rate"
            else:
                blocks, direction = dec_blocks, "decreasing_bad_rate"
            interior = [edges[int(block["end"]) + 1] for block in blocks[:-1]]
            edges = np.array([-np.inf, *interior, np.inf], dtype=float)

        labels = pd.Series(MISSING, index=values.index, dtype="string")
        labels.loc[nonmissing] = (
            pd.cut(numeric[nonmissing], bins=edges, labels=False, include_lowest=True)
            .astype("Int64")
            .astype("string")
        )
        table = _woe_table(labels, target, self.smoothing)
        woe_map = dict(zip(table["bin"].astype(str), table["woe"].astype(float)))
        rule = {
            "type": "continuous",
            "edges": edges.tolist(),
            "direction": direction,
            "woe": woe_map,
        }
        return rule, table

    def _fit_categorical(self, values: pd.Series, target: pd.Series) -> tuple[dict, pd.DataFrame]:
        normalised = values.astype("string").fillna(MISSING).replace("", MISSING)
        frequencies = normalised.value_counts(normalize=True, dropna=False)
        rare_values = set(frequencies[frequencies < self.rare_category_fraction].index.astype(str))
        grouped = normalised.map(lambda value: OTHER if str(value) in rare_values else str(value)).astype("string")
        table = _woe_table(grouped, target, self.smoothing)
        woe_map = dict(zip(table["bin"].astype(str), table["woe"].astype(float)))
        rule = {
            "type": "categorical",
            "rare_values": sorted(rare_values),
            "known_values": sorted(normalised.astype(str).unique().tolist()),
            "woe": woe_map,
        }
        return rule, table

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "WOEBinner":
        rules: dict[str, dict[str, Any]] = {}
        tables: list[pd.DataFrame] = []
        iv: dict[str, float] = {}

        for feature in self.continuous_features + self.categorical_features:
            if feature in self.continuous_features:
                rule, table = self._fit_numeric(frame[feature], target)
            else:
                rule, table = self._fit_categorical(frame[feature], target)
            feature_iv = float(table["iv_component"].sum())
            table.insert(0, "feature", feature)
            table.insert(1, "feature_type", rule["type"])
            table["total_iv"] = feature_iv
            rules[feature] = rule
            tables.append(table)
            iv[feature] = feature_iv

        self.rules_ = rules
        self.bin_table_ = pd.concat(tables, ignore_index=True)
        self.iv_ = iv
        return self

    def bin_labels(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.rules_ is None:
            raise RuntimeError("WOEBinner must be fitted before transform.")
        output: dict[str, pd.Series] = {}
        for feature, rule in self.rules_.items():
            if rule["type"] == "continuous":
                numeric = pd.to_numeric(frame[feature], errors="coerce")
                labels = pd.Series(MISSING, index=frame.index, dtype="string")
                mask = numeric.notna()
                labels.loc[mask] = (
                    pd.cut(numeric[mask], bins=np.asarray(rule["edges"], dtype=float), labels=False, include_lowest=True)
                    .astype("Int64")
                    .astype("string")
                )
            else:
                raw = frame[feature].astype("string").fillna(MISSING).replace("", MISSING).astype(str)
                rare = set(rule["rare_values"])
                known = set(rule["known_values"])
                has_other = OTHER in rule["woe"]
                labels = raw.map(
                    lambda value: OTHER
                    if value in rare or (value not in known and has_other)
                    else value
                ).astype("string")
            output[feature] = labels
        return pd.DataFrame(output, index=frame.index)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.rules_ is None:
            raise RuntimeError("WOEBinner must be fitted before transform.")
        labels = self.bin_labels(frame)
        transformed = {}
        for feature, rule in self.rules_.items():
            transformed[feature] = labels[feature].map(rule["woe"]).fillna(0.0).astype(float)
        return pd.DataFrame(transformed, index=frame.index)

    def fit_transform(self, frame: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        return self.fit(frame, target).transform(frame)

    def to_dict(self) -> dict:
        if self.rules_ is None or self.iv_ is None:
            raise RuntimeError("WOEBinner has not been fitted.")

        def serialise(value: Any) -> Any:
            if isinstance(value, float) and np.isinf(value):
                return "inf" if value > 0 else "-inf"
            if isinstance(value, list):
                return [serialise(item) for item in value]
            if isinstance(value, dict):
                return {key: serialise(item) for key, item in value.items()}
            return value

        return {"rules": serialise(self.rules_), "iv": self.iv_}

    def readable_bin_table(self) -> pd.DataFrame:
        if self.bin_table_ is None or self.rules_ is None:
            raise RuntimeError("WOEBinner has not been fitted.")
        table = self.bin_table_.copy()
        descriptions: list[str] = []
        for row in table.itertuples(index=False):
            rule = self.rules_[row.feature]
            if rule["type"] == "continuous" and row.bin != MISSING:
                idx = int(row.bin)
                lower, upper = rule["edges"][idx], rule["edges"][idx + 1]
                descriptions.append(f"[{_format_edge(lower)}, {_format_edge(upper)})")
            else:
                descriptions.append(str(row.bin))
        table.insert(3, "bin_description", descriptions)
        return table

