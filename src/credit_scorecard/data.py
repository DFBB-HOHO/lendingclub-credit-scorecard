"""Data loading, validation, leakage-safe cleaning and time splitting."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd


RAW_NUMERIC = [
    "annual_inc",
    "dti",
    "loan_amnt",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "collections_12_mths_ex_med",
    "mths_since_last_major_derog",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "mort_acc",
    "mths_since_recent_inq",
    "num_accts_ever_120_pd",
    "num_actv_rev_tl",
    "num_tl_90g_dpd_24m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "pub_rec_bankruptcies",
    "total_bal_ex_mort",
    "total_bc_limit",
]
DERIVED_NUMERIC = ["fico_n", "loan_to_income", "credit_history_months"]
BASE_NUMERIC = RAW_NUMERIC + DERIVED_NUMERIC
BASE_CATEGORICAL = ["term", "emp_length", "home_ownership", "purpose", "application_type"]
SOURCE_USECOLS = [
    "id",
    "issue_d",
    "loan_status",
    "fico_range_low",
    "fico_range_high",
    "earliest_cr_line",
    *RAW_NUMERIC,
    *BASE_CATEGORICAL,
]


def load_model_data(path: str) -> pd.DataFrame:
    """Load raw data in chunks, retaining final outcomes and application-time fields."""
    final_statuses = {"Fully Paid", "Charged Off", "Default"}
    status_counts: Counter = Counter()
    raw_rows = 0
    parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(path, usecols=SOURCE_USECOLS, chunksize=200_000, low_memory=False):
        raw_rows += len(chunk)
        status_counts.update(chunk["loan_status"].fillna("__MISSING__").astype(str))
        chunk = chunk[chunk["loan_status"].isin(final_statuses)].copy()
        if chunk.empty:
            continue
        chunk["Default"] = chunk["loan_status"].isin({"Charged Off", "Default"}).astype("int8")
        chunk["issue_d"] = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        earliest = pd.to_datetime(chunk.pop("earliest_cr_line"), format="%b-%Y", errors="coerce")
        for column in RAW_NUMERIC + ["fico_range_low", "fico_range_high"]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("float32")
        chunk["fico_n"] = (chunk.pop("fico_range_low") + chunk.pop("fico_range_high")) / 2
        chunk["loan_to_income"] = chunk["loan_amnt"] / chunk["annual_inc"].replace(0, np.nan)
        chunk["credit_history_months"] = (
            (chunk["issue_d"].dt.year - earliest.dt.year) * 12
            + (chunk["issue_d"].dt.month - earliest.dt.month)
        ).astype("float32")
        chunk = chunk.drop(columns=["loan_status"])
        parts.append(chunk[["id", "issue_d", *BASE_NUMERIC, *BASE_CATEGORICAL, "Default"]])

    frame = pd.concat(parts, ignore_index=True)
    frame.attrs["load_audit"] = {
        "raw_rows": int(raw_rows),
        "final_status_rows": int(len(frame)),
        "status_counts": dict(status_counts),
    }
    return frame


def basic_clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply deterministic validity rules that do not learn from future data."""
    df = frame.copy()
    audit: dict = dict(frame.attrs.get("load_audit", {}))
    audit["rows_loaded"] = int(len(df))
    audit["duplicate_ids"] = int(df.duplicated("id").sum())
    audit["invalid_dates"] = int(df["issue_d"].isna().sum())
    audit["invalid_targets"] = int((~df["Default"].isin([0, 1])).sum())

    df = df.drop_duplicates("id", keep="first")
    df = df[df["issue_d"].notna() & df["Default"].isin([0, 1])].copy()

    invalid_rules = {
        "annual_inc": (df["annual_inc"] <= 0),
        "dti": (df["dti"] < 0),
        "loan_amnt": (df["loan_amnt"] <= 0),
        "fico_n": ((df["fico_n"] < 300) | (df["fico_n"] > 850)),
    }
    for column, mask in invalid_rules.items():
        audit[f"invalid_{column}"] = int(mask.fillna(False).sum())
        df.loc[mask, column] = np.nan

    for column in BASE_CATEGORICAL:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA})
            .fillna("__MISSING__")
        )

    audit["rows_after_basic_clean"] = int(len(df))
    audit["overall_bad_rate"] = float(df["Default"].mean())
    return df, audit


def vintage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise observed outcomes by issue year to expose label censoring."""
    out = (
        frame.assign(issue_year=frame["issue_d"].dt.year)
        .groupby("issue_year", as_index=False)
        .agg(observations=("Default", "size"), defaults=("Default", "sum"), bad_rate=("Default", "mean"))
    )
    return out


def time_split(frame: pd.DataFrame, split_config: dict) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Create development, validation and out-of-time samples by issue month."""
    train_start = pd.Timestamp(split_config["train_start"])
    train_end = pd.Timestamp(split_config["train_end"])
    validation_start = pd.Timestamp(split_config["validation_start"])
    validation_end = pd.Timestamp(split_config["validation_end"])
    oot_start = pd.Timestamp(split_config["oot_start"])
    oot_end = pd.Timestamp(split_config["oot_end"])

    samples = {
        "train": frame[frame["issue_d"].between(train_start, train_end, inclusive="both")].copy(),
        "validation": frame[
            frame["issue_d"].between(validation_start, validation_end, inclusive="both")
        ].copy(),
        "oot": frame[frame["issue_d"].between(oot_start, oot_end, inclusive="both")].copy(),
    }
    excluded = {
        "excluded_legacy_vintages": frame[frame["issue_d"] < train_start].copy(),
        "excluded_right_censored": frame[frame["issue_d"] > oot_end].copy(),
    }

    for name, sample in samples.items():
        if sample.empty:
            raise ValueError(f"Time split '{name}' is empty; check date configuration.")
    return samples, excluded


def split_summary(samples: dict[str, pd.DataFrame], excluded: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, sample in {**samples, **excluded}.items():
        rows.append(
            {
                "sample": name,
                "start_month": sample["issue_d"].min(),
                "end_month": sample["issue_d"].max(),
                "observations": int(len(sample)),
                "defaults": int(sample["Default"].sum()),
                "bad_rate": float(sample["Default"].mean()) if len(sample) else np.nan,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class NumericWinsorizer:
    """Quantile caps learned on development data and reused unchanged OOT."""

    columns: Iterable[str]
    lower_quantile: float = 0.005
    upper_quantile: float = 0.995
    bounds_: dict[str, tuple[float, float]] | None = None

    def fit(self, frame: pd.DataFrame) -> "NumericWinsorizer":
        bounds: dict[str, tuple[float, float]] = {}
        for column in self.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            lower = float(values.quantile(self.lower_quantile))
            upper = float(values.quantile(self.upper_quantile))
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                lower, upper = float(values.min()), float(values.max())
            bounds[column] = (lower, upper)
        self.bounds_ = bounds
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.bounds_ is None:
            raise RuntimeError("NumericWinsorizer must be fitted before transform.")
        out = frame.copy()
        for column, (lower, upper) in self.bounds_.items():
            out[column] = pd.to_numeric(out[column], errors="coerce").clip(lower, upper)
        return out

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def to_dict(self) -> dict:
        if self.bounds_ is None:
            raise RuntimeError("NumericWinsorizer has not been fitted.")
        return {
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "bounds": {key: [low, high] for key, (low, high) in self.bounds_.items()},
        }
