"""Construct calendar and historical inputs, and the saved preprocessing.

Availability rule: a forecast for the hour starting at t may use the calendar
fields of t and the count/weather of any *completed* hour s < t. Every
historical value is obtained by exact timestamp lookup (never by CSV row
shifting), so an absent source hour yields a missing input.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

CALENDAR_CATEGORICAL = ["hour", "weekday", "month"]
CALENDAR_BINARY = ["holiday", "workingday"]
HIST_NUMERIC = [
    "prev_hour_count", "prev_day_count", "prev_week_count", "prev_24h_mean",
    "prev_hour_temp", "prev_hour_hum", "prev_hour_windspeed",
]
LAG_SPEC = {  # feature -> (source column, hours back)
    "prev_hour_count": ("cnt", 1),
    "prev_day_count": ("cnt", 24),
    "prev_week_count": ("cnt", 168),
    "prev_hour_temp": ("temp", 1),
    "prev_hour_hum": ("hum", 1),
    "prev_hour_windspeed": ("windspeed", 1),
}
READABLE = {
    "prev_hour_count": "Count one hour earlier",
    "prev_day_count": "Count 24 hours earlier",
    "prev_week_count": "Count 168 hours earlier",
    "prev_24h_mean": "Mean of previous 24 counts",
    "prev_hour_temp": "Previous-hour temperature (normalised)",
    "prev_hour_hum": "Previous-hour humidity (normalised)",
    "prev_hour_windspeed": "Previous-hour wind speed (normalised)",
}


def build_raw_features(clean: pd.DataFrame, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """Raw (unimputed) inputs for each target timestamp.

    ``clean`` is the audited frame indexed by timestamp. Targets must be
    observed rows (their calendar fields come from the row; their cnt and
    weather are never read).
    """
    targets = pd.DatetimeIndex(targets)
    out = pd.DataFrame(index=targets)
    out.index.name = "timestamp"
    cal = clean.loc[targets]
    out["hour"] = targets.hour.astype(int)
    out["weekday"] = cal["weekday"].astype(int).values      # dataset convention Sun=0
    out["month"] = cal["mnth"].astype(int).values
    out["holiday"] = cal["holiday"].astype(int).values
    out["workingday"] = cal["workingday"].astype(int).values

    for feat, (col, k) in LAG_SPEC.items():
        src = targets - pd.Timedelta(hours=k)
        vals = clean[col].reindex(src).to_numpy(dtype=float)  # exact lookup; absent -> NaN
        out[feat] = vals
        out[f"{feat}__source"] = src.strftime("%Y-%m-%d %H:%M")

    # previous-24h mean: exactly the 24 counts at t-24 ... t-1, all required.
    m = np.full(len(targets), np.nan)
    cnt = clean["cnt"]
    for i, t in enumerate(targets):
        w = cnt.reindex(pd.date_range(t - pd.Timedelta(hours=24), periods=24, freq="h"))
        if w.notna().all():
            m[i] = float(w.mean())
    out["prev_24h_mean"] = m
    out["prev_24h_mean__source"] = [
        f"{(t - pd.Timedelta(hours=24)):%Y-%m-%d %H:%M} to {(t - pd.Timedelta(hours=1)):%Y-%m-%d %H:%M}"
        for t in targets]

    for f in HIST_NUMERIC:
        out[f"{f}__missing"] = out[f].isna().astype(int)
    return out


def build_raw_features_fast(clean: pd.DataFrame, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """Vectorised equivalent of :func:`build_raw_features` for whole periods.

    The 24-hour mean uses a rolling window over the complete hourly calendar
    (absent hours are NaN), ending at t-1 and requiring all 24 values.
    ``checks/test_features.py`` verifies agreement with the explicit version.
    """
    targets = pd.DatetimeIndex(targets)
    out = build_raw_features(clean, targets[:0])  # empty frame with right columns
    out = pd.DataFrame(index=targets, columns=out.columns)
    out.index.name = "timestamp"
    cal = clean.loc[targets]
    out["hour"] = targets.hour.astype(int)
    out["weekday"] = cal["weekday"].astype(int).values
    out["month"] = cal["mnth"].astype(int).values
    out["holiday"] = cal["holiday"].astype(int).values
    out["workingday"] = cal["workingday"].astype(int).values
    for feat, (col, k) in LAG_SPEC.items():
        src = targets - pd.Timedelta(hours=k)
        out[feat] = clean[col].reindex(src).to_numpy(dtype=float)
        out[f"{feat}__source"] = src.strftime("%Y-%m-%d %H:%M")
    full_idx = pd.date_range(clean.index.min() - pd.Timedelta(hours=24), clean.index.max(), freq="h")
    roll = clean["cnt"].astype(float).reindex(full_idx).rolling(24, min_periods=24).mean()
    out["prev_24h_mean"] = roll.reindex(targets - pd.Timedelta(hours=1)).to_numpy()
    out["prev_24h_mean__source"] = [
        f"{(t - pd.Timedelta(hours=24)):%Y-%m-%d %H:%M} to {(t - pd.Timedelta(hours=1)):%Y-%m-%d %H:%M}"
        for t in targets]
    for f in HIST_NUMERIC:
        out[f] = out[f].astype(float)
        out[f"{f}__missing"] = out[f].isna().astype(int)
    for c in CALENDAR_CATEGORICAL + CALENDAR_BINARY:
        out[c] = out[c].astype(int)
    return out


class Preprocessor:
    """Training-only imputation, one-hot encoding and standardisation.

    variant="full": calendar one-hots + 2 binaries + 7 imputed/scaled numeric
    inputs + 7 missing flags. variant="calendar": calendar inputs only.

    Rules: categories are those seen in training; an unseen category maps to
    an all-zero block. A numeric column with zero training standard deviation
    is centred only (scale 1). Missing flags describe missingness *before*
    replacement and stay 0/1. Feature order is fixed and saved.
    """

    def __init__(self, variant: str = "full"):
        assert variant in ("full", "calendar")
        self.variant = variant
        self.categories: dict[str, list[int]] = {}
        self.medians: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.scales: dict[str, float] = {}
        self.feature_names: list[str] = []

    def fit(self, raw_train: pd.DataFrame) -> "Preprocessor":
        self.categories = {c: sorted(int(v) for v in raw_train[c].unique()) for c in CALENDAR_CATEGORICAL}
        names = [f"{c}={v:02d}" for c in CALENDAR_CATEGORICAL for v in self.categories[c]]
        names += CALENDAR_BINARY
        if self.variant == "full":
            for f in HIST_NUMERIC:
                med = float(raw_train[f].median(skipna=True))
                self.medians[f] = med
                filled = raw_train[f].fillna(med).astype(float)
                self.means[f] = float(filled.mean())
                sd = float(filled.std(ddof=0))
                self.scales[f] = sd if sd > 0 else 1.0
            names += HIST_NUMERIC + [f"{f}__missing" for f in HIST_NUMERIC]
        self.feature_names = names
        return self

    def impute(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy()
        for f, med in self.medians.items():
            out[f] = out[f].astype(float).fillna(med)
        return out

    def transform(self, raw: pd.DataFrame) -> np.ndarray:
        blocks = []
        for c in CALENDAR_CATEGORICAL:
            vals = raw[c].to_numpy().astype(int)
            cats = np.array(self.categories[c])
            blocks.append((vals[:, None] == cats[None, :]).astype(float))
        blocks.append(raw[CALENDAR_BINARY].to_numpy(dtype=float))
        if self.variant == "full":
            imp = self.impute(raw)
            num = np.column_stack([(imp[f].to_numpy(dtype=float) - self.means[f]) / self.scales[f]
                                   for f in HIST_NUMERIC])
            flags = np.column_stack([raw[f].isna().to_numpy().astype(float) for f in HIST_NUMERIC])
            blocks += [num, flags]
        X = np.hstack(blocks)
        assert X.shape[1] == len(self.feature_names)
        return X

    def to_dict(self) -> dict:
        return dict(variant=self.variant, categories=self.categories, medians=self.medians,
                    means=self.means, scales=self.scales, feature_names=self.feature_names)

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        p = cls(d["variant"])
        p.categories = {k: list(v) for k, v in d["categories"].items()}
        p.medians, p.means, p.scales = d["medians"], d["means"], d["scales"]
        p.feature_names = d["feature_names"]
        return p

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "Preprocessor":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))
