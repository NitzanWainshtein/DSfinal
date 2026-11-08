"""Read the UCI hourly bike-sharing file, build the time index and audit it.

Rules (documented in README / report):
* timestamp = dteday + hr hours (local system time, no time zone).
* Duplicate timestamps: keep the row with the smallest ``instant`` and record
  the others as excluded (the original archive has none; the check still runs).
* Invalid targets (missing, non-integer or negative ``cnt``): excluded and
  recorded. An excluded/absent hour is never given a manufactured count.
* Missing hours are found by comparing the observed timestamps with a complete
  hourly calendar over the covered date range.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Chronological periods (inclusive dates), page 5 of the brief.
PERIODS = {
    "train": ("2011-01-01", "2012-04-30"),
    "validation": ("2012-05-01", "2012-07-31"),
    "test": ("2012-08-01", "2012-12-31"),
}
PERIOD_ORDER = ["train", "validation", "test"]

FIELD_DEFINITIONS = {
    "timestamp": "Start of the hour (dteday + hr). Row 08:00 covers 08:00-08:59.",
    "cnt": "Target: total recorded rentals across the system during the hour.",
    "hr": "Hour of day, 0-23 (calendar input, known in advance).",
    "weekday": "Day of week, dataset convention Sunday=0 ... Saturday=6.",
    "mnth": "Month, 1-12 (calendar input).",
    "holiday": "1 if the day is a public holiday, else 0.",
    "workingday": "1 if neither weekend nor holiday, else 0.",
    "temp": "Observed temperature in the hour, normalised (dataset units: t/41 C).",
    "hum": "Observed relative humidity in the hour, normalised (divided by 100).",
    "windspeed": "Observed wind speed in the hour, normalised (divided by 67).",
    "casual / registered": "Components of cnt. EXCLUDED: they reveal the outcome.",
    "instant": "Row index. EXCLUDED: carries no forecasting information.",
    "season, yr, weathersit, atemp": "Not among the required predictors; not used.",
}


@dataclass
class AuditResult:
    clean: pd.DataFrame                      # observed, valid rows indexed by timestamp
    missing_hours: pd.DatetimeIndex          # calendar hours with no row
    excluded: pd.DataFrame                   # excluded rows with reason
    audit_table: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    return df


def assign_period(ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    ts = pd.DatetimeIndex(ts)
    out = np.full(len(ts), "", dtype=object)
    for name, (start, end) in PERIODS.items():
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end) + pd.Timedelta(hours=23)
        out[(ts >= lo) & (ts <= hi)] = name
    out[out == ""] = "outside"
    return out


def audit(raw: pd.DataFrame) -> AuditResult:
    df = raw.sort_values(["timestamp", "instant"]).reset_index(drop=True)
    excluded = []

    dup_mask = df.duplicated("timestamp", keep="first")
    if dup_mask.any():
        e = df.loc[dup_mask].copy()
        e["reason"] = "duplicate timestamp (kept smallest instant)"
        excluded.append(e)
    df = df.loc[~dup_mask]

    cnt = pd.to_numeric(df["cnt"], errors="coerce")
    bad = cnt.isna() | (cnt < 0) | (cnt != np.floor(cnt))
    if bad.any():
        e = df.loc[bad].copy()
        e["reason"] = "invalid target count"
        excluded.append(e)
    df = df.loc[~bad].copy()
    df["cnt"] = df["cnt"].astype(int)

    clean = df.set_index("timestamp").sort_index()
    clean["period"] = assign_period(clean.index)

    full = pd.date_range(raw["timestamp"].min().normalize(),
                         raw["timestamp"].max().normalize() + pd.Timedelta(hours=23), freq="h")
    missing = full.difference(clean.index)
    excl = pd.concat(excluded) if excluded else pd.DataFrame(columns=list(raw.columns) + ["reason"])

    rows = []
    full_period = assign_period(full)
    miss_period = assign_period(missing)
    excl_period = assign_period(excl["timestamp"]) if len(excl) else np.array([], dtype=object)
    for p in PERIOD_ORDER + ["all"]:
        sel = (lambda arr: np.ones(len(arr), bool)) if p == "all" else (lambda arr, p=p: arr == p)
        n_expected = int(sel(full_period).sum())
        n_obs = int(sel(clean["period"].values).sum())
        n_missing = int(sel(miss_period).sum())
        n_dup = int(((excl_period == p) if p != "all" else np.ones(len(excl), bool))[
            (excl["reason"].str.startswith("duplicate")).values].sum()) if len(excl) else 0
        n_bad = int(((excl_period == p) if p != "all" else np.ones(len(excl), bool))[
            (excl["reason"] == "invalid target count").values].sum()) if len(excl) else 0
        zero = int((clean.loc[sel(clean["period"].values), "cnt"] == 0).sum())
        rows.append(dict(period=p, expected_hours=n_expected, observed_hours=n_obs,
                         missing_hours=n_missing, duplicate_rows_excluded=n_dup,
                         invalid_target_rows_excluded=n_bad, observed_zero_count_hours=zero,
                         first_timestamp=str(PERIODS[p][0]) if p in PERIODS else str(full[0]),
                         last_timestamp=(str(PERIODS[p][1]) + " 23:00") if p in PERIODS else str(full[-1])))
    table = pd.DataFrame(rows)
    table["raw_rows_in_file"] = len(raw)
    return AuditResult(clean=clean, missing_hours=missing, excluded=excl, audit_table=table)


def missing_hours_by_day(missing: pd.DatetimeIndex) -> pd.DataFrame:
    if len(missing) == 0:
        return pd.DataFrame(columns=["date", "missing_hours", "hours"])
    s = pd.Series(missing.hour, index=missing.normalize())
    g = s.groupby(level=0).agg(lambda x: ",".join(f"{h:02d}" for h in x))
    c = s.groupby(level=0).size()
    return pd.DataFrame({"date": g.index.date, "missing_hours": c.values, "hours": g.values})
