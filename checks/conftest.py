import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data as D  # noqa: E402


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def clean():
    return D.audit(D.load_raw(os.path.join(ROOT, "data", "hour.csv"))).clean


def make_fixture(counts, start="2012-03-01 00:00", drop=()):
    """Small timestamped frame shaped like the audited data (one row per hour)."""
    ts = pd.date_range(start, periods=len(counts), freq="h")
    df = pd.DataFrame({"cnt": counts, "temp": 0.5, "hum": 0.6, "windspeed": 0.1,
                       "weekday": ts.dayofweek.map(lambda d: (d + 1) % 7), "mnth": ts.month,
                       "holiday": 0, "workingday": 1}, index=ts)
    df.index.name = "timestamp"
    df["period"] = D.assign_period(df.index)
    return df.drop(index=[pd.Timestamp(t) for t in drop])
