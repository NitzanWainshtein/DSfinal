"""Learned values and choices are fixed by training/validation; saved objects reproduce results."""
import json
import os

import numpy as np
import pandas as pd
import pytest

import data as D
import features as F
import models as M
import train_evaluate as TE


@pytest.fixture(scope="module")
def runs(clean):
    a = TE.fit_select(clean, verbose=False)
    altered = clean.copy()
    test = altered["period"] == "test"
    rng = np.random.default_rng(1)
    altered.loc[test, "cnt"] = rng.integers(0, 5000, test.sum())
    altered.loc[test, "temp"] = rng.random(test.sum())
    b = TE.fit_select(altered, verbose=False)
    return a, b


def test_changing_test_values_leaves_preprocessing_unchanged(runs):
    a, b = runs
    assert a["pre"].to_dict() == b["pre"].to_dict()
    assert a["pre_cal"].to_dict() == b["pre_cal"].to_dict()


def test_changing_test_values_leaves_coefficients_and_default_unchanged(runs):
    a, b = runs
    assert a["default"] == b["default"]
    assert {k: v["setting"] for k, v in a["selected"].items()} == {k: v["setting"] for k, v in b["selected"].items()}
    for key, m in a["fitted"].items():
        if hasattr(m, "coef_"):
            np.testing.assert_array_equal(m.coef_, b["fitted"][key].coef_)
        if isinstance(m, M.BayesianLinear):
            np.testing.assert_array_equal(m.m, b["fitted"][key].m)
    pd.testing.assert_frame_equal(a["vr"], b["vr"])


def test_saved_models_reproduce_saved_predictions(root, clean):
    pre = F.Preprocessor.load(os.path.join(root, "models", "preprocessor_full.json"))
    pred = pd.read_csv(os.path.join(root, "outputs", "predictions.csv"))
    cfg = json.load(open(os.path.join(root, "config.json")))
    sample = clean.index[clean["period"] == "test"][::53]
    raw = F.build_raw_features(clean, sample)
    X = pre.transform(raw)
    for method in M.METHODS:
        mdl = M.load_model(os.path.join(root, "models", f"{method}.pkl"))
        d = mdl.predict_dist(raw, X)
        saved = pred[(pred.model == method) & (pred.period == "test")].set_index("timestamp")
        s = saved.loc[sample.strftime("%Y-%m-%d %H:%M")]
        np.testing.assert_allclose(d["point"], s["point_forecast"], atol=cfg["app_forecast_tolerance"])
        if method in M.HAS_INTERVALS:
            np.testing.assert_allclose(d["lo80"], s["lo80"], atol=1e-5)
            np.testing.assert_allclose(d["hi95"], s["hi95"], atol=1e-5)


def test_saved_files_consistent(root):
    pred = pd.read_csv(os.path.join(root, "outputs", "predictions.csv"))
    # every model scored on the same test hours, no absent hour scored
    hrs = pred[pred.period == "test"].groupby("model")["timestamp"].apply(frozenset)
    assert len(set(hrs)) == 1
    audit = pd.read_csv(os.path.join(root, "outputs", "data_audit.csv")).set_index("period")
    assert len(next(iter(hrs))) == audit.loc["test", "observed_hours"]
    # intervals present exactly for Poisson / Bayesian
    for m, g in pred.groupby("model"):
        assert g["lo80"].notna().all() == (m in M.HAS_INTERVALS)
    met = pd.read_csv(os.path.join(root, "outputs", "metrics.csv"))
    ps = met[met.table == "point_scores"].set_index("model")
    for m in M.METHODS:
        g = pred[(pred.model == m) & (pred.period == "test")]
        assert abs(ps.loc[m, "mae"] - M.mae(g.actual, g.point_forecast)) < 1e-6


def test_audit_finds_brief_missing_hour(clean):
    aud = D.audit(D.load_raw(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hour.csv")))
    assert pd.Timestamp("2011-01-02 05:00") in aud.missing_hours
    assert len(aud.clean) == 17379


def test_duplicate_and_invalid_rows_are_excluded():
    raw = pd.DataFrame({"instant": [1, 2, 3, 4], "dteday": ["2011-01-01"] * 4, "hr": [0, 0, 1, 2],
                        "cnt": [5, 7, -1, 3], "weekday": 6, "mnth": 1, "holiday": 0, "workingday": 0,
                        "temp": 0.2, "hum": 0.5, "windspeed": 0.1})
    raw["timestamp"] = pd.to_datetime(raw.dteday) + pd.to_timedelta(raw.hr, unit="h")
    aud = D.audit(raw)
    assert list(aud.clean["cnt"]) == [5, 3]           # kept smallest instant; dropped negative
    assert set(aud.excluded["reason"]) == {"duplicate timestamp (kept smallest instant)", "invalid target count"}
