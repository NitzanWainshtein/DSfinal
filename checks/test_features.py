"""Prediction boundary: inputs use only the target calendar and completed earlier hours."""
import numpy as np
import pandas as pd

import features as F
from conftest import make_fixture

T = pd.Timestamp("2012-03-08 10:00")   # 178 hours after fixture start: all lags exist


def feats(df, t=T):
    return F.build_raw_features(df, pd.DatetimeIndex([t])).iloc[0]


def num(row):
    return row[F.HIST_NUMERIC].astype(float)


def test_target_hour_outcome_and_weather_do_not_change_inputs():
    df = make_fixture(np.arange(200) + 10)
    base = num(feats(df))
    df2 = df.copy()
    df2.loc[T, ["cnt", "temp", "hum", "windspeed"]] = [99999, 0.99, 0.01, 0.9]
    pd.testing.assert_series_equal(base, num(feats(df2)))


def test_later_hours_do_not_change_inputs():
    df = make_fixture(np.arange(200) + 10)
    base = num(feats(df))
    df2 = df.copy()
    later = df2.index > T
    df2.loc[later, "cnt"] = 0
    df2.loc[later, "temp"] = 0.0
    pd.testing.assert_series_equal(base, num(feats(df2)))


def test_earlier_eligible_count_changes_its_input():
    df = make_fixture(np.arange(200) + 10)
    df2 = df.copy()
    df2.loc[T - pd.Timedelta(hours=1), "cnt"] = 5000
    a, b = feats(df), feats(df2)
    assert b["prev_hour_count"] == 5000 and a["prev_hour_count"] != 5000
    assert b["prev_24h_mean"] != a["prev_24h_mean"]
    assert b["prev_day_count"] == a["prev_day_count"]


def test_missing_previous_timestamp_is_missing_not_shifted():
    df = make_fixture(np.arange(200) + 10, drop=[T - pd.Timedelta(hours=1)])
    r = feats(df)
    assert np.isnan(r["prev_hour_count"]) and r["prev_hour_count__missing"] == 1
    assert np.isnan(r["prev_hour_temp"]) and r["prev_hour_temp__missing"] == 1
    # the row immediately above in the file is t-2; it must not be used
    assert r["prev_hour_count__source"] == (T - pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")


def test_real_missing_hour_2011_01_02_05(clean):
    r = F.build_raw_features(clean, pd.DatetimeIndex([pd.Timestamp("2011-01-02 06:00")])).iloc[0]
    assert np.isnan(r["prev_hour_count"]) and r["prev_hour_count__missing"] == 1


def test_observed_zero_is_kept():
    counts = np.arange(200) + 10
    counts[177] = 0                          # T - 1h
    r = feats(make_fixture(counts))
    assert r["prev_hour_count"] == 0 and r["prev_hour_count__missing"] == 0


def test_24h_mean_requires_complete_window():
    counts = np.zeros(200, int)
    counts[178 - 24:178] = np.arange(1, 25)  # t-24 ... t-1 = 1 ... 24
    assert feats(make_fixture(counts))["prev_24h_mean"] == 12.5
    r = feats(make_fixture(counts, drop=[T - pd.Timedelta(hours=5)]))
    assert np.isnan(r["prev_24h_mean"]) and r["prev_24h_mean__missing"] == 1


def test_fast_builder_matches_explicit(clean):
    targets = clean.index[::97]
    a = F.build_raw_features(clean, targets)
    b = F.build_raw_features_fast(clean, targets)
    for c in F.HIST_NUMERIC:
        np.testing.assert_allclose(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True)
    assert (a[[f"{c}__missing" for c in F.HIST_NUMERIC]].to_numpy()
            == b[[f"{c}__missing" for c in F.HIST_NUMERIC]].to_numpy()).all()


def test_brief_worked_example_inputs(clean):
    r = F.build_raw_features(clean, pd.DatetimeIndex([pd.Timestamp("2012-08-01 08:00")])).iloc[0]
    assert (r["hour"], r["weekday"], r["month"], r["holiday"], r["workingday"]) == (8, 3, 8, 0, 1)
    assert (r["prev_hour_count"], r["prev_day_count"], r["prev_week_count"]) == (487, 760, 788)
    assert abs(r["prev_24h_mean"] - 300.4167) < 1e-4
    assert (r["prev_hour_temp"], r["prev_hour_hum"], r["prev_hour_windspeed"]) == (0.64, 0.83, 0.1343)


def test_missing_inputs_use_training_median_and_flag_before_replacement():
    df = make_fixture(np.arange(200) + 10, drop=[T - pd.Timedelta(hours=1)])
    train = F.build_raw_features(df, df.index[170:180])
    pre = F.Preprocessor("full").fit(train)
    raw = F.build_raw_features(df, pd.DatetimeIndex([T]))
    imp = pre.impute(raw).iloc[0]
    assert imp["prev_hour_count"] == pre.medians["prev_hour_count"]
    X = pre.transform(raw)[0]
    names = pre.feature_names
    assert X[names.index("prev_hour_count__missing")] == 1.0
    assert X[names.index("prev_day_count__missing")] == 0.0


def test_unseen_category_maps_to_zero_block():
    df = make_fixture(np.arange(200) + 10)
    train = F.build_raw_features(df, df.index[170:175])     # hours 2..6 only
    pre = F.Preprocessor("full").fit(train)
    X = pre.transform(F.build_raw_features(df, pd.DatetimeIndex([T])))[0]  # hour 10 unseen
    hour_cols = [i for i, n in enumerate(pre.feature_names) if n.startswith("hour=")]
    assert X[hour_cols].sum() == 0
