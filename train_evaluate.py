"""Fit, select, freeze and replay all forecasts; write every saved result.

Sequence (no step reads a later period's outcomes):
 1. audit hour.csv                                   -> outputs/data_audit.csv
 2. build inputs for every observed target hour      (exact timestamp lookup)
 3. fit preprocessing + all candidates on TRAINING   -> models/
 4. score candidates on VALIDATION, select per method, choose default
                                                    -> outputs/validation_results.csv, config.json
 5. freeze; replay VALIDATION and TEST forecasts     -> outputs/predictions.csv
 6. test scores and diagnostics                      -> outputs/metrics.csv and companions

Usage:  python train_evaluate.py [--data data/hour.csv]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
import scipy

import data as D
import features as F
import models as M

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
MOD = os.path.join(ROOT, "models")

GRIDS = {
    "ridge": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "lasso": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
    "poisson": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "bayes": [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0],    # prior precision alpha; sigma2 estimated
    "ridge_calendar": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
}
LASSO_ZERO_TOL = 1e-8
TIE_ORDER = M.METHODS          # deterministic tie rule: earlier (simpler) method wins
APP_TOLERANCE = 1e-6           # app forecast vs saved prediction, absolute rentals


def setting_key(method, s):
    if method == "baseline":
        return "lookup"
    if method == "bayes":
        return f"alpha={s['alpha_prior_precision']:g}"
    return f"alpha={s['alpha']:g}"


def fit_select(clean: pd.DataFrame, verbose: bool = True) -> dict:
    """Steps 2-4: inputs, training-only preprocessing, candidate fits, validation selection.

    Reads test-period rows only to build their (unused here) inputs; no test outcome enters
    any fitted value or choice. checks/test_frozen.py verifies this.
    """
    # 2. inputs ------------------------------------------------------------
    idx = {p: clean.index[clean["period"] == p] for p in D.PERIOD_ORDER}
    rawf = {p: F.build_raw_features_fast(clean, idx[p]) for p in D.PERIOD_ORDER}
    y = {p: clean.loc[idx[p], "cnt"].to_numpy(dtype=float) for p in D.PERIOD_ORDER}

    # 3. preprocessing (training only) ---------------------------------------
    pre = F.Preprocessor("full").fit(rawf["train"])
    pre_cal = F.Preprocessor("calendar").fit(rawf["train"])
    X = {p: pre.transform(rawf[p]) for p in D.PERIOD_ORDER}
    Xc = {p: pre_cal.transform(rawf[p]) for p in D.PERIOD_ORDER}

    # 4. candidates on validation --------------------------------------------
    val_rows, fitted, coef_rows = [], {}, []

    def score(method, setting, model, Xv, experiment="principal"):
        d = model.predict_dist(rawf["validation"], Xv)
        row = dict(experiment=experiment, method=method, setting=setting_key(method, setting),
                   setting_json=json.dumps(setting, default=float), n_hours=len(y["validation"]),
                   validation_mae=M.mae(y["validation"], d["point"]),
                   validation_rmse=M.rmse(y["validation"], d["point"]))
        return row

    b = M.SeasonalBaseline().fit(rawf["train"], y["train"])
    val_rows.append(score("baseline", {}, b, X["validation"]))
    fitted[("baseline", "lookup")] = b

    for method in ["ridge", "lasso", "poisson", "bayes"]:
        for a in GRIDS[method]:
            s = {"alpha_prior_precision": a} if method == "bayes" else {"alpha": a}
            m = M.make_model(method, s).fit(X["train"], y["train"])
            r = score(method, s, m, X["validation"])
            if method == "lasso":
                nz = int(np.sum(np.abs(m.coef_) > LASSO_ZERO_TOL))
                r["n_nonzero_coef"] = nz
                r["n_coef_total"] = len(m.coef_)
                for fname, c in zip(pre.feature_names, m.coef_):
                    coef_rows.append(dict(alpha=a, feature=fname, coefficient=float(c),
                                          is_zero=bool(abs(c) <= LASSO_ZERO_TOL)))
                coef_rows.append(dict(alpha=a, feature="(intercept)", coefficient=m.intercept_, is_zero=False))
            if method == "bayes":
                r["sigma2"] = m.sigma2
                r["setting_json"] = json.dumps(m.settings(), default=float)
            val_rows.append(r)
            fitted[(method, setting_key(method, s))] = m
            if verbose:
                print(f"{method:8s} {setting_key(method, s):14s} val MAE={r['validation_mae']:.3f}")

    # history experiment: calendar-only ridge vs full-input ridge (both tuned)
    for a in GRIDS["ridge_calendar"]:
        m = M.make_model("ridge", {"alpha": a}).fit(Xc["train"], y["train"])
        val_rows.append(score("ridge", {"alpha": a}, m, Xc["validation"], experiment="history_calendar_only"))
    for a in GRIDS["ridge"]:
        r = next(r for r in val_rows if r["experiment"] == "principal" and r["method"] == "ridge"
                 and r["setting"] == f"alpha={a:g}")
        val_rows.append({**r, "experiment": "history_full_inputs"})

    vr = pd.DataFrame(val_rows)
    vr["selected"] = False
    selected = {}
    for method in M.METHODS:
        sub = vr[(vr.experiment == "principal") & (vr.method == method)]
        best = sub.sort_values(["validation_mae"], kind="stable").iloc[0]   # ties -> first in grid order
        vr.loc[best.name, "selected"] = True
        selected[method] = dict(setting=best["setting"], setting_json=json.loads(best["setting_json"]),
                                validation_mae=float(best["validation_mae"]))
    for exp in ["history_calendar_only", "history_full_inputs"]:
        sub = vr[vr.experiment == exp]
        vr.loc[sub["validation_mae"].idxmin(), "selected"] = True
    best_mae = min(selected[m]["validation_mae"] for m in M.METHODS)
    default = next(m for m in TIE_ORDER if selected[m]["validation_mae"] == best_mae)
    return dict(idx=idx, rawf=rawf, y=y, pre=pre, pre_cal=pre_cal, X=X, Xc=Xc, vr=vr,
                coef_rows=coef_rows, fitted=fitted, selected=selected, default=default)


def main(data_path: str):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(MOD, exist_ok=True)
    t_start = time.time()

    # 1. audit -------------------------------------------------------------
    raw = D.load_raw(data_path)
    aud = D.audit(raw)
    aud.audit_table.to_csv(os.path.join(OUT, "data_audit.csv"), index=False)
    D.missing_hours_by_day(aud.missing_hours).to_csv(os.path.join(OUT, "missing_hours.csv"), index=False)
    clean = aud.clean
    print(aud.audit_table.to_string(index=False))

    S = fit_select(clean)
    idx, rawf, y, pre, pre_cal, X, Xc = (S[k] for k in ['idx', 'rawf', 'y', 'pre', 'pre_cal', 'X', 'Xc'])
    vr, coef_rows, fitted, selected, default = (S[k] for k in ['vr', 'coef_rows', 'fitted', 'selected', 'default'])
    pre.save(os.path.join(MOD, "preprocessor_full.json"))
    pre_cal.save(os.path.join(MOD, "preprocessor_calendar.json"))
    vr.to_csv(os.path.join(OUT, "validation_results.csv"), index=False)
    pd.DataFrame(coef_rows).to_csv(os.path.join(OUT, "lasso_coefficients.csv"), index=False)
    print("selected:", {k: v["setting"] for k, v in selected.items()}, "default:", default)

    # 5. freeze: refit selected settings (timed) and save ----------------------
    final, train_time = {}, {}
    for method in M.METHODS:
        s = selected[method]["setting_json"]
        if method == "bayes":
            s = {"alpha_prior_precision": s["alpha_prior_precision"]}
        t0 = time.perf_counter()
        if method == "baseline":
            mdl = M.SeasonalBaseline().fit(rawf["train"], y["train"])
        else:
            p_tmp = F.Preprocessor("full").fit(rawf["train"])
            Xt = p_tmp.transform(rawf["train"])
            mdl = M.make_model(method, s).fit(Xt, y["train"])
        train_time[method] = time.perf_counter() - t0
        final[method] = mdl
        M.save_model(mdl, os.path.join(MOD, f"{method}.pkl"))
    # the refit must reproduce the selected candidate exactly
    for method in M.METHODS:
        ref = fitted[(method, selected[method]["setting"])]
        a = ref.predict_dist(rawf["validation"], X["validation"])["raw_mean"]
        bb = final[method].predict_dist(rawf["validation"], X["validation"])["raw_mean"]
        assert np.allclose(a, bb, atol=1e-6), method
    if isinstance(final["baseline"], M.SeasonalBaseline):
        final["baseline"].table_frame().to_csv(os.path.join(MOD, "baseline_lookup.csv"), index=False)
    np.savez(os.path.join(MOD, "bayes_posterior.npz"), m=final["bayes"].m, S=final["bayes"].S,
             alpha=final["bayes"].alpha, sigma2=final["bayes"].sigma2,
             feature_names=np.array(["(intercept)"] + pre.feature_names))

    # TV bins from training counts, frozen before validation is examined
    q = np.quantile(y["train"], np.linspace(0.1, 0.9, 9))
    bounds = sorted(set(int(round(v)) for v in q if v > 0))
    edges = [0] + bounds  # bins [e_i, e_{i+1}), final overflow bin [last, inf)

    config = dict(
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        data_file=os.path.relpath(data_path, ROOT),
        periods=D.PERIODS,
        availability_rule="Forecast for hour t uses calendar fields of t and count/weather of completed hours s<t, by exact timestamp lookup.",
        features=dict(calendar_categorical=F.CALENDAR_CATEGORICAL, calendar_binary=F.CALENDAR_BINARY,
                      historical_numeric=F.HIST_NUMERIC, lags_hours=dict(prev_hour_count=1, prev_day_count=24,
                      prev_week_count=168, weather=1), prev_24h_mean="mean of t-24..t-1, all 24 required",
                      missing_rule="training median + missing flag", unseen_category_rule="all-zero one-hot block",
                      constant_column_rule="scale 1 (centred only)", n_features_full=len(pre.feature_names)),
        grids=GRIDS,
        selected={m: dict(setting=selected[m]["setting"], settings=final[m].settings(),
                          validation_mae=selected[m]["validation_mae"]) for m in M.METHODS},
        default_model=default,
        tie_rule="lowest validation MAE; within a method the first value in grid order; across methods order " + ", ".join(TIE_ORDER),
        clipping="Ridge/Lasso/Bayesian point forecasts clipped at 0 for display and MAE/RMSE; raw mean kept.",
        lasso_zero_tolerance=LASSO_ZERO_TOL,
        count_bins_edges=edges,
        app_forecast_tolerance=APP_TOLERANCE,
        training_time_definition="seconds to fit preprocessing on training inputs plus the final model fit at the selected setting (excludes candidate search)",
        versions=dict(python=os.sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                      scikit_learn=sklearn.__version__, scipy=scipy.__version__),
    )
    M.to_json(config, os.path.join(ROOT, "config.json"))

    # 6. replay validation and test -------------------------------------------
    pred_rows = []
    for p in ["validation", "test"]:
        for method in M.METHODS:
            d = final[method].predict_dist(rawf[p], X[p])
            df = pd.DataFrame({"timestamp": idx[p].strftime("%Y-%m-%d %H:%M"), "period": p, "model": method,
                               "raw_mean": d["raw_mean"], "point_forecast": d["point"], "actual": y[p]})
            df["residual"] = df["actual"] - df["point_forecast"]
            df["abs_error"] = df["residual"].abs()
            for k in ["lo80", "hi80", "lo95", "hi95", "noise_var", "coef_var", "total_var"]:
                df[k] = d[k]
            pred_rows.append(df)
    pred = pd.concat(pred_rows, ignore_index=True)
    pred.to_csv(os.path.join(OUT, "predictions.csv"), index=False, float_format="%.6f")

    # test metrics ---------------------------------------------------------
    test = pred[pred.period == "test"]
    met = []
    for method in M.METHODS:
        t = test[test.model == method]
        met.append(dict(table="point_scores", model=method, period="test", n=len(t),
                        mae=M.mae(t.actual, t.point_forecast), rmse=M.rmse(t.actual, t.point_forecast),
                        train_time_s=train_time[method], is_default=method == default,
                        validation_mae=selected[method]["validation_mae"]))
    tp = test[test.model == "poisson"]
    met.append(dict(table="poisson_log_score", model="poisson", period="test", n=len(tp),
                    mean_log_loss=M.poisson_log_score(tp.actual, tp.raw_mean)))
    hours_meta = pd.to_datetime(test["timestamp"])
    wd_map = clean["workingday"]
    for method in sorted(M.HAS_INTERVALS):
        t = test[test.model == method].copy()
        t["hour"] = pd.to_datetime(t.timestamp).dt.hour
        t["workingday"] = wd_map.reindex(pd.to_datetime(t.timestamp)).to_numpy()
        for lvl, lo, hi in [(80, "lo80", "hi80"), (95, "lo95", "hi95")]:
            c, w = M.coverage_width(t.actual, t[lo], t[hi])
            met.append(dict(table="interval_coverage", model=method, period="test", level=lvl,
                            group_type="all", group="all", n=len(t), coverage=c, mean_width=w))
            for gt in ["hour", "workingday"]:
                vals = range(24) if gt == "hour" else [0, 1]
                for g in vals:
                    s = t[t[gt] == g]
                    c, w = M.coverage_width(s.actual, s[lo], s[hi])
                    met.append(dict(table="interval_coverage", model=method, period="test", level=lvl,
                                    group_type=gt, group=str(g), n=len(s), coverage=c, mean_width=w))
    td = test[test.model == default].copy()
    td["hour"] = pd.to_datetime(td.timestamp).dt.hour
    for h in range(24):
        s = td[td.hour == h]
        met.append(dict(table="residual_by_hour", model=default, period="test", group_type="hour",
                        group=str(h), n=len(s), mean_residual=float(s.residual.mean()) if len(s) else np.nan))
    pd.DataFrame(met).to_csv(os.path.join(OUT, "metrics.csv"), index=False)

    # largest errors of the default model (ties: earlier timestamp first)
    le = td.sort_values(["abs_error", "timestamp"], ascending=[False, True], kind="stable").head(10)
    le.drop(columns=["hour"]).to_csv(os.path.join(OUT, "largest_errors.csv"), index=False)

    # count distribution: training vs validation
    def props(v):
        cnts = np.histogram(v, bins=edges + [np.inf])[0]
        return cnts, cnts / cnts.sum()
    ct, pt = props(y["train"])
    cv, pv = props(y["validation"])
    labels = [f"{edges[i]}-{edges[i + 1] - 1}" for i in range(len(edges) - 1)] + [f"{edges[-1]}+"]
    cd = pd.DataFrame(dict(bin=labels, lower=edges, upper_exclusive=edges[1:] + [np.inf],
                           train_count=ct, train_prop=pt, validation_count=cv, validation_prop=pv))
    cd["abs_diff"] = (cd.train_prop - cd.validation_prop).abs()
    cd.to_csv(os.path.join(OUT, "count_distribution.csv"), index=False)
    tv = 0.5 * cd.abs_diff.sum()
    M.to_json(dict(tv=tv, n_train=int(len(y["train"])), n_validation=int(len(y["validation"])), edges=edges),
              os.path.join(OUT, "count_distribution_tv.json"))

    print(pd.DataFrame(met).query("table=='point_scores'")[["model", "n", "mae", "rmse", "validation_mae"]].to_string(index=False))
    print(f"TV(train, validation) = {tv:.4f}; total time {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "hour.csv"))
    main(ap.parse_args().data)
