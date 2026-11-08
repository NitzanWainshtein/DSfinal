"""Bike-Rental Forecast Dashboard (Streamlit).

Loads the models, preprocessing and results written by train_evaluate.py.
Nothing is refitted here. Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data as D
import features as F
import models as M

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
MOD = os.path.join(ROOT, "models")

st.set_page_config(page_title="Bike-Rental Forecast Dashboard", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_all():
    cfg = json.load(open(os.path.join(ROOT, "config.json")))
    aud = D.audit(D.load_raw(os.path.join(ROOT, cfg["data_file"])))
    pre = F.Preprocessor.load(os.path.join(MOD, "preprocessor_full.json"))
    mdl = {m: M.load_model(os.path.join(MOD, f"{m}.pkl")) for m in M.METHODS}
    res = dict(
        pred=pd.read_csv(os.path.join(OUT, "predictions.csv")),
        metrics=pd.read_csv(os.path.join(OUT, "metrics.csv")),
        val=pd.read_csv(os.path.join(OUT, "validation_results.csv")),
        lasso=pd.read_csv(os.path.join(OUT, "lasso_coefficients.csv")),
        audit=pd.read_csv(os.path.join(OUT, "data_audit.csv")),
        missing=pd.read_csv(os.path.join(OUT, "missing_hours.csv")),
        largest=pd.read_csv(os.path.join(OUT, "largest_errors.csv")),
        cdist=pd.read_csv(os.path.join(OUT, "count_distribution.csv")),
        tv=json.load(open(os.path.join(OUT, "count_distribution_tv.json"))),
    )
    return cfg, aud, pre, mdl, res


cfg, AUD, PRE, MODELS, R = load_all()
CLEAN = AUD.clean
DEFAULT = cfg["default_model"]
LBL = M.METHOD_LABELS
P = cfg["periods"]


def period_label(p):
    return f"{p} period ({P[p][0]} to {P[p][1]})"


def model_label(m):
    return LBL[m] + (" (default)" if m == DEFAULT else "")


def csv_button(df, name, label="Download CSV", key=None):
    st.download_button(label, df.to_csv(index=False).encode(), file_name=name, mime="text/csv",
                       key=key or f"dl_{name}")


def fmt_setting(m):
    s = cfg["selected"][m]["settings"]
    if m == "baseline":
        return "lookup by (hour, workingday); fallback: overall training mean"
    if m == "bayes":
        return f"prior precision alpha={s['alpha_prior_precision']:g}, sigma2={s['sigma2']:.1f}"
    return f"penalty alpha={s['alpha']:g}"


def forecast_one(ts: pd.Timestamp, method: str):
    """Build inputs for ts from completed earlier hours and apply the loaded model."""
    raw = F.build_raw_features(CLEAN, pd.DatetimeIndex([ts]))
    X = PRE.transform(raw)
    d = MODELS[method].predict_dist(raw, X)
    return raw, {k: float(v[0]) for k, v in d.items()}


def inputs_table(raw: pd.DataFrame, method: str) -> pd.DataFrame:
    r = raw.iloc[0]
    ts = raw.index[0]
    used_cal = {"hour", "workingday"} if method == "baseline" else set(F.CALENDAR_CATEGORICAL + F.CALENDAR_BINARY)
    rows = []
    for c in F.CALENDAR_CATEGORICAL + F.CALENDAR_BINARY:
        rows.append(dict(input=c, value=str(int(r[c])), source=f"calendar of target hour {ts:%Y-%m-%d %H:%M}",
                         missing="", replacement_used="",
                         used_by_model="yes" if c in used_cal else "no (context only)"))
    for f in F.HIST_NUMERIC:
        miss = bool(r[f"{f}__missing"])
        rows.append(dict(input=F.READABLE[f], value="(absent)" if miss else f"{r[f]:.4g}",
                         source=r[f"{f}__source"], missing="TRUE" if miss else "false",
                         replacement_used=f"training median {PRE.medians[f]:.4g}" if miss else "",
                         used_by_model="no (context only)" if method == "baseline" else "yes"))
    return pd.DataFrame(rows)


def show_dist(d: dict, method: str):
    c = st.columns(3)
    c[0].metric("Point forecast (rentals in the hour)", f"{d['point']:.1f}")
    if method in M.HAS_INTERVALS:
        c[1].metric("80% interval", f"[{d['lo80']:.1f}, {d['hi80']:.1f}]")
        c[2].metric("95% interval", f"[{d['lo95']:.1f}, {d['hi95']:.1f}]")
    else:
        c[1].metric("80% interval", "unavailable")
        c[2].metric("95% interval", "unavailable")
    if method in ("ridge", "lasso", "bayes") and d["raw_mean"] < 0:
        st.caption(f"Raw model mean {d['raw_mean']:.2f} was clipped to 0 for the point forecast.")
    if method == "bayes":
        st.table(pd.DataFrame({
            "quantity": ["Predictive mean (raw)", "Observation-noise variance sigma^2",
                         "Coefficient-uncertainty variance phi^T S phi", "Total predictive variance",
                         "Predictive standard deviation"],
            "value": [d["raw_mean"], d["noise_var"], d["coef_var"], d["total_var"], np.sqrt(d["total_var"])]
        }).style.format({"value": "{:.4f}"}))
    if method == "poisson":
        st.caption("Poisson intervals: central quantiles of Poisson(mu) with the fitted mean treated as fixed.")


# ---------------------------------------------------------------- screens
def screen_overview():
    st.header("Data overview")
    st.caption("Source: UCI Bike Sharing dataset, hour.csv (Capital Bikeshare, 2011-2012).")
    with st.expander("Field definitions"):
        st.table(pd.DataFrame(D.FIELD_DEFINITIONS.items(), columns=["field", "meaning"]))
    tab1, tab2, tab3 = st.tabs(["Hourly observations", "Timestamp audit", "Count distribution"])

    with tab1:
        lo, hi = CLEAN.index.min().date(), CLEAN.index.max().date()
        rng = st.date_input("Date range", value=(pd.Timestamp("2012-07-01").date(), pd.Timestamp("2012-08-31").date()),
                            min_value=lo, max_value=hi)
        if not (isinstance(rng, tuple) and len(rng) == 2):
            st.info("Select a start and end date.")
            return
        a, b = pd.Timestamp(rng[0]), pd.Timestamp(rng[1]) + pd.Timedelta(hours=23)
        sub = CLEAN.loc[a:b]
        miss = AUD.missing_hours[(AUD.missing_hours >= a) & (AUD.missing_hours <= b)]
        excl = AUD.excluded
        n_ex = int(((excl["timestamp"] >= a) & (excl["timestamp"] <= b)).sum()) if len(excl) else 0
        c = st.columns(4)
        c[0].metric("Observed hours", len(sub))
        c[1].metric("Missing hours", len(miss))
        c[2].metric("Duplicate rows excluded", int((excl["reason"].str.startswith("duplicate")).sum()) if len(excl) else 0)
        c[3].metric("Rows excluded in range", n_ex)
        full = sub["cnt"].reindex(pd.date_range(a, b, freq="h"))
        fig = go.Figure(go.Scatter(x=full.index, y=full.values, mode="lines", connectgaps=False, name="cnt"))
        fig.update_layout(title=f"Hourly rentals {rng[0]} to {rng[1]} (gaps = missing hours)",
                          xaxis_title="Hour start", yaxis_title="Rentals per hour", height=360)
        st.plotly_chart(fig, width="stretch")
        g = sub.groupby(["hr", "workingday"])["cnt"].agg(["mean", "size"]).reset_index()
        g["day type"] = g["workingday"].map({1: "working day", 0: "weekend/holiday"})
        fig2 = px.line(g, x="hr", y="mean", color="day type", markers=True, hover_data={"size": True},
                       labels={"hr": "Hour of day", "mean": "Average rentals per hour", "size": "n hours"},
                       title="Average rentals by hour and working-day status (selected range)")
        st.plotly_chart(fig2, width="stretch")
        st.dataframe(g.rename(columns={"hr": "hour", "mean": "mean_cnt", "size": "n_hours"})
                     [["hour", "day type", "mean_cnt", "n_hours"]], hide_index=True, height=240)
        st.subheader("Source rows")
        show = sub.reset_index()[["timestamp", "instant", "cnt", "hr", "weekday", "mnth", "holiday",
                                  "workingday", "temp", "hum", "windspeed", "period"]]
        st.dataframe(show, hide_index=True, height=300)
        csv_button(show, "source_rows.csv", "Download displayed rows")

    with tab2:
        st.markdown("Observed timestamps compared with a complete hourly calendar. "
                    "An absent hour is **not** a zero count: it is never scored and never used as history.")
        st.dataframe(R["audit"], hide_index=True)
        csv_button(R["audit"], "data_audit.csv", "Download audit")
        st.markdown("**Days with missing hours**")
        st.dataframe(R["missing"], hide_index=True, height=260)
        csv_button(R["missing"], "missing_hours.csv", "Download missing hours")

    with tab3:
        cd, tv = R["cdist"], R["tv"]
        st.markdown(f"Fixed full periods: **{period_label('train')}**, n={tv['n_train']}; "
                    f"**{period_label('validation')}**, n={tv['n_validation']}. "
                    "Bin boundaries were chosen from training-count deciles and frozen before validation was examined.")
        st.metric("Binned total variation distance TV(train, validation)", f"{tv['tv']:.4f}")
        long = pd.concat([cd.assign(period="train", proportion=cd.train_prop),
                          cd.assign(period="validation", proportion=cd.validation_prop)])
        fig = px.bar(long, x="bin", y="proportion", color="period", barmode="group",
                     labels={"bin": "Rentals per hour (bin)", "proportion": "Proportion of observed hours"},
                     title="Recorded hourly counts: training vs validation (shared bins)")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(cd, hide_index=True)
        csv_button(cd, "count_distribution.csv")


def screen_forecast():
    st.header("Forecast one hour")
    st.caption(f"Historical replay over the {period_label('test')}. Models and preprocessing are fixed; "
               "inputs use only the target hour's calendar and completed earlier hours.")
    test_idx = CLEAN.index[CLEAN["period"] == "test"]
    c = st.columns([1, 1, 2])
    day = c[0].date_input("Test date", value=pd.Timestamp("2012-08-01").date(),
                          min_value=test_idx.min().date(), max_value=test_idx.max().date())
    hours = test_idx[test_idx.normalize() == pd.Timestamp(day)]
    if len(hours) == 0:
        st.warning("No observed test hours on this date.")
        return
    hour_default = list(hours.hour).index(8) if 8 in list(hours.hour) else 0
    h = c[1].selectbox("Hour (observed hours only)", list(hours.hour), index=hour_default,
                       format_func=lambda x: f"{x:02d}:00")
    method = c[2].selectbox("Model", M.METHODS, index=M.METHODS.index(DEFAULT), format_func=model_label)
    ts = pd.Timestamp(day) + pd.Timedelta(hours=int(h))
    key = (str(ts), method)
    if st.session_state.get("fc_key") != key:          # new selection clears old forecast/reveal
        st.session_state.update(fc_key=key, fc=None, revealed=False)

    if st.button("Forecast", type="primary"):
        st.session_state.fc = forecast_one(ts, method)
        st.session_state.revealed = False
    if st.session_state.fc is None:
        st.info(f"Target hour {ts:%Y-%m-%d %H:%M}-{ts:%H}:59. Press Forecast.")
        return
    raw, d = st.session_state.fc
    st.subheader(f"{model_label(method)} - target {ts:%A %Y-%m-%d %H:%M}")
    st.caption(f"Model reference: models/{method}.pkl; setting: {fmt_setting(method)}; "
               "preprocessing: models/preprocessor_full.json (fitted on training only).")
    show_dist(d, method)
    st.markdown("**Inputs used**")
    st.dataframe(inputs_table(raw, method), hide_index=True)

    saved = R["pred"].query("period=='test' and model==@method and timestamp==@ts.strftime('%Y-%m-%d %H:%M')")
    if len(saved):
        diff = abs(saved["point_forecast"].iloc[0] - d["point"])
        ok = diff <= cfg["app_forecast_tolerance"]
        st.caption(f"Agreement with saved predictions.csv: |difference| = {diff:.2e} "
                   f"({'within' if ok else 'OUTSIDE'} tolerance {cfg['app_forecast_tolerance']:g}).")

    if st.button("Reveal recorded count"):
        st.session_state.revealed = True
    if st.session_state.revealed:
        actual = int(CLEAN.loc[ts, "cnt"])
        c = st.columns(3)
        c[0].metric("Recorded count", actual)
        c[1].metric("Signed error (actual - forecast)", f"{actual - d['point']:+.1f}")
        c[2].metric("Absolute error", f"{abs(actual - d['point']):.1f}")
        if method in M.HAS_INTERVALS:
            st.caption(f"80% interval {'covers' if d['lo80'] <= actual <= d['hi80'] else 'misses'} the outcome; "
                       f"95% interval {'covers' if d['lo95'] <= actual <= d['hi95'] else 'misses'} it.")


def screen_compare():
    st.header("Compare models")
    met = R["metrics"]
    st.subheader("Validation selection (" + period_label("validation") + ")")
    sel = pd.DataFrame([dict(model=model_label(m), selected_setting=fmt_setting(m),
                             validation_MAE=cfg["selected"][m]["validation_mae"]) for m in M.METHODS])
    st.dataframe(sel.style.format({"validation_MAE": "{:.3f}"}), hide_index=True)
    st.caption("Default = lowest validation MAE, chosen and frozen before test scoring. Tie rule: " + cfg["tie_rule"])
    with st.expander("All validation candidates"):
        v = R["val"][R["val"].experiment == "principal"]
        st.dataframe(v, hide_index=True)
        csv_button(v, "validation_candidates.csv")

    st.subheader("Full final-test scores (" + period_label("test") + ")")
    ps = met[met.table == "point_scores"][["model", "n", "mae", "rmse", "train_time_s", "is_default"]].copy()
    ps["model"] = ps["model"].map(model_label)
    ps = ps.rename(columns={"n": "evaluated_hours", "mae": "MAE", "rmse": "RMSE", "train_time_s": "training_time_s"})
    st.dataframe(ps.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "training_time_s": "{:.3f}"}), hide_index=True)
    st.caption("Training time: " + cfg["training_time_definition"] + ".")
    csv_button(ps, "test_scores.csv", "Download score table")

    st.subheader("Poisson probability score")
    pl = met[met.table == "poisson_log_score"].iloc[0]
    c = st.columns(2)
    c[0].metric("Mean -log Pr(Y = y | mu) on test (nats, lower is better)", f"{pl['mean_log_loss']:.3f}")
    c[1].metric("Evaluated hours", int(pl["n"]))
    st.caption("Discrete count-probability score of the Poisson forecast; not comparable with a Gaussian density score.")

    st.subheader("Forecasts against recorded counts")
    c = st.columns([1, 2])
    method = c[0].selectbox("Model", M.METHODS, index=M.METHODS.index(DEFAULT), format_func=model_label, key="cmp_m")
    rng = c[1].date_input("Test date range", value=(pd.Timestamp("2012-08-01").date(), pd.Timestamp("2012-08-14").date()),
                          min_value=pd.Timestamp(P["test"][0]).date(), max_value=pd.Timestamp(P["test"][1]).date(),
                          key="cmp_r")
    if not (isinstance(rng, tuple) and len(rng) == 2):
        st.info("Select a start and end date.")
        return
    a, b = pd.Timestamp(rng[0]), pd.Timestamp(rng[1]) + pd.Timedelta(hours=23)
    pr = R["pred"].query("period=='test' and model==@method").copy()
    pr["ts"] = pd.to_datetime(pr.timestamp)
    pr = pr[(pr.ts >= a) & (pr.ts <= b)]
    grid = pd.date_range(a, b, freq="h")
    s = pr.set_index("ts").reindex(grid)
    fig = go.Figure()
    fig.add_scatter(x=grid, y=s["actual"], name="Recorded count", mode="lines", connectgaps=False)
    fig.add_scatter(x=grid, y=s["point_forecast"], name=f"{LBL[method]} forecast", mode="lines", connectgaps=False)
    if method in M.HAS_INTERVALS:
        fig.add_scatter(x=grid, y=s["hi80"], line=dict(width=0), showlegend=False, connectgaps=False)
        fig.add_scatter(x=grid, y=s["lo80"], fill="tonexty", line=dict(width=0), name="80% interval",
                        fillcolor="rgba(100,100,200,0.2)", connectgaps=False)
    fig.update_layout(title=f"{LBL[method]}: test {rng[0]} to {rng[1]}", xaxis_title="Hour start",
                      yaxis_title="Rentals per hour", height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Range MAE: {pr['abs_error'].mean():.2f} over {len(pr)} hours (the table above is the full test period).")
    csv_button(pr.drop(columns=["ts"]), f"forecasts_{method}_{rng[0]}_{rng[1]}.csv", "Download displayed forecasts")


def screen_behaviour():
    st.header("Inspect model behaviour")
    tabs = st.tabs(["Regularisation", "History experiment", "Interval coverage", "Largest errors", "Residuals by hour"])
    val, met = R["val"], R["metrics"]

    with tabs[0]:
        lv = val[(val.experiment == "principal") & (val.method == "lasso")].copy()
        lv["alpha"] = lv["setting"].str.replace("alpha=", "").astype(float)
        sel_a = cfg["selected"]["lasso"]["settings"]["alpha"]
        st.caption(f"Lasso fits on the {period_label('train')}; scores on the {period_label('validation')}. "
                   f"Zero tolerance |coef| <= {cfg['lasso_zero_tolerance']:g}. Selected penalty: {sel_a:g}.")
        c = st.columns(2)
        for col, y, title in [(c[0], "validation_mae", "Validation MAE"), (c[1], "n_nonzero_coef", "Non-zero coefficients")]:
            fig = px.line(lv, x="alpha", y=y, markers=True, log_x=True, title=f"{title} vs Lasso penalty",
                          labels={"alpha": "Lasso penalty alpha (log scale)", y: title})
            fig.add_vline(x=sel_a, line_dash="dash", annotation_text="selected")
            col.plotly_chart(fig, width="stretch")
        st.dataframe(lv[["alpha", "validation_mae", "n_nonzero_coef", "n_coef_total", "n_hours", "selected"]], hide_index=True)
        co = R["lasso"]
        a = st.select_slider("Penalty for coefficient table", options=sorted(co.alpha.unique()), value=sel_a)
        t = co[co.alpha == a]
        st.metric("Intercept", f"{t[t.feature == '(intercept)'].coefficient.iloc[0]:.3f}")
        st.dataframe(t[t.feature != "(intercept)"][["feature", "coefficient", "is_zero"]], hide_index=True, height=300)
        piv = co[co.feature != "(intercept)"].pivot(index="feature", columns="alpha", values="coefficient")
        piv = piv.reindex(PRE.feature_names)
        st.markdown("**Coefficients across tested penalties** (0 = removed by this fit)")
        st.dataframe(piv.round(3), height=320)
        csv_button(co, "lasso_coefficients.csv")

    with tabs[1]:
        rows = []
        for exp, name, desc in [("history_calendar_only", "Calendar only", "hour, weekday, month (one-hot), holiday, workingday"),
                                ("history_full_inputs", "Full required input set",
                                 "calendar + 3 count lags + previous-24h mean + previous-hour weather + 7 missing flags")]:
            r = val[(val.experiment == exp) & val.selected].iloc[0]
            rows.append(dict(input_set=name, inputs=desc, selected_penalty=r.setting, validation_hours=r.n_hours,
                             validation_MAE=r.validation_mae))
        h = pd.DataFrame(rows)
        st.dataframe(h.style.format({"validation_MAE": "{:.3f}"}), hide_index=True)
        diff = h.validation_MAE[0] - h.validation_MAE[1]
        st.metric("Calendar-only MAE minus full-input MAE", f"{diff:+.3f}",
                  help="Positive: the full-input Ridge had lower validation error.")
        st.caption("Each input set tuned separately on validation. Rental history and weather are added together, "
                   "so this does not identify which of them produced the gain, nor any causal effect.")
        hv = val[val.experiment.str.startswith("history")]
        with st.expander("All candidates"):
            st.dataframe(hv, hide_index=True)
        csv_button(hv, "history_experiment.csv")

    with tabs[2]:
        c = st.columns(2)
        m = c[0].selectbox("Model", ["poisson", "bayes"], format_func=model_label, key="cov_m")
        lvl = c[1].selectbox("Nominal level", [80, 95], format_func=lambda x: f"{x}%")
        cv = met[(met.table == "interval_coverage") & (met.model == m) & (met.level == lvl)].copy()
        allr = cv[cv.group_type == "all"].iloc[0]
        c = st.columns(3)
        c[0].metric("Observed coverage (full test)", f"{100 * allr.coverage:.1f}%")
        c[1].metric("Nominal", f"{lvl}%")
        c[2].metric("Mean width (rentals)", f"{allr.mean_width:.1f}")
        st.caption(f"{period_label('test')}, n={int(allr.n)}. Endpoints inclusive; raw Gaussian endpoints "
                   "(including negative ones) are used as saved.")
        for gt, title in [("hour", "By hour of day"), ("workingday", "By working-day status")]:
            g = cv[cv.group_type == gt].copy()
            g["observed_coverage_%"] = (100 * g.coverage).round(1)
            g["nominal_%"] = lvl
            g = g.rename(columns={"group": gt, "n": "n_hours", "mean_width": "mean_width_rentals"})
            g.loc[g.n_hours == 0, ["observed_coverage_%", "mean_width_rentals"]] = np.nan
            st.markdown(f"**{title}**")
            st.dataframe(g[[gt, "n_hours", "observed_coverage_%", "nominal_%", "mean_width_rentals"]],
                         hide_index=True, height=300 if gt == "hour" else 110)
        csv_button(cv, f"interval_coverage_{m}_{lvl}.csv")

    with tabs[3]:
        le = R["largest"].copy()
        st.caption(f"Default model ({model_label(DEFAULT)}), {period_label('test')}. "
                   "Ten largest absolute errors; ties ordered by earlier timestamp.")
        st.dataframe(le[["timestamp", "model", "actual", "point_forecast", "residual", "abs_error"]], hide_index=True)
        csv_button(le, "largest_errors.csv")
        pick = st.selectbox("Open forecast details for", le.timestamp.tolist())
        ts = pd.Timestamp(pick)
        raw, d = forecast_one(ts, DEFAULT)
        st.markdown(f"**Selected hour: {ts:%A %Y-%m-%d %H:%M}** - recorded count {int(CLEAN.loc[ts, 'cnt'])}")
        show_dist(d, DEFAULT)
        st.dataframe(inputs_table(raw, DEFAULT), hide_index=True)

    with tabs[4]:
        rb = met[met.table == "residual_by_hour"].copy()
        rb["hour"] = rb.group.astype(int)
        fig = px.bar(rb, x="hour", y="mean_residual",
                     labels={"hour": "Hour of day", "mean_residual": "Mean signed residual (actual - forecast)"},
                     title=f"{model_label(DEFAULT)}: mean residual by hour, {period_label('test')}")
        fig.add_hline(y=0, line_color="black")
        st.plotly_chart(fig, width="stretch")
        st.caption("Positive = underprediction on average. Signed averages: opposing errors can cancel.")
        st.dataframe(rb[["hour", "mean_residual", "n"]], hide_index=True, height=300)
        csv_button(rb[["hour", "mean_residual", "n", "model", "period"]], "residuals_by_hour.csv")


SCREENS = {"Data overview": screen_overview, "Forecast one hour": screen_forecast,
           "Compare models": screen_compare, "Inspect model behaviour": screen_behaviour}
st.sidebar.title("Bike-Rental Forecasts")
choice = st.sidebar.radio("Screen", list(SCREENS))
st.sidebar.caption(f"Default model: {LBL[DEFAULT]}\n\nPeriods: train {P['train'][0]}-{P['train'][1]}; "
                   f"validation {P['validation'][0]}-{P['validation'][1]}; test {P['test'][0]}-{P['test'][1]}")
SCREENS[choice]()
