"""Bike-Rental Forecast Dashboard (Streamlit).

Loads the models, preprocessing and results written by train_evaluate.py.
Nothing is refitted here. Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import data as D
import features as F
import models as M
import ui

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


def show_interval_and_model_notes(d: dict, method: str):
    if method in ("ridge", "lasso", "bayes") and d["raw_mean"] < 0:
        ui.note(f"Raw model mean {d['raw_mean']:.2f} was clipped to 0 for the point forecast.")
    if method == "poisson":
        ui.note("Poisson intervals: central quantiles of Poisson(mu) with the fitted mean treated as fixed.")


def longest_gap(missing: pd.DatetimeIndex):
    """Longest run of consecutive absent hours: (length, first hour, last hour)."""
    best = (0, None, None)
    run_start = prev = None
    for t in missing.sort_values():
        run_start = t if prev is None or t - prev != pd.Timedelta(hours=1) else run_start
        n = int((t - run_start) / pd.Timedelta(hours=1)) + 1
        if n > best[0]:
            best = (n, run_start, t)
        prev = t
    return best


def peak_label(fig, hour, value, colour):
    fig.add_annotation(x=hour, y=value, text=f"{int(hour):02d}:00 · {value:.0f}", showarrow=False, yshift=14,
                       font=dict(color=colour, size=12))


# ---------------------------------------------------------------- screens
def overview_range():
    """The date-range section: counts, gap-preserving series, averages with group sizes, source rows."""
    with ui.card("range"):
        ui.heading("Explore a date range", "The selected range drives the counts, the series and the tables in this card.")
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
        n_dup = int((excl["reason"].str.startswith("duplicate")).sum()) if len(excl) else 0
        ui.kpis([ui.kpi_html("Observed hours", f"{len(sub):,}"), ui.kpi_html("Missing hours", f"{len(miss):,}"),
                 ui.kpi_html("Duplicate rows excluded", str(n_dup)), ui.kpi_html("Rows excluded in range", str(n_ex))])
        full = sub["cnt"].reindex(pd.date_range(a, b, freq="h"))
        fig = go.Figure(go.Scatter(x=full.index, y=full.values, mode="lines", connectgaps=False, name="cnt",
                                   line=dict(color=ui.ACTUAL_COLOUR, width=1.2)))
        fig.update_layout(xaxis_title="Hour start", yaxis_title="Rentals per hour", showlegend=False)
        ui.show(f'<h3 class="fods-h2" style="font-size:1.1rem">Hourly rentals, {rng[0]} to {rng[1]}</h3>'
                '<p class="fods-note">Gaps in the line are missing hours.</p>')
        ui.chart(fig, 320)

        g = sub.groupby(["hr", "workingday"])["cnt"].agg(["mean", "size"]).reset_index()
        g["day type"] = g["workingday"].map({1: "working day", 0: "weekend/holiday"})
        fig2 = go.Figure()
        for day_type, colour in (("working day", ui.NAVY), ("weekend/holiday", ui.COPPER)):
            part = g[g["day type"] == day_type]
            fig2.add_scatter(x=part["hr"], y=part["mean"], name=day_type, mode="lines+markers", customdata=part["size"],
                             line=dict(color=colour), hovertemplate="%{x}:00, %{y:.1f} rentals, n=%{customdata} hours<extra></extra>")
        fig2.update_layout(xaxis_title="Hour of day", yaxis_title="Average rentals per hour")
        ui.show('<h3 class="fods-h2" style="font-size:1.1rem">Average rentals by hour and working-day status (selected range)</h3>')
        ui.chart(fig2, 320)
        st.dataframe(g.rename(columns={"hr": "hour", "mean": "mean_cnt", "size": "n_hours"})
                     [["hour", "day type", "mean_cnt", "n_hours"]], hide_index=True, height=240)
        ui.show('<h3 class="fods-h2" style="font-size:1.1rem">Source rows</h3>')
        show = sub.reset_index()[["timestamp", "instant", "cnt", "hr", "weekday", "mnth", "holiday",
                                  "workingday", "temp", "hum", "windspeed", "period"]]
        st.dataframe(show, hide_index=True, height=300)
        csv_button(show, "source_rows.csv", "Download displayed rows")


def screen_overview():
    ui.page_header("Data audit", "Data overview", "What was recorded, what is absent, and how does hourly demand behave?")
    ui.note("Source: UCI Bike Sharing dataset, hour.csv (Capital Bikeshare, 2011-2012).")

    audit = R["audit"].set_index("period")
    total = audit.loc["all"]
    gap, gap_a, gap_b = longest_gap(AUD.missing_hours)
    ui.kpis([
        ui.kpi_html("Observed hours", f"{int(total.observed_hours):,}", f"of {int(total.expected_hours):,} expected"),
        ui.kpi_html("Missing hours", f"{int(total.missing_hours):,}",
                    " · ".join(f"{p} {int(audit.loc[p, 'missing_hours'])}" for p in ("train", "validation", "test")), "copper"),
        ui.kpi_html("Duplicates / invalid rows", f"{int(total.duplicate_rows_excluded)} / {int(total.invalid_target_rows_excluded)}",
                    "rows excluded by the audit"),
        ui.kpi_html("Longest gap", f"{gap} h", f"{gap_a:%Y-%m-%d %H:%M} to {gap_b:%Y-%m-%d %H:%M}"),
    ])

    with ui.card("rhythm"):
        ui.heading("The daily rhythm", f"Average rentals by hour of day over the {period_label('train')}.")
        train = CLEAN[CLEAN["period"] == "train"]
        means = train.groupby(["workingday", "hr"])["cnt"].mean()
        fig = go.Figure()
        fig.add_scatter(x=means[1].index, y=means[1].values, name="Working days", mode="lines+markers", line=dict(color=ui.NAVY))
        fig.add_scatter(x=means[0].index, y=means[0].values, name="Weekends and holidays", mode="lines+markers",
                        line=dict(color=ui.COPPER))
        morning, evening = means[1][means[1].index < 12], means[1][means[1].index >= 12]
        peak_label(fig, morning.idxmax(), morning.max(), ui.NAVY)
        peak_label(fig, evening.idxmax(), evening.max(), ui.NAVY)
        peak_label(fig, means[0].idxmax(), means[0].max(), ui.COPPER_TEXT)
        fig.update_layout(xaxis_title="Hour of day", yaxis_title="Average rentals per hour", xaxis_dtick=2)
        ui.chart(fig, 340)

    left, right = st.columns(2, gap="medium")
    with left, ui.card("audit"):
        ui.heading("Timestamp audit", "Observed timestamps compared with a complete hourly calendar.")
        ui.show('<p class="fods-sub">An absent hour is <b>not</b> a zero count: it is never scored and never used as history.</p>')
        st.dataframe(R["audit"], hide_index=True)
        csv_button(R["audit"], "data_audit.csv", "Download audit")
    with right, ui.card("demand"):
        cd, tv = R["cdist"], R["tv"]
        ui.heading("Demand shifted upward", right=ui.pill(f"TV = {tv['tv']:.3f}", "copper"))
        ui.note(f"Fixed full periods: {period_label('train')}, n={tv['n_train']}; {period_label('validation')}, "
                f"n={tv['n_validation']}. Bin boundaries were chosen from training-count deciles and frozen before "
                "validation was examined.")
        fig = go.Figure()
        fig.add_bar(x=cd["bin"], y=cd["train_prop"], name="Training", marker_color=ui.NAVY)
        fig.add_bar(x=cd["bin"], y=cd["validation_prop"], name="Validation", marker_color=ui.COPPER)
        fig.update_layout(barmode="group", xaxis_title="Rentals per hour (bin)", yaxis_title="Proportion of observed hours")
        ui.chart(fig, 320)
        with st.expander("Bin table"):
            st.dataframe(cd, hide_index=True)
        csv_button(cd, "count_distribution.csv")

    overview_range()

    with ui.card("missing"):
        ui.heading("Days with missing hours")
        st.dataframe(R["missing"], hide_index=True, height=260)
        csv_button(R["missing"], "missing_hours.csv", "Download missing hours")
    with st.expander("Field definitions"):
        st.table(pd.DataFrame(D.FIELD_DEFINITIONS.items(), columns=["field", "meaning"]))


def uncertainty_card(d: dict, method: str, diff: float | None):
    with ui.card("uncertainty"):
        ui.heading("Uncertainty")
        if method == "bayes":
            share = 100 * d["noise_var"] / d["total_var"]
            ui.show(f'<div class="stack"><i style="flex:{d["noise_var"]:.6f};background:{ui.MODEL_COLOURS["bayes"]}"></i>'
                    f'<i style="flex:{max(d["coef_var"], d["total_var"] * 0.01):.6f};background:{ui.COPPER}"></i></div>'
                    f'<div class="ib-legend" style="margin-bottom:10px"><span><em style="background:{ui.MODEL_COLOURS["bayes"]}"></em>Noise σ²</span>'
                    f'<span><em style="background:{ui.COPPER}"></em>Coefficient variance φᵀSφ</span></div>'
                    + ui.kv([("Observation noise σ²", f"{d['noise_var']:.3f}"),
                             ("Coefficient variance φᵀSφ", f"{d['coef_var']:.3f}"),
                             ("Total predictive variance", f"{d['total_var']:.3f}"),
                             ("Predictive standard deviation", f"{np.sqrt(d['total_var']):.3f}")]))
            ui.note(f"Observation noise is {share:.1f}% of the predictive variance; the rest comes from uncertainty "
                    f"in the coefficients. Raw predictive mean: {d['raw_mean']:.4f}.")
        elif method == "poisson":
            ui.note("The noise and coefficient split belongs to the Bayesian model. A Poisson forecast has variance equal "
                    "to its mean, with the fitted mean treated as fixed.")
        else:
            ui.note("This model gives a point forecast only, so it has no predictive variance to split.")
        if diff is not None:
            tol = cfg["app_forecast_tolerance"]
            text = f"(|Δ| {diff:.2e}, tolerance {tol:g})"
            ui.show(ui.pill(f"Matches saved predictions.csv {text}", "ok") if diff <= tol
                    else ui.pill(f"OUTSIDE tolerance of saved predictions.csv {text}", "warn"))


def calendar_tiles(table: pd.DataFrame, ts: pd.Timestamp) -> str:
    row = table.set_index("input")
    ctx = lambda c: " · context only" if row.loc[c, "used_by_model"] != "yes" else ""  # noqa: E731
    workingday, holiday = int(row.loc["workingday", "value"]), int(row.loc["holiday", "value"])
    day_type = "Working day" if workingday else ("Holiday" if holiday else "Weekend")

    def tile(label, value, sub):
        return f'<div class="tile"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'

    return ('<div class="tiles">'
            + tile("Hour", f"{ts:%H:%M}", f"code {row.loc['hour', 'value']}{ctx('hour')}")
            + tile("Weekday", f"{ts:%A}", f"code {row.loc['weekday', 'value']}{ctx('weekday')}")
            + tile("Month", f"{ts:%B}", f"code {row.loc['month', 'value']}{ctx('month')}")
            + tile("Day type", day_type, f"holiday {holiday}{ctx('holiday')} · workingday {workingday}{ctx('workingday')}")
            + "</div>")


def history_rows(table: pd.DataFrame, method: str) -> str:
    rows = []
    for r in table.iloc[len(F.CALENDAR_CATEGORICAL) + len(F.CALENDAR_BINARY):].itertuples():
        status = ui.pill("Missing", "warn") if r.missing == "TRUE" else ui.pill("Observed", "ok")
        if r.missing == "TRUE" and method != "baseline":
            status += f' <span class="fods-note">{ui.e(r.replacement_used)}</span>'
        if method == "baseline":
            status += " " + ui.pill("Context only", "plain")
        rows.append([ui.e(r.input), ui.e(r.value), ui.e(r.source), status])
    return ui.table(["Input", "Value", "Source hour", "Status"], rows)


def inputs_card(raw: pd.DataFrame, ts: pd.Timestamp, method: str):
    table = inputs_table(raw, method)
    with ui.card("inputs"):
        ui.heading("Inputs used", "Only the target hour's calendar and completed earlier hours; the recorded count of the target hour is never an input.")
        ui.show(f'<p class="fods-eyebrow" style="margin-top:6px">Calendar of the target hour ({ts:%Y-%m-%d %H:%M})</p>'
                + calendar_tiles(table, ts))
        ui.show('<p class="fods-eyebrow" style="margin-top:10px">Recent history</p>' + history_rows(table, method))


def screen_forecast():
    ui.page_header("Historical replay", "Forecast one hour",
                   "What would each model have forecast for one test hour, and how close was it to what was recorded?")
    ui.note(f"Historical replay over the {period_label('test')}. Models and preprocessing are fixed; "
            "inputs use only the target hour's calendar and completed earlier hours.")
    test_idx = CLEAN.index[CLEAN["period"] == "test"]
    with ui.card("controls", horizontal=True, vertical_alignment="bottom", gap="small"):
        day = st.date_input("Test date", value=pd.Timestamp("2012-08-01").date(), width=132,
                            min_value=test_idx.min().date(), max_value=test_idx.max().date())
        hours = test_idx[test_idx.normalize() == pd.Timestamp(day)]
        if len(hours) == 0:
            st.warning("No observed test hours on this date.")
            return
        hour_default = list(hours.hour).index(8) if 8 in list(hours.hour) else 0
        h = st.selectbox("Observed hour", list(hours.hour), index=hour_default, width=104,
                         format_func=lambda x: f"{x:02d}:00")
        method = st.segmented_control("Model", list(ui.MODEL_COLOURS), default=DEFAULT, required=True, key="fc_model", wrap=True,
                                      format_func=lambda m: ui.MODEL_SHORT[m] + (" (default)" if m == DEFAULT else ""))
        ts = pd.Timestamp(day) + pd.Timedelta(hours=int(h))
        key = (str(ts), method)
        if st.session_state.get("fc_key") != key:      # new selection clears old forecast/reveal
            st.session_state.update(fc_key=key, fc=None, revealed=False)
        if st.button("Forecast", type="primary"):
            st.session_state.fc = forecast_one(ts, method)
            st.session_state.revealed = False
        if st.button("Reveal recorded count", key="reveal_btn", disabled=st.session_state.fc is None):
            st.session_state.revealed = True

    if st.session_state.fc is None:
        with ui.card("waiting"):
            ui.show(f'<div class="dashed">Target hour {ts:%Y-%m-%d %H:%M}-{ts:%H}:59. Press Forecast.</div>')
        return
    raw, d = st.session_state.fc
    actual = int(CLEAN.loc[ts, "cnt"]) if st.session_state.revealed else None

    saved = R["pred"].query("period=='test' and model==@method and timestamp==@ts.strftime('%Y-%m-%d %H:%M')")
    diff = abs(saved["point_forecast"].iloc[0] - d["point"]) if len(saved) else None

    wide, narrow = st.columns([2, 1], gap="medium")
    with wide, ui.card("result"):
        ui.show(f'<div class="fc-when">{ts:%A %Y-%m-%d, %H:%M}-{ts:%H}:59</div>'
                f'<div class="fc-hero"><span class="fc-num">{d["point"]:.0f}</span><span class="fc-unit">rentals expected</span></div>'
                f'<div class="fc-exact">Point forecast {d["point"]:.1f} rentals in the hour</div>'
                f'<span class="chip">{ui.dot(method)}{ui.e(model_label(method))} · {ui.e(fmt_setting(method))}</span>')
        ui.note(f"Model reference: models/{method}.pkl; preprocessing: models/preprocessor_full.json (fitted on training only).")
        if method in M.HAS_INTERVALS:
            ui.show(ui.interval_bar(method, d, actual))
        else:
            ui.show('<div class="dashed">Intervals unavailable for this model</div>')
        show_interval_and_model_notes(d, method)
        if actual is None:
            ui.show('<div class="dashed">The recorded count stays hidden until you press Reveal.</div>')
        else:
            c = st.columns(3)
            c[0].metric("Recorded", actual)
            c[1].metric("Error (actual − forecast)", f"{actual - d['point']:+.1f}")
            c[2].metric("Absolute error", f"{abs(actual - d['point']):.1f}")
            if method in M.HAS_INTERVALS:
                in80, in95 = d["lo80"] <= actual <= d["hi80"], d["lo95"] <= actual <= d["hi95"]
                if in80 and in95:
                    ui.show(ui.pill("Inside the 80% and 95% intervals", "ok"))
                elif in95:
                    ui.show(ui.pill("Outside the 80% interval, inside the 95% interval", "plain"))
                elif in80:
                    ui.show(ui.pill("Inside the 80% interval, outside the 95% interval", "plain"))
                else:
                    ui.show(ui.pill("Outside the 80% and 95% intervals", "plain"))
    with narrow:
        uncertainty_card(d, method, diff)
    inputs_card(raw, ts, method)


def compare_table(ps: pd.DataFrame) -> str:
    rows, default_row = [], set()
    order = list(ui.MODEL_COLOURS)
    for i, m in enumerate(order):
        r = ps.loc[m]
        name = (f'<span style="white-space:nowrap">{ui.dot(m)}{ui.e(ui.MODEL_SHORT[m])}'
                + (" " + ui.pill("default", "navy") if m == DEFAULT else "") + "</span>")
        rows.append([name, ui.e(fmt_setting(m)), f"{cfg['selected'][m]['validation_mae']:.3f}", f"{r.mae:.2f}", f"{r.rmse:.2f}",
                     f"{r.train_time_s:.3f}"])
        if m == DEFAULT:
            default_row.add(i)
    return ui.table(["Model", "Selected setting", "Val MAE", "Test MAE", "Test RMSE", "Train (s)"], rows, numeric={2, 3, 4, 5},
                    highlight=default_row)


@st.cache_data(show_spinner=False)
def poisson_expected_score() -> float:
    """Average Poisson entropy of the saved test means: the expected score if outcomes really were Poisson."""
    mu = R["pred"].query("period=='test' and model=='poisson'")["raw_mean"].to_numpy()
    return float(stats.poisson.entropy(mu).mean())


def compare_forecasts_card():
    with ui.card("forecasts"):
        ui.heading("Forecasts against recorded counts")
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
        ui.note(f"MAE in range {pr['abs_error'].mean():.2f} over {len(pr)} hours ({rng[0]} to {rng[1]}); "
                "the scores above are for the full test period.")
        fig = go.Figure()
        if method in M.HAS_INTERVALS:
            fig.add_scatter(x=grid, y=s["hi80"], line=dict(width=0), showlegend=False, connectgaps=False, hoverinfo="skip")
            fig.add_scatter(x=grid, y=s["lo80"], fill="tonexty", line=dict(width=0), name="80% interval",
                            fillcolor=ui.INTERVAL_COLOURS[method][95], connectgaps=False, hoverinfo="skip")
        fig.add_scatter(x=grid, y=s["actual"], name="Recorded count", mode="lines", connectgaps=False,
                        line=dict(color=ui.ACTUAL_COLOUR, width=1.6))
        fig.add_scatter(x=grid, y=s["point_forecast"], name=f"{LBL[method]} forecast", mode="lines", connectgaps=False,
                        line=dict(color=ui.MODEL_COLOURS[method], width=1.8))
        fig.update_layout(xaxis_title="Hour start", yaxis_title="Rentals per hour", xaxis_tickformat="%a %d %b", xaxis_nticks=10)
        ui.chart(fig, 400)
        csv_button(pr.drop(columns=["ts"]), f"forecasts_{method}_{rng[0]}_{rng[1]}.csv", "Download displayed forecasts")


def screen_compare():
    ui.page_header("Model comparison", "Compare models", "Which of the five methods forecasts best on the frozen test period?")
    met = R["metrics"]
    raw_ps = met[met.table == "point_scores"].set_index("model")
    best, base = raw_ps.loc[DEFAULT], raw_ps.loc["baseline"]
    n_hours = int(raw_ps["n"].iloc[0]) if raw_ps["n"].nunique() == 1 else None
    hours_text = f", {n_hours:,} evaluated hours per model" if n_hours else ""
    default_name = LBL[DEFAULT].replace(" regression", "")
    ui.kpis([
        ui.kpi_html("Default model", default_name, f"lowest validation MAE {cfg['selected'][DEFAULT]['validation_mae']:.2f}", "navy"),
        ui.kpi_html("Test MAE", f"{best.mae:.2f}", "rentals per hour, full test period"),
        ui.kpi_html("Change vs seasonal baseline", f"{100 * (best.mae / base.mae - 1):+.1f}%", f"baseline test MAE {base.mae:.2f}", "green"),
        ui.kpi_html("Test RMSE", f"{best.rmse:.2f}", "rentals per hour, full test period"),
    ])
    left, right = st.columns([1, 2], gap="medium")
    with left, ui.card("mae_bars"):
        ui.heading("Test MAE by model", f"Full final test: {period_label('test')}.")
        order = raw_ps.sort_values("mae", ascending=False).index
        fig = go.Figure(go.Bar(x=raw_ps.loc[order, "mae"], y=[ui.MODEL_SHORT[m] for m in order], orientation="h",
                               marker_color=[ui.MODEL_COLOURS[m] for m in order], cliponaxis=False,
                               text=[f"{v:.2f}" for v in raw_ps.loc[order, "mae"]], textposition="outside"))
        fig.update_layout(xaxis_title="Test MAE (rentals per hour)", showlegend=False, margin=dict(r=48, t=8))
        ui.chart(fig, 440)
    with right, ui.card("scores"):
        ui.heading("Full final-test scores", f"{period_label('test')}{hours_text}. Val MAE: {period_label('validation')}.")
        ui.show(compare_table(raw_ps))
        ps = met[met.table == "point_scores"][["model", "n", "mae", "rmse", "train_time_s", "is_default"]].copy()
        ps["model"] = ps["model"].map(model_label)
        ps = ps.rename(columns={"n": "evaluated_hours", "mae": "MAE", "rmse": "RMSE", "train_time_s": "training_time_s"})
        csv_button(ps, "test_scores.csv")
        pl = met[met.table == "poisson_log_score"].iloc[0]
        ui.show(f'<div class="softbox"><p class="fods-eyebrow">Poisson probability score</p><div class="v">{pl["mean_log_loss"]:.2f} nats</div>'
                f'<p class="fods-sub">Mean -log Pr(Y = y | mu) over {int(pl["n"]):,} test hours; lower is better. If outcomes really '
                f'followed the fitted Poisson means the expected score would be {poisson_expected_score():.2f} nats. '
                'Discrete count-probability score; not comparable with a Gaussian density score.</p></div>')
    ui.note("Default = lowest validation MAE, chosen and frozen before test scoring. Tie rule: " + cfg["tie_rule"])
    ui.note("Training time: " + cfg["training_time_definition"] + ".")
    compare_forecasts_card()
    with st.expander("All validation candidates"):
        v = R["val"][R["val"].experiment == "principal"]
        st.dataframe(v, hide_index=True)
        csv_button(v, "validation_candidates.csv")


def tab_coverage(met):
    tiles = []
    for m in ("bayes", "poisson"):
        for lvl in (80, 95):
            r = met[(met.table == "interval_coverage") & (met.model == m) & (met.level == lvl) & (met.group_type == "all")].iloc[0]
            track = (f'<div class="track"><i style="width:{100 * r.coverage:.1f}%;background:{ui.MODEL_COLOURS[m]}"></i>'
                     f'<b style="left:{lvl}%"></b></div>')
            tiles.append(ui.kpi_html(f"{ui.dot(m)}{ui.MODEL_SHORT[m]} {lvl}%", f"{100 * r.coverage:.1f}%",
                                     f"nominal {lvl}% · mean width {r.mean_width:.1f} rentals", extra=track))
    ui.kpis(tiles)
    with ui.card("cov_chart"):
        ui.heading("Observed 80% coverage by hour", f"{period_label('test')}.")
        h80 = met[(met.table == "interval_coverage") & (met.level == 80) & (met.group_type == "hour")].copy()
        h80["hour"] = h80["group"].astype(float).astype(int)
        fig = go.Figure()
        for m, symbol in (("bayes", "circle"), ("poisson", "square")):
            part = h80[h80.model == m].sort_values("hour")
            fig.add_scatter(x=part["hour"], y=100 * part["coverage"], name=ui.MODEL_SHORT[m], mode="lines+markers",
                            line=dict(color=ui.MODEL_COLOURS[m]), marker=dict(symbol=symbol, size=8))
        fig.add_scatter(x=[-0.5, 23.5], y=[80, 80], name="nominal 80%", mode="lines", line=dict(color=ui.INK, dash="dash", width=1.2))
        fig.update_layout(xaxis_title="Hour of day", yaxis_title="Observed coverage (%)", yaxis_range=[0, 104], xaxis_dtick=2)
        ui.chart(fig, 340)
    with ui.card("cov_tables"):
        ui.heading("Coverage tables")
        c = st.columns(2)
        m = c[0].selectbox("Model", ["poisson", "bayes"], format_func=model_label, key="cov_m")
        lvl = c[1].selectbox("Nominal level", [80, 95], format_func=lambda x: f"{x}%")
        cv = met[(met.table == "interval_coverage") & (met.model == m) & (met.level == lvl)].copy()
        allr = cv[cv.group_type == "all"].iloc[0]
        ui.kpis([ui.kpi_html("Observed coverage (full test)", f"{100 * allr.coverage:.1f}%"), ui.kpi_html("Nominal", f"{lvl}%"),
                 ui.kpi_html("Mean width (rentals)", f"{allr.mean_width:.1f}")])
        ui.note(f"{period_label('test')}, n={int(allr.n)}. Endpoints inclusive; raw Gaussian endpoints "
                "(including negative ones) are used as saved.")
        for gt, title in [("hour", "By hour of day"), ("workingday", "By working-day status")]:
            g = cv[cv.group_type == gt].copy()
            g["observed_coverage_%"] = (100 * g.coverage).round(1)
            g["nominal_%"] = lvl
            g = g.rename(columns={"group": gt, "n": "n_hours", "mean_width": "mean_width_rentals"})
            g.loc[g.n_hours == 0, ["observed_coverage_%", "mean_width_rentals"]] = np.nan
            ui.show(f'<h3 class="fods-h2" style="font-size:1.05rem">{title}</h3>')
            st.dataframe(g[[gt, "n_hours", "observed_coverage_%", "nominal_%", "mean_width_rentals"]],
                         hide_index=True, height=300 if gt == "hour" else 110)
        csv_button(cv, f"interval_coverage_{m}_{lvl}.csv")


def tab_residuals(met):
    rb = met[met.table == "residual_by_hour"].copy()
    rb["hour"] = rb.group.astype(int)
    with ui.card("residuals"):
        ui.heading("Mean residual by hour", f"{model_label(DEFAULT)}, {period_label('test')}.")
        fig = go.Figure(go.Bar(x=rb["hour"], y=rb["mean_residual"],
                               marker_color=[ui.COPPER if v > 0 else ui.NAVY for v in rb["mean_residual"]]))
        fig.add_hline(y=0, line_color=ui.INK, line_width=1)
        fig.update_layout(xaxis_title="Hour of day", yaxis_title="Mean signed residual (actual - forecast)", xaxis_dtick=2, showlegend=False)
        ui.chart(fig, 360)
        fmt = lambda r: f"{int(r.hour):02d}:00 ({r.mean_residual:+.1f})"  # noqa: E731
        under = ", ".join(fmt(r) for r in rb.nlargest(2, "mean_residual").itertuples())
        over = ", ".join(fmt(r) for r in rb.nsmallest(2, "mean_residual").itertuples())
        ui.note(f"Largest underprediction (copper): {under}. Largest overprediction (navy): {over}.")
        ui.note("Positive = underprediction on average. Signed averages: opposing errors can cancel.")
        st.dataframe(rb[["hour", "mean_residual", "n"]], hide_index=True, height=300)
        csv_button(rb[["hour", "mean_residual", "n", "model", "period"]], "residuals_by_hour.csv")


def tab_largest():
    le = R["largest"].copy()
    ui.note(f"Default model ({model_label(DEFAULT)}), {period_label('test')}. Ten largest absolute errors; ties ordered by earlier timestamp.")
    left, right = st.columns([3, 2], gap="medium")
    with left, ui.card("largest"):
        ui.heading("Largest errors", "Select a row to open that hour; the largest error is shown first.")
        event = st.dataframe(le[["timestamp", "model", "actual", "point_forecast", "residual", "abs_error"]], hide_index=True,
                             on_select="rerun", selection_mode="single-row", key="largest_tbl")
        csv_button(le, "largest_errors.csv")
    rows = event.selection.rows
    ts = pd.Timestamp(le.timestamp.iloc[rows[0] if rows else 0])
    raw, d = forecast_one(ts, DEFAULT)
    recorded = int(CLEAN.loc[ts, "cnt"])
    with right, ui.card("largest_detail"):
        ui.heading(f"{ts:%A %Y-%m-%d %H:%M}", eyebrow="Selected hour")
        pairs = [("Point forecast", f"{d['point']:.1f}"), ("Recorded count", str(recorded)),
                 ("Signed error (actual - forecast)", f"{recorded - d['point']:+.1f}"),
                 ("80% interval", f"[{d['lo80']:.1f}, {d['hi80']:.1f}]" if DEFAULT in M.HAS_INTERVALS else "unavailable"),
                 ("95% interval", f"[{d['lo95']:.1f}, {d['hi95']:.1f}]" if DEFAULT in M.HAS_INTERVALS else "unavailable")]
        if DEFAULT == "bayes":
            pairs += [("Observation noise σ²", f"{d['noise_var']:.3f}"), ("Coefficient variance φᵀSφ", f"{d['coef_var']:.3f}"),
                      ("Predictive standard deviation", f"{np.sqrt(d['total_var']):.3f}")]
        ui.show(ui.kv(pairs))
        show_interval_and_model_notes(d, DEFAULT)
        ui.show('<p class="fods-eyebrow" style="margin-top:8px">Inputs used</p>')
        st.dataframe(inputs_table(raw, DEFAULT), hide_index=True, height=300)


def tab_regularisation(val):
    lv = val[(val.experiment == "principal") & (val.method == "lasso")].copy()
    lv["alpha"] = lv["setting"].str.replace("alpha=", "").astype(float)
    sel_a = cfg["selected"]["lasso"]["settings"]["alpha"]
    ui.note(f"Lasso fits on the {period_label('train')}; scores on the {period_label('validation')}. "
            f"Zero tolerance |coef| <= {cfg['lasso_zero_tolerance']:g}. Selected penalty: {sel_a:g}.")
    colour = ui.MODEL_COLOURS["lasso"]
    with ui.card("reg_table"):
        ui.heading("Each penalty tested", f"{int(lv['n_hours'].iloc[0]):,} validation hours per penalty.")
        rows, selected_row = [], set()
        for i, r in enumerate(lv.itertuples()):
            tag = " " + ui.pill("selected", "navy") if r.alpha == sel_a else ""
            if r.alpha == sel_a:
                selected_row.add(i)
            rows.append([f"{r.alpha:g}{tag}", ui.mini_bar(f"{r.validation_mae:.2f}", r.validation_mae / lv.validation_mae.max(), colour),
                         ui.mini_bar(f"{int(r.n_nonzero_coef)}", r.n_nonzero_coef / r.n_coef_total, colour), str(int(r.n_coef_total))])
        ui.show(ui.table(["Penalty", "Validation MAE", "Non-zero coefficients", "Of"], rows, highlight=selected_row))
    c = st.columns(2, gap="medium")
    for col, y, title, name in [(c[0], "validation_mae", "Validation MAE", "reg_mae"), (c[1], "n_nonzero_coef", "Non-zero coefficients", "reg_nz")]:
        with col, ui.card(name):
            ui.heading(f"{title} vs Lasso penalty")
            fig = go.Figure(go.Scatter(x=lv["alpha"], y=lv[y], mode="lines+markers", line=dict(color=colour), showlegend=False))
            fig.add_vline(x=sel_a, line_dash="dash", line_color=ui.COPPER, annotation_text="selected")
            fig.update_layout(xaxis_type="log", xaxis_title="Lasso penalty alpha (log scale)", yaxis_title=title)
            ui.chart(fig, 300)
    with ui.card("reg_coef"):
        ui.heading("Coefficients by penalty")
        co = R["lasso"]
        penalties = sorted(co.alpha.unique())
        a = st.selectbox("Penalty for coefficient table", penalties, index=penalties.index(sel_a), format_func=lambda x: f"{x:g}")
        t = co[co.alpha == a]
        ui.kpis([ui.kpi_html("Intercept", f"{t[t.feature == '(intercept)'].coefficient.iloc[0]:.3f}")])
        st.dataframe(t[t.feature != "(intercept)"][["feature", "coefficient", "is_zero"]], hide_index=True, height=300)
        piv = co[co.feature != "(intercept)"].pivot(index="feature", columns="alpha", values="coefficient")
        piv = piv.reindex(PRE.feature_names)
        ui.show('<h3 class="fods-h2" style="font-size:1.05rem">Coefficients across tested penalties</h3>'
                '<p class="fods-note">0 = removed by this fit.</p>')
        st.dataframe(piv.round(3), height=320)
        csv_button(co, "lasso_coefficients.csv")


def tab_history(val):
    rows = []
    for exp, name, desc in [("history_calendar_only", "Calendar only", "hour, weekday, month (one-hot), holiday, workingday"),
                            ("history_full_inputs", "Full required input set",
                             "calendar + 3 count lags + previous-24h mean + previous-hour weather + 7 missing flags")]:
        r = val[(val.experiment == exp) & val.selected].iloc[0]
        rows.append(dict(input_set=name, inputs=desc, selected_penalty=r.setting, validation_hours=r.n_hours,
                         validation_MAE=r.validation_mae))
    h = pd.DataFrame(rows)
    diff = h.validation_MAE[0] - h.validation_MAE[1]
    c = st.columns(2, gap="medium")
    for col, r, tone in ((c[0], h.iloc[0], ""), (c[1], h.iloc[1], "navy")):
        col.html(f'<div class="bign {tone}"><p class="fods-eyebrow">{ui.e(r.input_set)}</p><div class="v">{r.validation_MAE:.2f}</div>'
                 f'<p class="fods-sub">Validation MAE (rentals per hour), {int(r.validation_hours):,} hours, Ridge {ui.e(r.selected_penalty)}.</p>'
                 f'<p class="fods-sub">{ui.e(r.inputs)}</p></div>')
    ui.kpis([ui.kpi_html("Calendar-only MAE minus full-input MAE", f"{diff:+.2f}",
                         "Positive: the full-input Ridge had lower validation error.", "green")])
    ui.note("Each input set tuned separately on validation. Rental history and weather are added together, "
            "so this does not identify which of them produced the gain, nor any causal effect.")
    hv = val[val.experiment.str.startswith("history")]
    with st.expander("All candidates"):
        st.dataframe(hv, hide_index=True)
    csv_button(hv, "history_experiment.csv")


def screen_behaviour():
    ui.page_header("Diagnostics", "Inspect model behaviour", "Where do the forecasts and their intervals go wrong, and what did each modelling choice change?")
    tabs = st.tabs(["Interval coverage", "Residuals by hour", "Largest errors", "Regularisation", "History experiment"])
    val, met = R["val"], R["metrics"]
    with tabs[0]:
        tab_coverage(met)
    with tabs[1]:
        tab_residuals(met)
    with tabs[2]:
        tab_largest()
    with tabs[3]:
        tab_regularisation(val)
    with tabs[4]:
        tab_history(val)


SCREENS = {"Data overview": screen_overview, "Forecast one hour": screen_forecast,
           "Compare models": screen_compare, "Inspect model behaviour": screen_behaviour}
NAV_LABELS = {"Data overview": ":material/table_chart: Data overview", "Forecast one hour": ":material/schedule: Forecast one hour",
              "Compare models": ":material/leaderboard: Compare models", "Inspect model behaviour": ":material/troubleshoot: Inspect behaviour"}
ui.inject_css()
st.sidebar.html('<div class="side-eyebrow">FoDS · Option B</div><div class="side-title">Bike-Rental Forecasts</div>')
choice = st.sidebar.radio("Screen", list(SCREENS), format_func=NAV_LABELS.get, label_visibility="collapsed", width="stretch")
st.sidebar.html(f'<div class="side-foot"><div class="h">Default model</div><b>{ui.dot(DEFAULT)}{ui.e(LBL[DEFAULT])}</b>'
                f'<div class="h">Train</div>{P["train"][0]} to {P["train"][1]}'
                f'<div class="h">Validation</div>{P["validation"][0]} to {P["validation"][1]}'
                f'<div class="h">Test</div>{P["test"][0]} to {P["test"][1]}</div>')
SCREENS[choice]()
