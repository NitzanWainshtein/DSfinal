"""Presentation helpers for the dashboard: design tokens, CSS, chart template, HTML fragments.

Nothing here computes a model result; every value shown is passed in by app.py.
"""
from __future__ import annotations

import html
import math

import plotly.graph_objects as go
import streamlit as st

INK = "#1C2B36"
MUTED = "#4F5B65"
CREAM = "#F4F1EA"
CARD = "#FFFFFF"
BORDER = "#E3DDD1"
GRID = "#ECE6DA"
NAVY = "#1B3A4B"
COPPER = "#B8652A"
COPPER_TEXT = "#9A4F22"
SUCCESS = "#1E6A45"
SUCCESS_BG = "#E4F2EA"
AMBER = "#8A5A00"
AMBER_BG = "#FBEFD5"
COPPER_BG = "#F8EDE3"

# One fixed colour per model, everywhere. Order = order of the model selector.
MODEL_COLOURS = {
    "bayes": "#2F6690",
    "ridge": "#3A8D7F",
    "lasso": "#7B5EA7",
    "poisson": "#D9822B",
    "baseline": "#9AA1A8",
}
MODEL_SHORT = {"bayes": "Bayesian", "ridge": "Ridge", "lasso": "Lasso", "poisson": "Poisson", "baseline": "Baseline"}
ACTUAL_COLOUR = INK
INTERVAL_COLOURS = {
    "bayes": {80: "#7FA8C9", 95: "#CFE0EE"},
    "poisson": {80: "#EBB376", 95: "#F7E2CB"},
}
SERIF = "'Source Serif 4', serif"

_CSS = """
.block-container{max-width:1320px;padding-top:4.5rem;padding-bottom:4rem}
@media (min-width:641px){
 section[data-testid="stSidebar"]{width:260px!important;min-width:260px!important}
 .block-container{padding-left:2.5rem;padding-right:2.5rem}
}
div[class*="st-key-card"]{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:28px}
.fods-eyebrow{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.14em;color:var(--copper-text);margin:0 0 6px}
.fods-h1{font-family:var(--serif);font-weight:600;font-size:2.5rem;line-height:1.15;color:var(--ink);margin:0 0 8px}
.fods-intro{color:var(--muted);font-size:1.05rem;line-height:1.5;margin:0;max-width:62rem}
.fods-h2{font-family:var(--serif);font-weight:600;font-size:1.4rem;line-height:1.25;color:var(--ink);margin:0 0 4px}
.fods-sub{color:var(--muted);font-size:.875rem;line-height:1.5;margin:0}
.fods-note{color:var(--muted);font-size:.8125rem;line-height:1.5;margin:0}
.fods-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;vertical-align:baseline}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:20px 22px;min-width:0}
.kpi .l{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.kpi .v{font-family:var(--serif);font-weight:600;font-size:2.05rem;line-height:1.2;color:var(--ink);margin-top:6px;overflow-wrap:anywhere}
.kpi .s{font-size:.8125rem;line-height:1.45;color:var(--muted);margin-top:4px}
.kpi.navy{background:var(--navy);border-color:var(--navy)}
.kpi.navy .l,.kpi.navy .s{color:#D5DEE3}.kpi.navy .v{color:#fff}
.kpi.copper{background:var(--copper-bg);border-color:#EBD8C6}.kpi.copper .l{color:var(--copper-text)}
.kpi.green .v{color:var(--success)}
.track{position:relative;height:6px;border-radius:3px;background:var(--grid);margin:12px 0 4px}
.track i{position:absolute;left:0;top:0;bottom:0;border-radius:3px}
.track b{position:absolute;top:-4px;width:2px;height:14px;background:var(--ink)}
.pill{display:inline-block;font-size:12px;font-weight:600;line-height:1.5;border-radius:12px;padding:2px 10px}
.pill.ok{color:var(--success);background:var(--success-bg)}
.pill.warn{color:var(--amber);background:var(--amber-bg)}
.pill.plain{color:var(--muted);background:var(--cream)}
.pill.navy{color:#fff;background:var(--navy)}
.pill.copper{color:var(--copper-text);background:var(--copper-bg)}
.chip{display:inline-block;font-size:.8125rem;color:var(--ink);background:var(--cream);border-radius:10px;padding:5px 12px;line-height:1.45}
.fc-when{color:var(--muted);font-size:.9375rem}
.fc-hero{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px;margin:2px 0 2px}
.fc-num{font-family:var(--serif);font-weight:600;font-size:5rem;line-height:1.05;color:var(--ink)}
.fc-unit{font-size:1.05rem;color:var(--muted)}
.fc-exact{color:var(--muted);font-size:.8125rem;margin-bottom:10px}
.dashed{border:1.5px dashed #CFC6B5;border-radius:12px;padding:18px 20px;color:var(--muted);font-size:.9375rem;text-align:center}
.ib{margin:6px 0 2px}
.ib-row{position:relative;height:20px;font-size:12px;font-weight:600;color:var(--ink)}
.ib-row span{position:absolute;transform:translateX(-50%);white-space:nowrap}
.ib-row.rec span{color:var(--copper-text)}
.ib-track{position:relative;height:26px;background:var(--cream);border-radius:6px}
.ib-track i{position:absolute;top:0;bottom:0}
.ib-track .tick{position:absolute;top:-4px;bottom:-4px;width:3px;margin-left:-1.5px;background:var(--navy);border-radius:2px}
.ib-track .dia{position:absolute;top:50%;width:13px;height:13px;margin:-6.5px 0 0 -6.5px;background:var(--copper);transform:rotate(45deg);border:1.5px solid #fff}
.ib-axis{position:relative;height:18px;margin-top:2px;font-size:11px;color:var(--muted)}
.ib-axis span{position:absolute;transform:translateX(-50%)}
.ib-legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12px;color:var(--muted);margin-top:6px}
.ib-legend em{display:inline-block;width:14px;height:10px;border-radius:2px;margin-right:6px;font-style:normal}
.stack{display:flex;gap:2px;height:16px;border-radius:5px;overflow:hidden;margin:10px 0}
.stack i{display:block;min-width:4px}
.kv{display:grid;grid-template-columns:1fr auto;gap:6px 12px;font-size:.875rem}
.kv .n{color:var(--muted)}.kv .x{font-weight:600;color:var(--ink);text-align:right;font-variant-numeric:tabular-nums}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.tile{background:var(--cream);border-radius:10px;padding:10px 14px;min-width:0}
.tile .l{font-size:12px;color:var(--muted)}.tile .v{font-weight:600;font-size:1.05rem;color:var(--ink)}.tile .s{font-size:12px;color:var(--muted)}
.tbl-wrap{overflow-x:auto}
.tbl{width:100%;border-collapse:collapse;font-size:.875rem}
.tbl th{text-align:left;font-size:12px;font-weight:600;color:var(--muted);padding:8px 10px;border-bottom:1px solid var(--border);white-space:nowrap}
.tbl td{padding:9px 10px;border-bottom:1px solid var(--grid);color:var(--ink);vertical-align:top}
.tbl td.num,.tbl th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tbl tr.hl td{background:#EEF3F6;font-weight:600}
.mini{display:flex;align-items:center;gap:8px;min-width:130px}.mini span{width:46px;text-align:right;font-variant-numeric:tabular-nums}
.mini div{flex:1;height:8px;background:var(--grid);border-radius:4px;overflow:hidden}.mini i{display:block;height:100%}
.softbox{background:var(--copper-bg);border-radius:12px;padding:16px 18px}
.softbox .v{font-family:var(--serif);font-weight:600;font-size:1.75rem;color:var(--ink)}
.bign{border-radius:16px;padding:24px;border:1px solid var(--border);background:var(--card);height:100%}
.bign .v{font-family:var(--serif);font-weight:600;font-size:3.25rem;line-height:1.1;color:var(--ink)}
.bign.navy{background:var(--navy);border-color:var(--navy)}.bign.navy .v{color:#fff}.bign.navy .fods-sub,.bign.navy .fods-eyebrow{color:#D5DEE3}
.side-eyebrow{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.14em;color:#E0A77C}
.side-title{font-family:var(--serif);font-weight:600;font-size:1.5rem;line-height:1.2;color:#fff;margin:4px 0 8px}
.side-foot{font-size:.8125rem;line-height:1.55;color:#D5DEE3;border-top:1px solid #2E5468;padding-top:14px}
.side-foot b{color:#fff;font-weight:600}.side-foot .h{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:#E0A77C;margin:10px 0 2px}
section[data-testid="stSidebar"] [role="radiogroup"]{gap:2px;width:100%}
section[data-testid="stSidebar"] [role="radiogroup"] label{width:100%;padding:8px 10px;border-radius:10px;margin:0;cursor:pointer}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]>div>div:first-child:not([data-testid="stMarkdownContainer"]){display:none}
[data-testid="stSidebarUserContent"]{display:flex;flex-direction:column;min-height:calc(100dvh - 7rem)}
[data-testid="stSidebarUserContent"]>div{flex:1;display:flex;flex-direction:column}
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"]{flex:1}
[data-testid="stSidebarUserContent"] [data-testid="stElementContainer"]:last-child{margin-top:auto}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#234B60}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:#2C5A72}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{font-weight:600;color:#fff}
.st-key-reveal_btn button{background:#fff;border:1.5px solid var(--copper);color:var(--copper-text)}
.st-key-reveal_btn button:hover{background:var(--copper-bg);border-color:var(--copper);color:var(--copper-text)}
.st-key-reveal_btn button:disabled{border-color:var(--border);color:#A7A196;background:#fff}
[data-testid="stMetric"]{background:var(--cream);border-radius:10px;padding:12px 16px}
[data-testid="stMetricValue"]{font-family:var(--serif);font-weight:600}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *{white-space:normal!important;overflow:visible!important;text-overflow:clip!important}
.st-key-fc_model button{padding-left:8px;padding-right:8px;min-height:40px}
[data-testid="stBaseButton-primary"]:hover,[data-testid="stBaseButton-primary"]:focus{background:#234B60;border-color:#234B60;color:#fff}
@media (max-width:640px){
 div[class*="st-key-card"]{padding:18px}
 .fods-h1{font-size:2rem}.fc-num{font-size:4rem}
 .kpis{grid-template-columns:1fr 1fr;gap:10px}.kpi{padding:14px}.kpi .v{font-size:1.6rem}
 .bign .v{font-size:2.5rem}
 .tbl th,.tbl td{padding:8px 5px;font-size:.8125rem}.pill{font-size:11px;padding:2px 7px}.mini{min-width:96px}
}
"""


def inject_css() -> None:
    """Add the stylesheet for this run (colours come from the tokens above)."""
    tokens = {"ink": INK, "muted": MUTED, "cream": CREAM, "card": CARD, "border": BORDER, "grid": GRID, "navy": NAVY,
              "copper": COPPER, "copper-text": COPPER_TEXT, "copper-bg": COPPER_BG, "success": SUCCESS,
              "success-bg": SUCCESS_BG, "amber": AMBER, "amber-bg": AMBER_BG, "serif": SERIF}
    root = ":root{" + "".join(f"--{k}:{v};" for k, v in tokens.items()) + "}"
    dots = "".join(
        f'.st-key-fc_model button:nth-of-type({i})::before{{content:"";display:inline-block;width:10px;height:10px;'
        f"border-radius:50%;margin-right:7px;background:{c}}}"
        for i, c in enumerate(MODEL_COLOURS.values(), start=1))
    st.html(f"<style>{root}{_CSS}{dots}</style>")


def show(fragment: str) -> None:
    st.html(fragment)


def e(text) -> str:
    return html.escape(str(text))


def dot(model: str) -> str:
    return f'<span class="fods-dot" style="background:{MODEL_COLOURS[model]}"></span>'


def page_header(eyebrow: str, title: str, intro: str) -> None:
    show(f'<p class="fods-eyebrow">{e(eyebrow)}</p><h1 class="fods-h1">{e(title)}</h1><p class="fods-intro">{e(intro)}</p>')


def card(name: str, **kwargs):
    """White rounded card; style comes from the st-key-card* CSS rule."""
    return st.container(key=f"card_{name}", **kwargs)


def heading(title: str, sub: str = "", eyebrow: str = "", right: str = "") -> None:
    top = f'<p class="fods-eyebrow">{e(eyebrow)}</p>' if eyebrow else ""
    sub_html = f'<p class="fods-sub">{e(sub)}</p>' if sub else ""
    pill_html = f"<div>{right}</div>" if right else ""
    show(f'{top}<div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;align-items:baseline">'
         f'<h2 class="fods-h2">{e(title)}</h2>{pill_html}</div>{sub_html}')


def note(text: str) -> None:
    show(f'<p class="fods-note">{e(text)}</p>')


def pill(text: str, kind: str = "plain") -> str:
    return f'<span class="pill {kind}">{e(text)}</span>'


def kpi_html(label: str, value: str, sub: str = "", tone: str = "", extra: str = "") -> str:
    sub_html = f'<div class="s">{e(sub)}</div>' if sub else ""
    return f'<div class="kpi {tone}"><div class="l">{label}</div><div class="v">{e(value)}</div>{extra}{sub_html}</div>'


def kpis(items: list[str]) -> None:
    show(f'<div class="kpis">{"".join(items)}</div>')


def table(headers: list[str], rows: list[list[str]], numeric: set[int] = frozenset(), highlight: set[int] = frozenset()) -> str:
    """HTML table; cells are ready-made HTML, so callers escape free text with e()."""
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{h}</th>' for i, h in enumerate(headers))
    body = ""
    for r, row in enumerate(rows):
        cls = ' class="hl"' if r in highlight else ""
        body += f"<tr{cls}>" + "".join(f'<td class="{"num" if i in numeric else ""}">{c}</td>' for i, c in enumerate(row)) + "</tr>"
    return f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def mini_bar(label: str, fraction: float, colour: str) -> str:
    pct = max(0.0, min(1.0, fraction)) * 100
    return f'<div class="mini"><span>{e(label)}</span><div><i style="width:{pct:.1f}%;background:{colour}"></i></div></div>'


def kv(pairs: list[tuple[str, str]]) -> str:
    return '<div class="kv">' + "".join(f'<div class="n">{e(k)}</div><div class="x">{e(v)}</div>' for k, v in pairs) + "</div>"


def _nice_ticks(low: float, high: float, count: int = 6) -> list[float]:
    raw = (high - low) / max(count - 1, 1)
    power = 10 ** math.floor(math.log10(raw))
    step = next((m * power for m in (1, 2, 2.5, 5, 10) if m * power >= raw), 10 * power)
    start = math.ceil(low / step) * step
    ticks = []
    value = start
    while value <= high + 1e-9:
        ticks.append(value)
        value += step
    return ticks


def interval_bar(model: str, d: dict, actual: int | None = None) -> str:
    """Horizontal bar: 95% band, 80% band, point-forecast tick and (after Reveal) the recorded count."""
    values = [d["lo95"], d["hi95"], d["point"]] + ([] if actual is None else [actual])
    low, high = min(values), max(values)
    pad = max((high - low) * 0.08, 1.0)
    low, high = low - pad, high + pad

    def pos(x: float) -> float:
        return 100 * (x - low) / (high - low)

    def label_pos(x: float) -> float:
        return min(93.0, max(7.0, pos(x)))

    colours = INTERVAL_COLOURS[model]
    band = lambda a, b, c: f'<i style="left:{pos(a):.2f}%;width:{pos(b) - pos(a):.2f}%;background:{c}"></i>'  # noqa: E731
    track = band(d["lo95"], d["hi95"], colours[95]) + band(d["lo80"], d["hi80"], colours[80])
    track += f'<div class="tick" style="left:{pos(d["point"]):.2f}%"></div>'
    below, recorded_legend = "", ""
    if actual is not None:
        track += f'<div class="dia" style="left:{pos(actual):.2f}%"></div>'
        below = f'<div class="ib-row rec"><span style="left:{label_pos(actual):.2f}%">{actual} recorded</span></div>'
        recorded_legend = f'<span><em style="background:{COPPER};width:10px"></em>Recorded count</span>'
    axis = "".join(f'<span style="left:{pos(t):.2f}%">{t:g}</span>' for t in _nice_ticks(low, high) if 4 <= pos(t) <= 96)
    legend = (f'<span><em style="background:{colours[95]}"></em>95% interval [{d["lo95"]:.1f}, {d["hi95"]:.1f}]</span>'
              f'<span><em style="background:{colours[80]}"></em>80% interval [{d["lo80"]:.1f}, {d["hi80"]:.1f}]</span>'
              f'<span><em style="background:{NAVY};width:4px"></em>Point forecast</span>{recorded_legend}')
    return (f'<div class="ib"><div class="ib-row"><span style="left:{label_pos(d["point"]):.2f}%">{d["point"]:.1f}</span></div>'
            f'<div class="ib-track">{track}</div>{below}<div class="ib-axis">{axis}</div>'
            f'<div class="ib-legend">{legend}</div></div>')


_TEMPLATE = go.layout.Template(layout=dict(
    font=dict(family="Inter, sans-serif", size=12, color=INK),
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    colorway=list(MODEL_COLOURS.values()),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER, automargin=True),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER, automargin=True),
    legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=8, r=12, t=32, b=8),
    hoverlabel=dict(font=dict(family="Inter, sans-serif", size=12)),
))


def chart(fig: go.Figure, height: int = 340, key: str | None = None) -> None:
    fig.update_layout(template=_TEMPLATE, height=height)
    st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displayModeBar": False})
