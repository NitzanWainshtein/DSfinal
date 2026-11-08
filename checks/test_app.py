"""Run the dashboard headlessly: forecast, reveal, switch, inspect, and agreement with saved results."""
import os
import re

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _app():
    return AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=180).run()


def metrics(at):
    return {m.label: m.value for m in at.metric}


def page_html(at):
    return " ".join(h.value for h in at.html)


@pytest.mark.parametrize("screen", ["Data overview", "Forecast one hour", "Compare models", "Inspect model behaviour"])
def test_screens_render_without_errors(screen):
    at = _app()
    at.sidebar.radio[0].set_value(screen).run()
    assert not at.exception and not at.error


def test_forecast_reveal_and_agreement():
    at = _app()
    at.sidebar.radio[0].set_value("Forecast one hour").run()
    at.button[0].click().run()                     # Forecast (default model, 2012-08-01 08:00)
    m, page = metrics(at), page_html(at)
    assert "Recorded" not in m and "681 recorded" not in page   # outcome hidden until Reveal
    pred = pd.read_csv(os.path.join(ROOT, "outputs", "predictions.csv"))
    model = at.segmented_control[0].value
    saved = pred.query("period=='test' and model==@model and timestamp=='2012-08-01 08:00'").iloc[0]
    shown = re.search(r"Point forecast ([\d.]+) rentals in the hour", page).group(1)
    assert float(shown) == pytest.approx(saved.point_forecast, abs=0.05)
    at.button[1].click().run()                     # Reveal
    m = metrics(at)
    assert m["Recorded"] == "681" and "681 recorded" in page_html(at)
    assert float(m["Error (actual − forecast)"]) == pytest.approx(681 - saved.point_forecast, abs=0.05)


def test_switching_model_clears_forecast_and_baseline_has_no_intervals():
    at = _app()
    at.sidebar.radio[0].set_value("Forecast one hour").run()
    at.button[0].click().run()
    at.button[1].click().run()
    at.segmented_control[0].set_value("baseline").run()
    assert "Recorded" not in metrics(at) and "rentals expected" not in page_html(at)
    at.button[0].click().run()
    page = page_html(at)
    assert "Intervals unavailable for this model" in page and "95% interval [" not in page
