"""Run the dashboard headlessly: forecast, reveal, switch, inspect, and agreement with saved results."""
import os

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _app():
    return AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=180).run()


def metrics(at):
    return {m.label: m.value for m in at.metric}


@pytest.mark.parametrize("screen", ["Data overview", "Forecast one hour", "Compare models", "Inspect model behaviour"])
def test_screens_render_without_errors(screen):
    at = _app()
    at.sidebar.radio[0].set_value(screen).run()
    assert not at.exception and not at.error


def test_forecast_reveal_and_agreement():
    at = _app()
    at.sidebar.radio[0].set_value("Forecast one hour").run()
    at.button[0].click().run()                     # Forecast (default model, 2012-08-01 08:00)
    m = metrics(at)
    assert "Recorded count" not in m               # outcome hidden until Reveal
    pred = pd.read_csv(os.path.join(ROOT, "outputs", "predictions.csv"))
    model = at.selectbox[1].value
    saved = pred.query("period=='test' and model==@model and timestamp=='2012-08-01 08:00'").iloc[0]
    assert float(m["Point forecast (rentals in the hour)"]) == pytest.approx(saved.point_forecast, abs=0.05)
    at.button[1].click().run()                     # Reveal
    m = metrics(at)
    assert m["Recorded count"] == "681"
    assert float(m["Signed error (actual - forecast)"]) == pytest.approx(681 - saved.point_forecast, abs=0.05)


def test_switching_model_clears_forecast_and_baseline_has_no_intervals():
    at = _app()
    at.sidebar.radio[0].set_value("Forecast one hour").run()
    at.button[0].click().run()
    at.button[1].click().run()
    at.selectbox[1].set_value("baseline").run()
    assert "Recorded count" not in metrics(at) and "Point forecast (rentals in the hour)" not in metrics(at)
    at.button[0].click().run()
    m = metrics(at)
    assert m["80% interval"] == "unavailable" and m["95% interval"] == "unavailable"
