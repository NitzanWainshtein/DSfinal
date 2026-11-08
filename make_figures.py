"""Create the report figures from the saved results.

Reads only config.json, outputs/*.csv, outputs/*.json and data/hour.csv. It does
not import the modelling code and does not refit anything. Each figure is written
to docs/figures/ as a 200 dpi PNG.

Usage: python make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "docs" / "figures"

DPI = 200
FIG_SIZE = (7.0, 3.8)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

MODEL_LABELS = {
    "baseline": "Seasonal baseline",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "poisson": "Poisson",
    "bayes": "Bayesian",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": SECONDARY_INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": SECONDARY_INK,
        "ytick.color": SECONDARY_INK,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def load_config() -> dict:
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def load_metrics(table: str) -> pd.DataFrame:
    metrics = pd.read_csv(OUTPUTS / "metrics.csv")
    return metrics[metrics["table"] == table].copy()


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT).as_posix()}")


def hour_axis(ax: plt.Axes) -> None:
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.7, 23.7)
    ax.set_xlabel("Hour of day (start of hour, 0-23)")


def fig_hourly_pattern(config: dict) -> None:
    raw = pd.read_csv(
        ROOT / config["data_file"],
        usecols=["dteday", "hr", "workingday", "cnt"],
        parse_dates=["dteday"],
    )
    start, end = pd.to_datetime(config["periods"]["train"])
    train = raw[raw["dteday"].between(start, end)]
    means = train.groupby(["workingday", "hr"])["cnt"].mean().unstack("workingday")

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(means.index, means[1], color=BLUE, marker="o", ms=3.5, label="Working days")
    ax.plot(means.index, means[0], color=ORANGE, marker="s", ms=3.5, label="Weekends/holidays")
    ax.set_title(
        f"Average hourly rentals by hour of day, training period "
        f"({start:%Y-%m-%d} to {end:%Y-%m-%d})"
    )
    hour_axis(ax)
    ax.set_ylabel("Average rentals (count per hour)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    save(fig, "fig_hourly_pattern.png")


def fig_test_week(config: dict) -> None:
    model = config["default_model"]
    predictions = pd.read_csv(
        OUTPUTS / "predictions.csv",
        usecols=["timestamp", "period", "model", "point_forecast", "actual", "lo80", "hi80"],
        parse_dates=["timestamp"],
    )
    hours = pd.date_range("2012-08-01 00:00", "2012-08-07 23:00", freq="h")
    week = (
        predictions[(predictions["model"] == model) & (predictions["period"] == "test")]
        .set_index("timestamp")
        .reindex(hours)  # an absent hour stays a gap in the lines
    )

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.fill_between(
        week.index, week["lo80"], week["hi80"], color=BLUE, alpha=0.2, linewidth=0,
        label="80% interval",
    )
    ax.plot(week.index, week["point_forecast"], color=BLUE, label=f"Forecast ({MODEL_LABELS[model]}, default)")
    ax.plot(week.index, week["actual"], color=ORANGE, linewidth=1.4, label="Recorded count")
    ax.set_title("Recorded count vs default-model forecast, 2012-08-01 to 2012-08-07")
    ax.set_xlabel("Date in 2012 (tick at 00:00 of each day)")
    ax.set_ylabel("Rentals (count per hour)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
    ax.set_xlim(hours[0], hours[-1])
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=3)
    save(fig, "fig_test_week.png")


def fig_lasso_path(config: dict) -> None:
    results = pd.read_csv(OUTPUTS / "validation_results.csv")
    lasso = results[(results["experiment"] == "principal") & (results["method"] == "lasso")].copy()
    lasso["alpha"] = lasso["setting_json"].map(lambda text: json.loads(text)["alpha"])
    lasso = lasso.sort_values("alpha")
    selected = config["selected"]["lasso"]["settings"]["alpha"]
    n_total = int(lasso["n_coef_total"].iloc[0])
    n_hours = int(lasso["n_hours"].iloc[0])

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.6), sharex=True)
    left.plot(lasso["alpha"], lasso["validation_mae"], color=BLUE, marker="o", ms=4, label="Candidate penalties")
    left.set_ylabel("Validation MAE (rentals per hour)")
    right.plot(lasso["alpha"], lasso["n_nonzero_coef"], color=BLUE, marker="o", ms=4)
    right.set_ylabel(f"Non-zero coefficients (count, of {n_total})")
    right.set_ylim(0, n_total + 2)

    for ax in (left, right):
        ax.axvline(selected, color=ORANGE, linewidth=1.4, label=f"Selected penalty ({selected:g})")
        ax.set_xscale("log")
        ax.set_xlim(lasso["alpha"].min() / 2, lasso["alpha"].max() * 2)
        ax.set_xlabel("Lasso penalty alpha (no units, log scale)")

    fig.suptitle(f"Lasso: validation error and sparsity vs penalty ({n_hours:,} validation hours)", fontsize=10.5, color=INK)
    fig.legend(*left.get_legend_handles_labels(), loc="lower center", ncols=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig_lasso_path.png")


def fig_coverage_by_hour() -> None:
    coverage = load_metrics("interval_coverage")
    coverage = coverage[(coverage["group_type"] == "hour") & (coverage["level"] == 80)].copy()
    coverage["hour"] = coverage["group"].astype(float).astype(int)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    styles = {"bayes": (BLUE, "o"), "poisson": (ORANGE, "s")}
    for model, (color, marker) in styles.items():
        rows = coverage[coverage["model"] == model].sort_values("hour")
        ax.plot(rows["hour"], 100 * rows["coverage"], color=color, marker=marker, ms=4, label=MODEL_LABELS[model])
    ax.axhline(80, color=INK, linewidth=1.1, linestyle="--", label="Nominal 80%")
    ax.set_title("Observed 80% interval coverage by hour of day, test period")
    hour_axis(ax)
    ax.set_ylabel("Observed coverage (% of test hours)")
    ax.set_ylim(0, 104)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=3)
    save(fig, "fig_coverage_by_hour.png")


def fig_residuals_by_hour(config: dict) -> None:
    residuals = load_metrics("residual_by_hour")
    residuals = residuals[residuals["model"] == config["default_model"]].copy()
    residuals["hour"] = residuals["group"].astype(float).astype(int)
    residuals = residuals.sort_values("hour")

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(residuals["hour"], residuals["mean_residual"], width=0.75, color=BLUE)
    ax.axhline(0, color=INK, linewidth=1.0)
    ax.set_title(f"{MODEL_LABELS[config['default_model']]} (default model): mean signed residual by hour, test period")
    hour_axis(ax)
    ax.set_ylabel("Mean residual, actual - forecast\n(rentals per hour)")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_residuals_by_hour.png")


def fig_count_distribution() -> None:
    bins = pd.read_csv(OUTPUTS / "count_distribution.csv")
    tv = json.loads((OUTPUTS / "count_distribution_tv.json").read_text(encoding="utf-8"))["tv"]

    positions = np.arange(len(bins))
    width = 0.38
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(positions - width / 2 - 0.01, bins["train_prop"], width, color=BLUE, label="Training")
    ax.bar(positions + width / 2 + 0.01, bins["validation_prop"], width, color=ORANGE, label="Validation")
    ax.set_title(f"Training vs validation hourly count distribution (TV = {tv:.3f})")
    ax.set_xticks(positions, bins["bin"], fontsize=7.5)
    ax.set_xlabel("Hourly count bin (rentals per hour)")
    ax.set_ylabel("Proportion of hours in period (fraction)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    save(fig, "fig_count_distribution.png")


def main() -> None:
    config = load_config()
    fig_hourly_pattern(config)
    fig_test_week(config)
    fig_lasso_path(config)
    fig_coverage_by_hour()
    fig_residuals_by_hour(config)
    fig_count_distribution()


if __name__ == "__main__":
    main()
