"""Run the whole delay study and write every number the guide quotes.

Usage: python run_analysis.py [DATA_DIR]
DATA_DIR defaults to ../railpulse-genai/data (the raw liveboard export plus
trips.csv and routes.csv). Outputs go to results/ and figures/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import model, prepare

HERE = Path(__file__).parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "railpulse-genai" / "data"
RESULTS, FIGURES = HERE / "results", HERE / "figures"

# Temporal split: fit on the first week, judge on the days that follow, the
# way the model would be used.
TEST_FROM = pd.Timestamp("2026-08-04")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    facts: dict = {}

    raw = prepare.load_raw(DATA / "liveboard_records.csv")
    events = prepare.to_stop_events(raw)
    trips = pd.read_csv(DATA / "trips.csv", sep=";", encoding="cp1252")
    routes = pd.read_csv(DATA / "routes.csv", sep=";", encoding="cp1252")
    df = prepare.add_features(events, prepare.train_categories(trips, routes))
    # Rail-replacement buses run on roads, not on the network studied here,
    # and none of them was late: left in, they make the logit diverge.
    facts["bus_stops_dropped"] = int((df["category"] == "BUS").sum())
    facts["bus_late"] = int(df.loc[df["category"] == "BUS", "late"].sum())
    df = df[df["category"] != "BUS"].reset_index(drop=True)

    facts["raw_rows"] = len(raw)
    facts["stop_events_analysed"] = len(df)
    facts["stop_events_total"] = len(events)
    facts["snapshots_per_stop"] = round(len(raw) / len(events), 1)
    facts["runs"] = int(df["run_id"].nunique())
    facts["stations"] = int(df["station_id"].nunique())
    facts["days"] = [str(df["service_date"].min().date()), str(df["service_date"].max().date())]
    facts["seen_after_departure_share"] = round(float(df["seen_after_departure"].mean()), 3)
    facts["late_rate"] = round(float(df["late"].mean()), 4)
    facts["category_counts"] = df["category"].value_counts().to_dict()

    # Descriptive view first: raw late rates per factor, before any model.
    for col in ["hour_band", "category", "weekend", "brussels_junction"]:
        (df.groupby(col)["late"].agg(late_rate="mean", stops="size")
           .to_csv(RESULTS / f"late_rate_by_{col}.csv"))

    # 1. Explanatory model on all data, naive vs clustered standard errors.
    naive = model.fit_logit(df, model.PLANNING_FORMULA, cluster=None)
    clustered = model.fit_logit(df, model.PLANNING_FORMULA)
    or_naive = model.odds_ratio_table(naive)
    or_clustered = model.odds_ratio_table(clustered)
    or_clustered.to_csv(RESULTS / "odds_ratios_planning.csv")
    se_ratio = (clustered.bse / naive.bse).drop("Intercept")
    se_ratio.rename("clustered_se_over_naive_se").to_csv(RESULTS / "se_inflation.csv")
    facts["se_inflation_median"] = round(float(se_ratio.median()), 2)

    # Sensitivity 1: only stops whose last snapshot came after the planned
    # departure, i.e. a delay observed rather than announced.
    after = df[df["seen_after_departure"]]
    model.odds_ratio_table(model.fit_logit(after, model.PLANNING_FORMULA)).to_csv(
        RESULTS / "odds_ratios_planning_seen_after_departure.csv")
    # Sensitivity 2: one dummy per day absorbs anything that hit a whole day
    # (weather, an incident); `weekend` is then not identifiable and dropped.
    by_day = model.PLANNING_FORMULA.replace(" + weekend", "") + " + C(service_date)"
    day_or = model.odds_ratio_table(model.fit_logit(df, by_day))
    day_or[~day_or.index.str.startswith("C(service_date")].to_csv(
        RESULTS / "odds_ratios_planning_day_effects.csv")

    # 2. Live model: add the delay at the previous stop (first stops dropped).
    live_df = df.dropna(subset=["prev_stop_delay_min"])
    live = model.fit_logit(live_df, model.LIVE_FORMULA)
    model.odds_ratio_table(live).to_csv(RESULTS / "odds_ratios_live.csv")

    # 3. Out-of-sample check on the last days.
    cat_df = model.as_categories(df, model.PLANNING_FEATURES)
    train = cat_df[cat_df["service_date"] < TEST_FROM]
    test = cat_df[cat_df["service_date"] >= TEST_FROM]
    facts["train_stops"], facts["test_stops"] = len(train), len(test)

    scores = {}
    base = np.full(len(test), train["late"].mean())
    scores["baseline (train late rate)"] = model.evaluate(test["late"], base)
    p_logit = model.fit_logit(df.loc[train.index], model.PLANNING_FORMULA).predict(df.loc[test.index])
    scores["logistic, planning"] = model.evaluate(test["late"], p_logit.to_numpy())
    booster = model.fit_boosting(train, model.PLANNING_FEATURES)
    p_boost = model.predict_boosting(booster, test, model.PLANNING_FEATURES)
    scores["boosting, planning"] = model.evaluate(test["late"], p_boost)
    live_train = live_df[live_df["service_date"] < TEST_FROM]
    live_test = live_df[live_df["service_date"] >= TEST_FROM]
    p_live = model.fit_logit(live_train, model.LIVE_FORMULA).predict(live_test)
    scores["logistic, live (+ previous stop delay)"] = model.evaluate(live_test["late"], p_live.to_numpy())
    pd.DataFrame(scores).T.round(4).to_csv(RESULTS / "holdout_scores.csv")
    facts["test_late_rate"] = round(float(test["late"].mean()), 4)

    model.calibration_table(test["late"], p_logit.to_numpy()).round(4).to_csv(
        RESULTS / "calibration_planning.csv")

    # 4. What a station manager reads: predicted risk per hour band and
    # category at an average station, mid-trip, on a weekday.
    grid = pd.MultiIndex.from_product(
        [sorted(df["hour_band"].unique()), sorted(df["category"].unique())],
        names=["hour_band", "category"]).to_frame(index=False)
    grid["weekend"] = 0
    grid["brussels_junction"] = 0
    grid["log_station_stops_per_day"] = df["log_station_stops_per_day"].median()
    grid["trip_progress"] = 0.5
    grid["risk_outside_brussels"] = clustered.predict(grid)
    grid["risk_brussels_junction"] = clustered.predict(grid.assign(brussels_junction=1))
    grid.round(4).to_csv(RESULTS / "risk_grid.csv", index=False)

    _forest_plot(or_clustered, FIGURES / "odds_ratios_planning.png")
    _calibration_plot(model.calibration_table(test["late"], p_logit.to_numpy()),
                      FIGURES / "calibration_planning.png")
    (RESULTS / "facts.json").write_text(json.dumps(facts, indent=2, default=str))
    print(json.dumps(facts, indent=2, default=str))
    print(pd.DataFrame(scores).T.round(4))
    print(or_clustered.round(3))
    print("naive vs clustered CI width, first terms:")
    print(pd.DataFrame({"naive": or_naive["ci_high"] - or_naive["ci_low"],
                        "clustered": or_clustered["ci_high"] - or_clustered["ci_low"]}).round(3))


def _forest_plot(table: pd.DataFrame, path: Path) -> None:
    """Odds ratios with their intervals on a log axis, 1 = no effect."""
    t = table.sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1))
    y = np.arange(len(t))
    ax.errorbar(t["odds_ratio"], y, xerr=[t["odds_ratio"] - t["ci_low"], t["ci_high"] - t["odds_ratio"]],
                fmt="o", color="#1f4e79", capsize=3)
    ax.axvline(1, color="grey", linestyle="--")
    ax.set_xscale("log")
    ticks = [0.3, 0.5, 0.7, 1, 1.5, 2, 3]
    ax.set_xticks(ticks, [str(v) for v in ticks])
    ax.minorticks_off()
    ax.set_yticks(y, t.index)
    ax.set_xlabel("odds ratio of a late departure (≥ 6 min), 95% CI, log scale")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _calibration_plot(table: pd.DataFrame, path: Path) -> None:
    """Predicted vs observed late rate per decile; the diagonal is perfect."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(table["predicted"], table["observed"], "o-", color="#1f4e79")
    top = max(table["predicted"].max(), table["observed"].max()) * 1.05
    ax.plot([0, top], [0, top], "--", color="grey")
    ax.set_xlabel("predicted late rate")
    ax.set_ylabel("observed late rate")
    ax.set_title("Calibration on held-out days")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
