"""Turn raw SNCB liveboard polling snapshots into one row per stop event.

The export is a log of 15-minute polls: the same train at the same station is
captured on every poll until it leaves, so raw rows over-weight whatever was
polled most. The unit of analysis here is the stop event, one train at one
station on one day, keeping the latest snapshot because it is the closest to
the delay the train actually left with.

In this export `scheduled_time` is NOT the timetable time: it moves with the
delay (08:05 with 0 s, then 08:06 with 60 s). The planned time is therefore
rebuilt as `scheduled_time - delay_seconds`, and every time feature (hour,
weekday) is cut on that planned time, never on the drifting one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Infrabel counts a train as punctual below 6 minutes of delay; delays in the
# feed are whole minutes, so "late" means 360 s or more.
LATE_SECONDS = 360

# The North-South junction: six tracks between Nord and Midi carry most of the
# network's through traffic, the usual suspect for delay in Belgium.
BRUSSELS_JUNCTION = {
    "gs:nmbssncb:8812005",  # Bruxelles-Nord
    "gs:nmbssncb:8813045",  # Bruxelles-Congres
    "gs:nmbssncb:8813003",  # Bruxelles-Central
    "gs:nmbssncb:8813037",  # Bruxelles-Chapelle
    "gs:nmbssncb:8814001",  # Bruxelles-Midi
}

# Hour bands follow commuter rhythm rather than equal widths, so each band has
# a meaning a station manager recognises.
HOUR_BANDS = [
    (0, 6, "early (<6h)"),
    (6, 9, "morning peak (6-9h)"),
    (9, 16, "daytime (9-16h)"),
    (16, 19, "evening peak (16-19h)"),
    (19, 24, "late (19h+)"),
]

# Categories with too few stops for a stable estimate are pooled, so the
# model never reports an odds ratio resting on a handful of trains.
MIN_CATEGORY_STOPS = 500

RAW_COLUMNS = ["station_id", "vehicle_id", "scheduled_time", "delay_seconds",
               "pulled_at", "Stations Name"]


def load_raw(path) -> pd.DataFrame:
    """Read the liveboard export (cp1252, ';'-separated, day-first dates)."""
    df = pd.read_csv(path, sep=";", encoding="cp1252", usecols=RAW_COLUMNS,
                     low_memory=False)
    for col in ("scheduled_time", "pulled_at"):
        df[col] = pd.to_datetime(df[col], format="%d/%m/%Y %H:%M")
    return df.rename(columns={"Stations Name": "station_name"})


def to_stop_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per (train, station, planned day): the latest snapshot.

    The key uses the planned day, not the drifting scheduled_time, otherwise a
    stop whose delay changed between polls would count as two stops.
    """
    df = raw.copy()
    df["planned_time"] = df["scheduled_time"] - pd.to_timedelta(df["delay_seconds"], unit="s")
    df["service_date"] = df["planned_time"].dt.normalize()
    key = ["vehicle_id", "station_id", "service_date"]
    events = (df.sort_values("pulled_at", kind="stable")
                .drop_duplicates(key, keep="last")
                .reset_index(drop=True))
    # Minutes between the last snapshot and the planned departure; negative
    # means the train was still seen after its planned time.
    lead = (events["planned_time"] - events["pulled_at"]).dt.total_seconds() / 60
    events["seen_after_departure"] = lead <= 0
    return events


def train_categories(trips: pd.DataFrame, routes: pd.DataFrame) -> pd.Series:
    """Map vehicle_id (a GTFS trip_id in this export) to its train category."""
    cat = trips.merge(routes[["route_id", "train_category"]], on="route_id", how="left")
    return cat.drop_duplicates("trip_id").set_index("trip_id")["train_category"]


def hour_band(hours: pd.Series) -> pd.Series:
    """Label each planned hour with its commuter band."""
    out = pd.Series(index=hours.index, dtype=object)
    for lo, hi, label in HOUR_BANDS:
        out[(hours >= lo) & (hours < hi)] = label
    return out


def add_features(events: pd.DataFrame, categories: pd.Series) -> pd.DataFrame:
    """Add the target and the explanatory variables used by the models.

    Two kinds of variable are kept apart on purpose: what is known from the
    timetable a day ahead (hour, weekday, category, station, position along
    the trip) and what is only known live (the delay at the previous stop).
    """
    df = events.copy()
    df["late"] = (df["delay_seconds"] >= LATE_SECONDS).astype(int)
    df["hour_band"] = hour_band(df["planned_time"].dt.hour)
    df["weekend"] = (df["planned_time"].dt.dayofweek >= 5).astype(int)

    cat = df["vehicle_id"].map(categories).fillna("unknown")
    counts = cat.value_counts()
    df["category"] = cat.where(cat.map(counts) >= MIN_CATEGORY_STOPS, "other")

    df["brussels_junction"] = df["station_id"].isin(BRUSSELS_JUNCTION).astype(int)
    # Station size as stops per day it handles in the data, on a log scale so
    # that doubling traffic is one step whatever the station's size.
    days = df["service_date"].nunique()
    per_day = df.groupby("station_id")["station_id"].transform("size") / days
    df["log_station_stops_per_day"] = np.log(per_day)

    # A run is one train on one day; stops are ordered by planned time.
    df["run_id"] = df["vehicle_id"] + "|" + df["service_date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values(["run_id", "planned_time"], kind="stable").reset_index(drop=True)
    run = df.groupby("run_id")
    n = run["run_id"].transform("size")
    rank = run.cumcount()
    df["trip_progress"] = np.where(n > 1, rank / (n - 1).clip(lower=1), 0.0)
    df["prev_stop_delay_min"] = run["delay_seconds"].shift(1) / 60
    return df
