"""Tests for the stop-event reconstruction: the steps that decide what a row is."""

import pandas as pd
import pytest

from src import model, prepare


def _raw(rows):
    df = pd.DataFrame(rows, columns=["station_id", "vehicle_id", "scheduled_time",
                                     "delay_seconds", "pulled_at", "station_name"])
    for col in ("scheduled_time", "pulled_at"):
        df[col] = pd.to_datetime(df[col])
    return df


def test_a_stop_whose_delay_drifts_between_polls_stays_one_event():
    """scheduled_time moves with the delay, so it must not be part of the key."""
    raw = _raw([
        ("S1", "V1", "2026-08-04 08:05", 0, "2026-08-04 07:45", "A"),
        ("S1", "V1", "2026-08-04 08:06", 60, "2026-08-04 08:15", "A"),
    ])
    events = prepare.to_stop_events(raw)
    assert len(events) == 1
    assert events.loc[0, "delay_seconds"] == 60          # latest snapshot wins
    assert events.loc[0, "planned_time"] == pd.Timestamp("2026-08-04 08:05")
    assert bool(events.loc[0, "seen_after_departure"])


def test_the_same_train_on_two_days_gives_two_events():
    raw = _raw([
        ("S1", "V1", "2026-08-04 08:05", 0, "2026-08-04 08:00", "A"),
        ("S1", "V1", "2026-08-05 08:05", 0, "2026-08-05 08:00", "A"),
    ])
    assert len(prepare.to_stop_events(raw)) == 2


def test_a_stop_last_seen_before_departure_is_flagged():
    raw = _raw([("S1", "V1", "2026-08-04 08:05", 0, "2026-08-04 07:50", "A")])
    assert not bool(prepare.to_stop_events(raw).loc[0, "seen_after_departure"])


@pytest.mark.parametrize("hour, band", [
    (5, "early (<6h)"), (6, "morning peak (6-9h)"), (8, "morning peak (6-9h)"),
    (9, "daytime (9-16h)"), (16, "evening peak (16-19h)"), (23, "late (19h+)"),
])
def test_hour_bands_cover_the_day(hour, band):
    assert prepare.hour_band(pd.Series([hour]))[0] == band


def _events():
    raw = _raw([
        ("gs:nmbssncb:8814001", "V1", "2026-08-04 08:00", 0, "2026-08-04 08:15", "Midi"),
        ("S2", "V1", "2026-08-04 08:20", 420, "2026-08-04 08:30", "B"),
        ("S3", "V1", "2026-08-04 08:46", 360, "2026-08-04 08:45", "C"),
        ("S2", "V2", "2026-08-08 18:00", 120, "2026-08-08 18:00", "B"),
    ])
    return prepare.to_stop_events(raw)


def test_features_target_progress_and_previous_delay():
    cats = pd.Series({"V1": "IC", "V2": "L"})
    df = prepare.add_features(_events(), cats).set_index(["vehicle_id", "station_id"])
    v1 = df.loc["V1"]
    # 420 s and 360 s are late (>= 6 min); 0 s is not.
    assert v1["late"].tolist() == [0, 1, 1]
    assert v1["trip_progress"].tolist() == [0.0, 0.5, 1.0]
    assert pd.isna(v1["prev_stop_delay_min"].iloc[0])
    assert v1["prev_stop_delay_min"].iloc[1:].tolist() == [0.0, 7.0]
    assert v1["brussels_junction"].tolist() == [1, 0, 0]
    v2 = df.loc[("V2", "S2")]
    assert v2["trip_progress"] == 0.0                   # single-stop run
    assert v2["weekend"] == 1                            # 8 Aug 2026 is a Saturday


def test_rare_categories_are_pooled(monkeypatch):
    monkeypatch.setattr(prepare, "MIN_CATEGORY_STOPS", 3)
    cats = pd.Series({"V1": "IC", "V2": "NJ"})
    df = prepare.add_features(_events(), cats)
    assert set(df["category"]) == {"IC", "other"}


def test_unknown_vehicle_gets_unknown_category(monkeypatch):
    monkeypatch.setattr(prepare, "MIN_CATEGORY_STOPS", 1)
    df = prepare.add_features(_events(), pd.Series({"V1": "IC"}))
    assert df.loc[df["vehicle_id"] == "V2", "category"].item() == "unknown"


def test_booster_refuses_uncast_text_features():
    df = pd.DataFrame({"category": ["IC", "L"], "late": [0, 1]})
    with pytest.raises(ValueError):
        model._encode(df, ["category"])


def test_readable_term_names():
    assert model._readable("C(category, Treatment('IC'))[T.L]") == "category = L"
    assert model._readable("weekend") == "weekend"
