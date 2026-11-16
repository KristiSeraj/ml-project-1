"""Loading, the target, the findings table, cleaning, and the split.

`data/raw/` is read-only: nothing here writes back to it. `rebuild()` goes
from the published files to the table a model trains on in one call, so the
whole pipeline can be reproduced from a fresh clone.

Every decision is in one of two places and nowhere else: `VALIDITY_RULES`
(is this row a real measurement?) and `POPULATION_RULES` (does this trip
belong in the question?). The notebooks report them; they never re-implement
them. Nothing here learns anything from the data — no fitted value, no
statistic, no threshold derived from the rows — so cleaning cannot leak.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import DATA_PROCESSED, DATA_RAW

TRIPS_FILE = "yellow_tripdata_2026_sample.parquet"
ZONES_FILE = "taxi_zone_lookup.csv"

AIRPORT_ZONES = {1: "EWR", 132: "JFK", 138: "LGA"}
UNKNOWN_ZONES = (264, 265)

# Payment and rate codes, from the data dictionary.
PAYMENT_FLEX_FARE = 0
PAYMENT_VOIDED = 6
RATECODE_NON_STANDARD = (5, 6, 99)  # negotiated, group ride, unknown

# Thresholds from PROBLEM.md. Fixed numbers, not quantiles of this data:
# a threshold fitted to the rows would be a learned step outside the pipeline.
MIN_DURATION_MIN = 1.0
MAX_DURATION_MIN = 180.0
MAX_DISTANCE_MI = 100.0
MIN_SPEED_MPH = 1.0
MAX_SPEED_MPH = 80.0

SPLIT_MONTHS = {"train": 5, "validation": 6, "test": 7}


# ---------------------------------------------------------------- loading


def load_trips() -> pd.DataFrame:
    """Read the 600,000-trip sample exactly as published."""
    return pd.read_parquet(DATA_RAW / TRIPS_FILE)


def load_zones() -> pd.DataFrame:
    """Read the zone lookup: 265 zone numbers, their name and borough."""
    return pd.read_csv(DATA_RAW / ZONES_FILE)


def add_zones(trips: pd.DataFrame, zones: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach pickup and drop-off borough and zone names.

    Left joins, so a zone number missing from the lookup becomes NaN rather
    than silently dropping the trip.
    """
    if zones is None:
        zones = load_zones()
    out = trips
    for side, id_col in (("pickup", "PULocationID"), ("dropoff", "DOLocationID")):
        side_zones = zones.rename(
            columns={
                "LocationID": id_col,
                "Borough": f"{side}_borough",
                "Zone": f"{side}_zone",
            }
        )
        out = out.merge(
            side_zones[[id_col, f"{side}_borough", f"{side}_zone"]],
            on=id_col,
            how="left",
        )
    return out


def add_target(trips: pd.DataFrame) -> pd.DataFrame:
    """Add `duration_min` (the target) and the columns the audit needs.

    `speed_mph` is diagnostic only: it divides by the duration, so it is made
    of the answer and must never become a feature.
    """
    out = trips.copy()
    delta = out["tpep_dropoff_datetime"] - out["tpep_pickup_datetime"]
    out["duration_min"] = delta.dt.total_seconds() / 60.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out["speed_mph"] = out["trip_distance"] / (out["duration_min"] / 60.0)
    out["pickup_month"] = out["tpep_pickup_datetime"].dt.month

    # Flags: reported, and in two cases acted on. See PROBLEM.md.
    out["is_flex_fare"] = out["payment_type"] == PAYMENT_FLEX_FARE
    out["is_voided"] = out["payment_type"] == PAYMENT_VOIDED
    out["is_unknown_zone"] = out["PULocationID"].isin(UNKNOWN_ZONES) | out[
        "DOLocationID"
    ].isin(UNKNOWN_ZONES)
    out["is_non_standard_fare"] = out["RatecodeID"].isin(RATECODE_NON_STANDARD)
    out["has_negative_amount"] = (out["fare_amount"] < 0) | (out["total_amount"] < 0)
    out["is_airport_trip"] = out["PULocationID"].isin(AIRPORT_ZONES) | out[
        "DOLocationID"
    ].isin(AIRPORT_ZONES)
    return out


# ------------------------------------------------------- the two rule sets


def validity_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Is this row a real measurement? True means the row passes."""
    return {
        "duration present": df["duration_min"].notna(),
        "duration > 0": df["duration_min"] > 0,
        f"duration >= {MIN_DURATION_MIN:g} min": df["duration_min"] >= MIN_DURATION_MIN,
        f"duration <= {MAX_DURATION_MIN:g} min": df["duration_min"] <= MAX_DURATION_MIN,
        "distance > 0": df["trip_distance"] > 0,
        f"distance <= {MAX_DISTANCE_MI:g} mi": df["trip_distance"] <= MAX_DISTANCE_MI,
        f"speed >= {MIN_SPEED_MPH:g} mph": df["speed_mph"] >= MIN_SPEED_MPH,
        f"speed <= {MAX_SPEED_MPH:g} mph": df["speed_mph"] <= MAX_SPEED_MPH,
    }


def population_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Does this trip belong in the question? True means the row stays."""
    return {
        "not a voided trip": ~df["is_voided"],
        "both zones known": ~df["is_unknown_zone"],
    }


def drop_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Every rule that removes a row, in the order the findings table shows."""
    return {**validity_rules(df), **population_rules(df)}


FINDINGS = [
    # problem, decision, what the code does
    ("Zero or negative duration", "drop", "duration_min > 0"),
    ("Duration under a minute", "drop", f"duration_min >= {MIN_DURATION_MIN:g}"),
    ("Duration over three hours", "drop", f"duration_min <= {MAX_DURATION_MIN:g}"),
    ("Zero trip_distance", "drop", "trip_distance > 0"),
    ("Meter-fault distance", "drop", f"trip_distance <= {MAX_DISTANCE_MI:g}"),
    ("Impossible implied speed", "drop", f"{MIN_SPEED_MPH:g}-{MAX_SPEED_MPH:g} mph"),
    ("Voided trip (payment_type 6)", "drop", "not billed, not a journey"),
    ("Unknown zone (264, 265)", "drop", "no borough, no geography"),
    ("Flex fare trip (payment_type 0)", "fix", "kept; passenger_count imputed in the pipeline"),
    ("Missing passenger_count", "fix", "median imputation inside the pipeline"),
    ("Negative fare or total", "flag", "has_negative_amount, reported in slices"),
    ("Non-standard RatecodeID (5, 6, 99)", "flag", "is_non_standard_fare; never a filter, never a feature"),
    ("Airport trips", "flag", "is_airport_trip, used as a feature and a slice"),
    ("April pickups outside the sample", "drop", "excluded by the month split"),
]


def findings_table(df: pd.DataFrame) -> pd.DataFrame:
    """The fix / flag / drop table the project asks for, with the counts.

    `rows` is how many rows the problem touches; `removed_here` is how many
    this rule is the first to remove, applying them in order.
    """
    counts = {
        "Zero or negative duration": df["duration_min"] <= 0,
        "Duration under a minute": df["duration_min"] < MIN_DURATION_MIN,
        "Duration over three hours": df["duration_min"] > MAX_DURATION_MIN,
        "Zero trip_distance": df["trip_distance"] <= 0,
        "Meter-fault distance": df["trip_distance"] > MAX_DISTANCE_MI,
        "Impossible implied speed": ~df["speed_mph"].between(MIN_SPEED_MPH, MAX_SPEED_MPH),
        "Voided trip (payment_type 6)": df["is_voided"],
        "Unknown zone (264, 265)": df["is_unknown_zone"],
        "Flex fare trip (payment_type 0)": df["is_flex_fare"],
        "Missing passenger_count": df["passenger_count"].isna(),
        "Negative fare or total": df["has_negative_amount"],
        "Non-standard RatecodeID (5, 6, 99)": df["is_non_standard_fare"],
        "Airport trips": df["is_airport_trip"],
        "April pickups outside the sample": ~df["pickup_month"].isin(SPLIT_MONTHS.values()),
    }

    # Cumulative effect of the drop rules, in order.
    removed_here = {}
    kept = pd.Series(True, index=df.index)
    order = list(drop_rules(df).items())
    label_for = {
        "duration > 0": "Zero or negative duration",
        f"duration >= {MIN_DURATION_MIN:g} min": "Duration under a minute",
        f"duration <= {MAX_DURATION_MIN:g} min": "Duration over three hours",
        "distance > 0": "Zero trip_distance",
        f"distance <= {MAX_DISTANCE_MI:g} mi": "Meter-fault distance",
        f"speed >= {MIN_SPEED_MPH:g} mph": "Impossible implied speed",
        f"speed <= {MAX_SPEED_MPH:g} mph": "Impossible implied speed",
        "not a voided trip": "Voided trip (payment_type 6)",
        "both zones known": "Unknown zone (264, 265)",
    }
    for rule_name, mask in order:
        before = int(kept.sum())
        kept = kept & mask
        label = label_for.get(rule_name)
        if label:
            removed_here[label] = removed_here.get(label, 0) + before - int(kept.sum())

    n = len(df)
    rows = []
    for problem, decision, action in FINDINGS:
        mask = counts[problem]
        rows.append(
            {
                "problem": problem,
                "decision": decision,
                "what the code does": action,
                "rows": int(mask.sum()),
                "pct": round(100 * float(mask.mean()), 3),
                "removed_here": removed_here.get(problem, 0),
            }
        )
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {
        "problem": "TOTAL",
        "decision": "",
        "what the code does": f"{int(kept.sum()):,} rows kept",
        "rows": n - int(kept.sum()),
        "pct": round(100 * (n - int(kept.sum())) / n, 3),
        "removed_here": n - int(kept.sum()),
    }
    return out


# --------------------------------------------------------------- cleaning


def clean(trips: pd.DataFrame) -> pd.DataFrame:
    """Apply every drop rule. Expects `add_target` to have run.

    Learns nothing: every threshold is a constant written in PROBLEM.md.
    """
    kept = pd.Series(True, index=trips.index)
    for mask in drop_rules(trips).values():
        kept = kept & mask
    return trips.loc[kept].reset_index(drop=True)


def rebuild() -> pd.DataFrame:
    """Raw published files to the clean table, in one call."""
    return clean(add_target(add_zones(load_trips())))


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """May / June / July by pickup month, the split fixed for every team.

    July is not to be touched before `04-evaluate.ipynb`.
    """
    return {
        name: df.loc[df["pickup_month"] == month].reset_index(drop=True)
        for name, month in SPLIT_MONTHS.items()
    }


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Write a derived table to data/processed/ (never committed)."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / filename
    df.to_parquet(path, index=False)
    return path


def load_processed(filename: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / filename)
