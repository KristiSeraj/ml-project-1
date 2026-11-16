"""Feature engineering, and the guard that keeps the pipeline honest.

Everything here is knowable the moment the passenger gets in. `LEAKY` lists
the columns settled at or after drop-off; `assert_no_leakage` raises on any
of them, so a leak has to be deliberate rather than accidental.

The preprocessor is a scikit-learn `ColumnTransformer`: every learned step
(the imputers, the scaler, the encoder) is fitted inside it, on May only.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import AIRPORT_ZONES

TARGET = "duration_min"

# Settled at or after the drop-off. Never a feature. See PROBLEM.md.
LEAKY = [
    "tpep_dropoff_datetime",
    "duration_min",
    "speed_mph",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "extra",
    "mta_tax",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "Airport_fee",
    "airport_fee",
    "cbd_congestion_fee",
    "payment_type",
    "RatecodeID",
    # Flags built from end-of-trip columns: reporting slices, not inputs.
    "is_flex_fare",
    "is_voided",
    "is_non_standard_fare",
    "has_negative_amount",
]

NUMERIC = [
    "trip_distance",
    "pickup_hour",
    "pickup_dow",
    "pickup_minutes_into_day",
    "passenger_count",
]

CATEGORICAL = [
    "pickup_borough",
    "dropoff_borough",
    "pickup_airport",
    "dropoff_airport",
    "is_weekend",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clock features from the pickup timestamp — the only congestion proxy here."""
    out = df.copy()
    ts = out["tpep_pickup_datetime"]
    out["pickup_hour"] = ts.dt.hour
    out["pickup_dow"] = ts.dt.dayofweek
    out["pickup_minutes_into_day"] = ts.dt.hour * 60 + ts.dt.minute
    out["is_weekend"] = (out["pickup_dow"] >= 5).map({True: "weekend", False: "weekday"})
    return out


def add_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    """Airport flags per side: airport trips behave unlike anything else."""
    out = df.copy()
    out["pickup_airport"] = out["PULocationID"].map(AIRPORT_ZONES).fillna("none").astype(str)
    out["dropoff_airport"] = out["DOLocationID"].map(AIRPORT_ZONES).fillna("none").astype(str)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Every derived column, in the order 03-features added them."""
    return add_zone_features(add_time_features(df))


def assert_no_leakage(columns) -> None:
    """Refuse any feature set containing a column settled at the end of the trip."""
    bad = sorted(set(columns) & set(LEAKY))
    if bad:
        raise ValueError(f"these are not known at pickup: {bad}")


def feature_frame(df: pd.DataFrame, numeric=None, categorical=None) -> pd.DataFrame:
    """The model's input: allow-listed columns only, leakage checked."""
    cols = (NUMERIC if numeric is None else list(numeric)) + (
        CATEGORICAL if categorical is None else list(categorical)
    )
    assert_no_leakage(cols)
    return df[cols]


def build_preprocessor(numeric=None, categorical=None) -> ColumnTransformer:
    """The column-wise transform every model in this project shares.

    `passenger_count` is missing on every flex fare trip — a quarter of the
    sample — so the numeric branch imputes rather than drops. The imputer's
    median is learned inside the pipeline, from May only.
    """
    numeric = NUMERIC if numeric is None else list(numeric)
    categorical = CATEGORICAL if categorical is None else list(categorical)
    assert_no_leakage(numeric + categorical)

    # A branch with no columns is left out rather than given an empty list:
    # ColumnTransformer rejects one, and 03 fits a distance-only model.
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))
    if not transformers:
        raise ValueError("no columns given to the preprocessor")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Readable names for a fitted preprocessor, for coefficient tables."""
    return list(preprocessor.get_feature_names_out())
