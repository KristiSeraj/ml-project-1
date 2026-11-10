"""Feature engineering.

One pipeline, fitted on the training split only, so a feature can never see
a value from the future or from a row the model is scored on.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
) -> ColumnTransformer:
    """The column-wise transform every model in this project shares."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Hand-built features, added one at a time so each addition can be measured."""
    out = df.copy()
    # TODO: one function call per feature, in the order they were added.
    return out
