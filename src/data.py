"""Loading and cleaning.

`data/raw/` is read-only: nothing here ever writes back to it.
Everything derived goes to `data/processed/`, which is disposable because
this module can regenerate it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import DATA_PROCESSED, DATA_RAW


def load_raw(filename: str) -> pd.DataFrame:
    """Read one original file from data/raw/ exactly as it is on disk."""
    return pd.read_csv(DATA_RAW / filename)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning decisions recorded in notebooks/02-clean-and-split.ipynb.

    Every drop, cast, rename and imputation belongs here, so the audit in the
    notebook and the table the model trains on can never drift apart.
    """
    out = df.copy()
    # TODO: the decisions from the audit, one block per decision.
    return out


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Write a derived table to data/processed/ (never committed)."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / filename
    df.to_parquet(path, index=False)
    return path
