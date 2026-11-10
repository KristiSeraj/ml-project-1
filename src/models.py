"""Training and prediction.

Models are built as a single pipeline (preprocessor + estimator) so that
fitting, saving and scoring all carry the same transforms.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from . import MODELS


def build_model(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    """Glue the shared preprocessor to one estimator."""
    return Pipeline([("pre", preprocessor), ("est", estimator)])


def save(model: Pipeline, name: str) -> Path:
    """Write a fitted artefact to models/ (output, not source: never committed)."""
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load(name: str) -> Pipeline:
    """Read back an artefact saved by `save`."""
    return joblib.load(MODELS / f"{name}.joblib")
