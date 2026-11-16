"""The baseline ladder, the two model families, and saving.

Rungs 1 and 2 are estimators, not numbers in a notebook cell, so every rung
is scored through the same code path as the models above it.

The project allows linear regression and a single decision tree. Nothing
else is fitted anywhere in this repository.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from . import MODELS


class MedianDuration(BaseEstimator, RegressorMixin):
    """Rung 1: the median duration of May, for every trip."""

    def fit(self, X, y):
        self.median_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.median_)


class TypicalSpeed(BaseEstimator, RegressorMixin):
    """Rung 2: distance ÷ a typical speed, the speed taken from May.

    The speed is the median of (distance / duration) over the training rows,
    so the rule is fitted rather than guessed — and fitted on May only.
    """

    def __init__(self, distance_col: str = "trip_distance"):
        self.distance_col = distance_col

    def fit(self, X, y):
        distance = np.asarray(X[self.distance_col], dtype=float)
        hours = np.asarray(y, dtype=float) / 60.0
        self.speed_mph_ = float(np.median(distance / hours))
        return self

    def predict(self, X):
        distance = np.asarray(X[self.distance_col], dtype=float)
        return distance / self.speed_mph_ * 60.0


def shallow_tree(max_depth: int = 2, random_state: int = 0) -> DecisionTreeRegressor:
    """Rung 3: a decision tree two levels deep."""
    return DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)


def linear_regression() -> LinearRegression:
    return LinearRegression()


def decision_tree(max_depth: int, random_state: int = 0) -> DecisionTreeRegressor:
    return DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)


def build_model(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    """Preprocessor and estimator as one object: nothing is fitted outside it."""
    return Pipeline([("pre", preprocessor), ("est", estimator)])


def save(model, name: str) -> Path:
    """Write a fitted artefact to models/ (output, not source: never committed)."""
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load(name: str):
    return joblib.load(MODELS / f"{name}.joblib")
