"""Metrics, slice tables, the buffer, the bootstrap, and figures.

Every number reported anywhere — a notebook, `reports/report.md`,
`MODEL_CARD.md` — comes through here, so two numbers that claim to be the
same thing are the same thing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import FIGURES

COVERAGE = 0.90  # the promise in PROBLEM.md: 90% arrive before the time shown


def score(y_true, y_pred) -> dict[str, float]:
    """MAE (headline) and RMSE (what the tail does), in minutes."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),  # negative means the app runs optimistic
        "n": int(len(y_true)),
    }


def ladder(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """The ladder table: one row per rung, in the order they were tried."""
    out = pd.DataFrame(results).T
    out.index.name = "model"
    out["mae_vs_previous"] = out["mae"].diff()
    return out.reset_index()


def buffer_minutes(y_true, y_pred, coverage: float = COVERAGE) -> float:
    """Minutes to add to a prediction so `coverage` of passengers arrive first.

    The quantile of (actual − predicted): how far short the model runs, at the
    point where only 1 − coverage of passengers are still late.
    """
    short_by = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(max(0.0, np.quantile(short_by, coverage)))


def coverage_at(y_true, y_pred, buffer: float) -> float:
    """Share of trips arriving before prediction + buffer."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_true <= y_pred + buffer))


def distance_band(distance: pd.Series) -> pd.Series:
    """Fixed bands, so a June table and a July table compare."""
    return pd.cut(
        distance,
        bins=[0, 1, 3, 6, 12, np.inf],
        labels=["<1 mi", "1-3 mi", "3-6 mi", "6-12 mi", "12+ mi"],
        right=False,
    )


def by_slice(
    df: pd.DataFrame,
    by: str,
    y_true: str = "duration_min",
    y_pred: str = "prediction",
    coverage: float = COVERAGE,
    buffer: float | None = None,
    min_n: int = 100,
) -> pd.DataFrame:
    """Scores per group, with the trip count the project requires.

    `needs_buffer` is what that group would need on its own; `coverage_global`
    is what it actually gets from the single buffer the app shows.
    """
    rows = []
    for key, g in df.groupby(by, dropna=False, observed=True):
        if len(g) < min_n:
            continue
        row = {by: key, **score(g[y_true], g[y_pred])}
        row["needs_buffer"] = buffer_minutes(g[y_true], g[y_pred], coverage)
        if buffer is not None:
            row["coverage_global"] = coverage_at(g[y_true], g[y_pred], buffer)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae", ascending=False).reset_index(drop=True)


def bootstrap_ci(
    y_true,
    y_pred,
    metric: str = "mae",
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile interval — for the one look at July.

    Without it, a difference of a few seconds of MAE reads as a result.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = score(y_true[idx], y_pred[idx])[metric]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return score(y_true, y_pred)[metric], float(lo), float(hi)


def residuals(df: pd.DataFrame, y_true: str = "duration_min", y_pred: str = "prediction") -> pd.Series:
    """actual − predicted. Positive means the trip took longer than promised."""
    return df[y_true] - df[y_pred]


def save_figure(fig, name: str) -> Path:
    """Write a figure to reports/figures/ so it is findable when writing up."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
