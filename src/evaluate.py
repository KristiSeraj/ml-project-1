"""Metrics and plots.

The baseline ladder and the slice table both run through here, so a number in
the report and a number in a notebook are computed the same way.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import FIGURES


def score(y_true, y_pred) -> dict[str, float]:
    """The project's headline metrics, defined once.

    Named in PROBLEM.md before any modelling, and not changed afterwards
    without saying so in MODEL_CARD.md.
    """
    raise NotImplementedError("Pick the metrics in PROBLEM.md first.")


def by_slice(df: pd.DataFrame, y_true: str, y_pred: str, by: str) -> pd.DataFrame:
    """Scores per group, which is where a good average usually breaks."""
    return (
        df.groupby(by, dropna=False)
        .apply(lambda g: pd.Series(score(g[y_true], g[y_pred])))
        .reset_index()
    )


def save_figure(fig, name: str) -> Path:
    """Write a figure to reports/figures/ so it is findable when writing up."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
