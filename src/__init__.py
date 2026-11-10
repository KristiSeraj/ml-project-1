"""Shared code for ml-project-1.

Anything used in more than one notebook lives here, not in a notebook cell.
Notebooks import from this package:

    from src.data import load_raw, clean
    from src.features import build_pipeline
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"

__all__ = ["ROOT", "DATA_RAW", "DATA_PROCESSED", "MODELS", "FIGURES"]
