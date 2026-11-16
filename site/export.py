"""Build the static site's data files. Run from the repository root:

    python site/export.py

Writes three JSON files next to the page:

  site/model.json    the preprocessor's fitted numbers and the decision tree
  site/zones.json    zone names, boroughs, centroids, and the detour factor
  site/results.json  the June ladder, the July scores, the buffer, the slices

Nothing here is hand-written: the page cannot drift from the notebooks,
because every number it shows is produced by this script from the same
`src/` code the notebooks import.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import shapefile  # noqa: E402  (pyshp)

from src import DATA_RAW  # noqa: E402
from src.data import SPLIT_MONTHS, load_zones, rebuild, split  # noqa: E402
from src.evaluate import (  # noqa: E402
    COVERAGE,
    bootstrap_ci,
    buffer_minutes,
    by_slice,
    coverage_at,
    distance_band,
    score,
)
from src.features import CATEGORICAL, NUMERIC, build_features, feature_frame  # noqa: E402
from src.models import MedianDuration, TypicalSpeed, build_model, decision_tree  # noqa: E402
from src.features import build_preprocessor  # noqa: E402

SITE = Path(__file__).resolve().parent
FEET_PER_MILE = 5280.0
CHOSEN_DEPTH = 12  # chosen on June in notebooks/03-features.ipynb


# ----------------------------------------------------------------- the model


def export_model(model) -> dict:
    """The fitted preprocessor's numbers and the tree, as plain arrays.

    The browser rebuilds the feature vector in the same column order the
    ColumnTransformer produces — numeric columns first, then the one-hot
    blocks in order — and then walks the tree. No ML runtime needed.
    """
    pre = model.named_steps["pre"]
    est = model.named_steps["est"]

    num_pipe = pre.named_transformers_["num"]
    cat_pipe = pre.named_transformers_["cat"]
    tree = est.tree_

    return {
        "numeric": {
            "columns": list(NUMERIC),
            "impute": [float(v) for v in num_pipe.named_steps["impute"].statistics_],
            "mean": [float(v) for v in num_pipe.named_steps["scale"].mean_],
            "scale": [float(v) for v in num_pipe.named_steps["scale"].scale_],
        },
        "categorical": {
            "columns": list(CATEGORICAL),
            "impute": [str(v) for v in cat_pipe.named_steps["impute"].statistics_],
            "categories": [
                [str(c) for c in cats]
                for cats in cat_pipe.named_steps["encode"].categories_
            ],
        },
        "feature_names": [str(n) for n in pre.get_feature_names_out()],
        "tree": {
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(),
            "threshold": [float(t) for t in tree.threshold],
            "value": [round(float(v), 9) for v in tree.value[:, 0, 0]],
            "n_node_samples": tree.n_node_samples.tolist(),
        },
        "depth": int(est.get_depth()),
        "n_nodes": int(tree.node_count),
    }


# ------------------------------------------------------------------ the zones


def zone_centroids() -> dict[int, tuple[float, float]]:
    """Centroid of each zone, in feet, from the published shapefile.

    The shapefile is NAD83 State Plane New York Long Island, in feet — not
    latitude and longitude. Distances derived from it are therefore plane
    distances, which is all the straight-line estimate needs.
    """
    reader = shapefile.Reader(str(DATA_RAW / "taxi_zones" / "taxi_zones"))
    fields = [f[0] for f in reader.fields[1:]]
    loc_idx = fields.index("LocationID")

    out: dict[int, tuple[float, float]] = {}
    for rec, shape in zip(reader.records(), reader.shapes()):
        loc = int(rec[loc_idx])
        pts = np.asarray(shape.points, dtype=float)
        if len(pts) == 0:
            continue
        # Area-weighted centroid of the largest ring, falling back to the
        # mean of its points when the ring is degenerate.
        parts = list(shape.parts) + [len(pts)]
        rings = [pts[parts[i] : parts[i + 1]] for i in range(len(parts) - 1)]
        ring = max(rings, key=len)
        x, y = ring[:, 0], ring[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        area = cross.sum() / 2.0
        if abs(area) > 1e-6:
            cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * area)
            cy = ((y + np.roll(y, -1)) * cross).sum() / (6 * area)
        else:
            cx, cy = ring[:, 0].mean(), ring[:, 1].mean()
        out[loc] = (float(cx), float(cy))
    return out


def straight_line_miles(
    pu: pd.Series, do: pd.Series, centroids: dict[int, tuple[float, float]]
) -> pd.Series:
    """Centre-to-centre distance in miles, NaN when a zone has no shape."""
    cx = {k: v[0] for k, v in centroids.items()}
    cy = {k: v[1] for k, v in centroids.items()}
    dx = pu.map(cx) - do.map(cx)
    dy = pu.map(cy) - do.map(cy)
    return np.sqrt(dx**2 + dy**2) / FEET_PER_MILE


def learn_detour(train: pd.DataFrame, centroids: dict) -> dict:
    """How much longer the metered route is than the straight line.

    Learned on **May only**, like everything else fitted in this project.
    Two numbers: a factor for trips between different zones, and a flat
    median distance for trips that start and end in the same zone, where the
    straight line is zero and the factor would be meaningless.
    """
    sl = straight_line_miles(train["PULocationID"], train["DOLocationID"], centroids)
    same_zone = train["PULocationID"] == train["DOLocationID"]

    usable = (~same_zone) & sl.notna() & (sl > 0.3)
    factor = float(np.median(train.loc[usable, "trip_distance"] / sl[usable]))
    same_zone_miles = float(np.median(train.loc[same_zone, "trip_distance"]))

    # How good is the estimate? Reported on the page rather than hidden.
    est = sl[usable] * factor
    err = (est - train.loc[usable, "trip_distance"]).abs()
    return {
        "factor": round(factor, 4),
        "same_zone_miles": round(same_zone_miles, 3),
        "median_abs_error_mi": round(float(np.median(err)), 3),
        "n_pairs_used": int(usable.sum()),
    }


# ---------------------------------------------------------------- the results


def slice_records(df: pd.DataFrame, by: str, buffer: float) -> list[dict]:
    table = by_slice(df, by, buffer=buffer)
    out = []
    for row in table.to_dict("records"):
        out.append(
            {
                "group": str(row[by]),
                "n": int(row["n"]),
                "mae": round(float(row["mae"]), 2),
                "rmse": round(float(row["rmse"]), 2),
                "needs_buffer": round(float(row["needs_buffer"]), 1),
                "coverage_global": round(float(row["coverage_global"]), 3),
            }
        )
    return out


def main() -> int:
    print("rebuilding from data/raw/ ...")
    df = build_features(rebuild())
    parts = split(df)
    train, val, test = parts["train"], parts["validation"], parts["test"]
    y_train, y_val = train["duration_min"], val["duration_min"]
    X_train, X_val = feature_frame(train), feature_frame(val)

    print(f"  May {len(train):,} · June {len(val):,} · July {len(test):,}")

    print("refitting the chosen model on May ...")
    model = build_model(build_preprocessor(), decision_tree(CHOSEN_DEPTH))
    model.fit(X_train, y_train)

    val = val.assign(prediction=model.predict(X_val))
    test = test.assign(prediction=model.predict(feature_frame(test)))
    buffer = buffer_minutes(val["duration_min"], val["prediction"], COVERAGE)

    print("scoring the ladder on June ...")
    rung1 = MedianDuration().fit(X_train, y_train)
    rung2 = TypicalSpeed().fit(train[["trip_distance"]], y_train)
    ladder = [
        {"name": "Median duration of May", **score(y_val, rung1.predict(X_val))},
        {
            "name": f"Distance ÷ {rung2.speed_mph_:.2f} mph",
            **score(y_val, rung2.predict(val[["trip_distance"]])),
        },
        {"name": f"Decision tree, depth {CHOSEN_DEPTH}", **score(y_val, val["prediction"])},
    ]
    for row in ladder:
        for k in ("mae", "rmse", "bias"):
            row[k] = round(float(row[k]), 3)

    print("bootstrapping July (scored once) ...")
    july = {}
    base_pred = rung1.predict(feature_frame(test))
    for label, pred in [("model", test["prediction"]), ("baseline", base_pred)]:
        july[label] = {}
        for metric in ("mae", "rmse"):
            point, lo, hi = bootstrap_ci(test["duration_min"], pred, metric=metric)
            july[label][metric] = [round(point, 3), round(lo, 3), round(hi, 3)]

    val = val.assign(
        pickup_hour_slice=val["tpep_pickup_datetime"].dt.hour,
        distance_band=distance_band(val["trip_distance"]),
    )

    results = {
        "coverage_target": COVERAGE,
        "buffer_min": round(buffer, 1),
        "coverage_no_buffer": round(coverage_at(val["duration_min"], val["prediction"], 0), 3),
        "coverage_july": round(coverage_at(test["duration_min"], test["prediction"], buffer), 3),
        "buffer_july_would_need": round(
            buffer_minutes(test["duration_min"], test["prediction"], COVERAGE), 1
        ),
        "ladder": ladder,
        "july": july,
        "n": {k: int(len(v)) for k, v in parts.items()},
        "slices": {
            "hour": slice_records(val, "pickup_hour_slice", buffer),
            "borough": slice_records(val, "pickup_borough", buffer),
            "distance_band": slice_records(val, "distance_band", buffer),
            "airport": slice_records(val, "is_airport_trip", buffer),
        },
    }

    print("reading zone centroids from the shapefile ...")
    centroids = zone_centroids()
    detour = learn_detour(train, centroids)
    print(f"  detour factor {detour['factor']} (median error "
          f"{detour['median_abs_error_mi']} mi)")

    lookup = load_zones().set_index("LocationID")
    zones = []
    for loc, (cx, cy) in sorted(centroids.items()):
        if loc not in lookup.index:
            continue
        row = lookup.loc[loc]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if str(row["Borough"]) == "Unknown":
            continue  # dropped in cleaning; the model has never seen one
        zones.append(
            {
                "id": int(loc),
                "zone": str(row["Zone"]),
                "borough": str(row["Borough"]),
                "x": round(cx, 1),
                "y": round(cy, 1),
            }
        )

    model_json = export_model(model)
    model_json["trained_on"] = {
        "month": "May 2026",
        "rows": int(len(train)),
        "split": {k: int(v) for k, v in SPLIT_MONTHS.items()},
    }

    for name, payload in [
        ("model.json", model_json),
        ("zones.json", {"detour": detour, "zones": zones}),
        ("results.json", results),
    ]:
        path = SITE / name
        path.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")

    # A sample for the parity check: the browser must agree with scikit-learn.
    sample = val.sample(2000, random_state=0)
    parity = {
        "rows": [
            {
                "numeric": [
                    None if pd.isna(r[c]) else float(r[c]) for c in NUMERIC
                ],
                "categorical": [
                    None if pd.isna(r[c]) else str(r[c]) for c in CATEGORICAL
                ],
                "sklearn": round(float(r["prediction"]), 6),
            }
            for _, r in sample.iterrows()
        ]
    }
    (SITE / "parity-sample.json").write_text(json.dumps(parity, separators=(",", ":")))
    print(f"wrote site/parity-sample.json  ({len(parity['rows'])} June trips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
