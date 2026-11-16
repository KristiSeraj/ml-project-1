# The site

A static page that runs this project's model in the browser. No backend, no
build step, no framework: four files and three exported JSON files.

```bash
python site/export.py      # refit on May, write model.json, zones.json, results.json
node site/parity.js        # the page's predictions vs scikit-learn, 2,000 June trips
python -m http.server -d site 8000
```

Then open <http://localhost:8000>. It must be served over HTTP, not opened as
a `file://` URL, because the page `fetch`es its JSON.

| File | What it is |
|---|---|
| `index.html` | The page. |
| `style.css` | Roles as CSS custom properties; light and dark. |
| `model.js` | Rebuilds the feature vector and walks the tree. Used by the page *and* by the parity check, so the tested code is the shipped code. |
| `app.js` | Loads the JSON, fills the tables, runs the prediction. |
| `export.py` | Refits the chosen model on May and writes everything the page shows. |
| `parity.js` | `node site/parity.js` — fails if the browser and scikit-learn disagree by more than 1e-6 minutes. |
| `model.json` | The fitted preprocessor's numbers and the 6,407-node tree. |
| `zones.json` | Zone names, boroughs, centroids in feet, and the detour factor. |
| `results.json` | The June ladder, the July scores with intervals, the buffer, the slices. |

## Why the predictions are reproducible, and why the numbers are not hand-typed

`export.py` imports the same `src/` modules the notebooks use and refits on
May, so a number on the page cannot drift from the notebooks. The only
arithmetic the page itself does is the prediction, and `parity.js` checks that
against scikit-learn.

## The one thing the page does that the notebooks do not

The model was trained on `trip_distance`, which the meter measures *during*
the trip. A real app has no such number at pickup, so the page estimates the
route: centre-to-centre straight line between the two zones (from
`taxi_zones.shp`, which is in feet), times a detour factor learned on May.
That estimate carries its own error, which the page states rather than hides.

## Deploying

`.github/workflows/pages.yml`, triggered by hand. It runs the parity check
first and refuses to deploy if the committed `model.json` disagrees with the
committed scikit-learn predictions.
