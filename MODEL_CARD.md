# Model card — trip duration at pickup

Team B. Every number here is produced by `notebooks/04-evaluate.ipynb` and can
be reproduced from a fresh clone.

## What this is

- **Predicts:** minutes from pickup to drop-off for a New York yellow taxi trip.
- **Model:** a single decision tree, `max_depth=12`, inside the shared
  preprocessor (median imputation + scaling for numeric columns, most-frequent
  imputation + one-hot for categorical ones).
- **Trained on:** May 2026 pickups only — 189,024 trips after cleaning.
- **Depth chosen on:** June (validation), not on the training data and not on July.
- **Artefact:** `models/chosen.joblib`, rebuilt by `notebooks/03-features.ipynb`.
- **Features:** `trip_distance`, pickup hour, day of week, minutes into the day,
  passenger count, pickup and drop-off borough, pickup and drop-off airport,
  weekend flag. Nothing else reaches the model.

## Intended use

Showing a passenger an arrival time at the moment the trip starts, **with the
buffer below**. Nothing else.

## Not for

- Pricing, dispatch, driver pay, or anything that settles money on the number.
- Trips outside New York, or trips through zones 264/265 — they are dropped
  from training and the model has never seen one.
- Months unlike May–July 2026. The only congestion signal here is the hour of
  the week, so a holiday, a storm, a parade or a closure is invisible to it.
- Estimating at **booking** time, before a route is known — see below.

## The assumption that flatters every number here

`trip_distance` is treated as the length of the planned route, but the meter
measured it **during** the trip, so it already contains whatever detour
actually happened. A deployed app would feed in a routing service's estimate
and would do worse than everything reported below. This is the single largest
caveat on this model, and it is stated in `PROBLEM.md`, in `01-frame.ipynb`
and here.

## How it scores

June (validation), 187,549 trips:

| Rung | MAE (min) | RMSE | vs. previous |
|---|---|---|---|
| 1. Median duration of May | 8.86 | 13.15 | — |
| 2. Distance ÷ typical speed (8.96 mph) | 9.67 | 18.52 | +0.81 |
| 3. Decision tree, depth 2 | 6.02 | 8.40 | −3.66 |
| 4. Linear regression | 5.64 | 8.04 | −0.38 |
| **5. Decision tree, depth 12 (chosen)** | **4.58** | **6.97** | −1.05 |

**Rung 2 is worse than rung 1**, and that is a real finding rather than a bug:
a single typical speed badly overestimates long trips, which is why its RMSE
(18.52) is the worst in the table. It is still the more useful rule, because
its errors are one-sided in a way the median's are not — its 90% buffer is
7.6 minutes against the median's 18.3.

July (test), **scored once**, with 95% bootstrap intervals (1,000 resamples):

| | MAE | RMSE |
|---|---|---|
| Chosen model | **4.677** [4.654, 4.703] | 7.036 [6.972, 7.105] |
| Median baseline | 8.490 [8.449, 8.525] | 12.290 [12.204, 12.368] |

The intervals do not overlap, so the gap over the baseline is not noise.
July MAE (4.68) is close to June's (4.58), which is the evidence that the
model did not quietly fit June while the depth was being chosen.

**Ship threshold from `PROBLEM.md`** — beat rung 2 by at least a minute of
MAE, with a smaller buffer: **met.** 5.09 minutes of MAE, and a buffer of 6.7
against rung 2's 7.6.

## The buffer

The app shows the predicted arrival plus a buffer so 90% of passengers arrive
before the time shown.

- Buffer chosen on June: **6.7 minutes**
- Coverage it achieved on **July: 92.4%** — the promise held
- July alone would have needed 5.4 minutes, so June's buffer is conservative
- Coverage with **no** buffer: 59.6% — the raw prediction is late four times
  in ten, which is why the buffer is part of the product and not a detail

## Where it fails

All on June, with trip counts. `needs` is the buffer that group would require
on its own; `gets` is the coverage it actually receives from the single 6.7
minute buffer.

**By distance** — the clearest failure:

| Band | n | MAE | needs |
|---|---|---|---|
| 12+ mi | 8,566 | 10.76 | 13.1 |
| 6–12 mi | 18,913 | 7.67 | 9.6 |
| 3–6 mi | 32,577 | 5.51 | 7.9 |
| 1–3 mi | 89,160 | 3.82 | 6.0 |
| <1 mi | 38,333 | 2.66 | 4.3 |

A 12-mile trip needs twice the buffer of a 1-mile trip and gets the same one.

**By borough:**

| Borough | n | MAE | needs | gets |
|---|---|---|---|---|
| Queens | 15,290 | 8.14 | 10.8 | **83%** |
| Brooklyn | 5,148 | 6.81 | 6.6 | 90% |
| Manhattan | 166,429 | 4.17 | 6.4 | 91% |
| Bronx | 668 | 8.31 | 4.2 | 94% |

**Queens is the group this model lies to:** 17 passengers in 100 arrive after
the time they were shown, against the 10 promised. Queens is where the
airports are, and airport trips are the worst slice in the data (n = 15,253,
MAE 8.62, needs 11.8 minutes).

**By hour:** 15:00–17:00 gets 87% coverage against 90% promised; the small
hours get 95–97%. The buffer is too small exactly when traffic is worst and
too large when the roads are empty.

**Flagged groups, kept but never used as features:**

| Group | n | MAE |
|---|---|---|
| Non-standard rate codes (5, 6, 99) | 726 | 9.69 |
| Flex fare trips | 46,915 | 4.87 |
| Negative fare or total | 563 | 4.99 |

Non-standard fares are twice as hard as everything else, and the model has no
way to know it is looking at one — `RatecodeID` is set at the end of the trip.

**Where a straight line fails,** from the residuals in `04`: linear regression
is optimistic by 2.5 minutes in the mid-afternoon and pessimistic by 8 minutes
at 05:00, because it must price every mile the same at every hour. The tree
keeps its hourly bias inside ±1 minute. Both still overestimate the longest
trips (12+ mi: linear −5.0, tree −3.0 minutes).

## Decisions that shape the result

| Decision | Where |
|---|---|
| Target, validity rules, population rules | `PROBLEM.md` |
| Findings table, fix/flag/drop, 6.08% of rows removed | `02-clean-and-split.ipynb` |
| Flex fare trips kept (25.7% of the sample); unknown zones and voided trips dropped | `PROBLEM.md`, `src/data.py` |
| `RatecodeID` never used to filter training rows — it is known only at the end | `PROBLEM.md`, `02` |
| Split by pickup month, not a shuffle | `src/data.py`, `02` |
| Features judged on June; airports and passenger count kept despite not paying | `03-features.ipynb` |
| Depth 12 chosen on June | `03-features.ipynb` |

## What I would do next

1. **Replace `trip_distance` with a real routing estimate.** Every number
   here rests on a distance measured after the fact; this is the difference
   between the reported score and a deployable one.
2. **A buffer per distance band, or per hour**, rather than one global number.
   The slice tables show a single buffer cannot keep a 90% promise to Queens
   and to Manhattan at once.
3. Not worth trying: more tree depth (14 and 16 were worse on June), the
   pickup zone instead of the borough (+0.011 MAE), and anything derived from
   the fare — it is the answer in disguise.
