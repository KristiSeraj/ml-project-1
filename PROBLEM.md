# Problem card — How long will this trip take?

Written before any modelling. Kept up to date: if the work contradicts this
file, this file is fixed in the same commit.

## The question

A passenger has just got into a yellow taxi. How many minutes until they
arrive?

## Who decides what, with the answer

The app shows the passenger an arrival time the moment the trip starts. The
passenger decides whether to stay in the cab, and plans what happens at the
other end, around that number. A number shown too early is worse than one
shown too late: the passenger misses the thing they planned for. So the app
shows an arrival time **with a buffer**, and the size of that buffer is part
of what this project has to deliver (see "The decision" below).

## The target

- `duration_min` = `tpep_dropoff_datetime` − `tpep_pickup_datetime`, in minutes.
- Type: regression.
- Neither timestamp is a feature. The drop-off time *is* the answer.

### Which durations count as real

Each rule below is a decision, not a fact. The counts in `notebooks/01-frame.ipynb` are
what each one would cost; week 2 settles the thresholds, records the removals
in `notebooks/02-clean-and-split.ipynb`, and implements them in
`src.data.clean` so the audit and the training table cannot drift apart.

| Rule | Why |
|---|---|
| `duration_min` > 0 | Zero and negative durations are meter or clock errors, not trips. |
| `duration_min` ≥ 1 | A trip of a few seconds is a meter started and stopped, not a ride. |
| `duration_min` ≤ 180 | Beyond three hours the meter was probably left running. Three hours is a choice; the audit shows what sits either side of it. |
| `trip_distance` > 0 | A trip with no measured distance has no route to reason about. |
| `trip_distance` ≤ 100 | Above this the distance is a meter fault, not a trip in New York. |
| Implied speed between 1 and 80 mph | Catches the pairs that are individually plausible and jointly impossible. |

Rows dropped are counted and reported, never silently removed.

## The moment of prediction

Known when the passenger gets in:

- `tpep_pickup_datetime` — and anything derived from it: hour, day of week, month.
- `PULocationID`, `DOLocationID` — and their zone and borough names, and
  whether either is an airport (zones 1, 132, 138) or unknown (264, 265).
- `trip_distance` — **treated as the length of the planned route.** This is an
  assumption, and it is generous: the meter measured it during the trip, so it
  already contains any detour that happened. A real app would use a routing
  service's estimate. Stated here because it flatters every score below.
- `passenger_count` — entered by the driver at the start. Missing on flex fare
  trips (a quarter of the sample), so it is used only with that in mind.

**Not known**, and therefore never a feature:

`tpep_dropoff_datetime`, `fare_amount` (the meter charges for time, so it
contains the answer), `tip_amount`, `total_amount`, `extra`, `mta_tax`,
`tolls_amount`, `improvement_surcharge`, `congestion_surcharge`,
`Airport_fee`, `cbd_congestion_fee`, `payment_type` (settled at the end),
`RatecodeID` (set at the end of the trip).

`src.features` will take an explicit allow-list of the columns above, so a
column settled at the end of the trip cannot reach a model by accident.

## The metric

- Headline: **MAE**, in minutes. It is the number the passenger feels, and it
  is not dominated by the long tail.
- Reported alongside: **RMSE**, which is, and so says something about the tail.
- Cost of arriving later than shown: high — the passenger planned around it.
- Cost of arriving earlier than shown: low — they wait.
- Reported with a bootstrap interval on July, so a difference of a few seconds
  of MAE is not read as a result.

### Worth shipping if

The chosen model's June MAE beats rung 2 of the ladder (distance ÷ typical
speed) by at least one minute, and the buffer needed for 90% coverage is
smaller than the one that rule needs. A model that only matches the rule is
not worth its complexity.

## The data

- Source: TLC yellow taxi trip records, fetched by `data/get_data.py`.
- 600,000 trips, 200,000 from each of May, June and July 2026, raw on purpose.
- Grain: one row per trip.
- Split, fixed for every team so results compare:

| Split | Pickup month | Used for |
|---|---|---|
| Train | May 2026 | Fitting |
| Validation | June 2026 | Every choice: features, model, hyperparameters, the buffer |
| Test | July 2026 | Looked at **once**, in `04-evaluate.ipynb` |

Split on pickup month, not on a random shuffle: the model will always be used
on a month it has not seen.

## The baseline ladder

1. The median duration of May, for every trip.
2. A rule: `trip_distance` ÷ a typical speed taken from May.
3. A decision tree two levels deep.

Then: linear regression and a decision tree, judged on June. A model earns its
place only by beating the rung below it.

## The analysis this needs

- Duration against distance — how far does distance alone get you?
- Average speed by hour of day and day of week. When is New York slowest?
- Speed by pickup zone and borough, and for airport trips.
- The long tail: the trips that took far longer than their distance suggests.

## The decision

The app shows an arrival time with a buffer. **How large must the buffer be
for 90% of passengers to arrive before the time shown?** Answered as the 90th
percentile of (actual − predicted) on June, checked once on July, and reported
per slice in `MODEL_CARD.md` — a single buffer that works on average will be
too small for some hours and boroughs.

## Known constraints

- Zones, not coordinates. The shapefile is in feet, not latitude/longitude.
- No traffic, weather or event data in the sample: the hour of the week is the
  only proxy for congestion.
- Cash tips are never recorded, and zones 264/265 are unknown and outside New
  York — neither affects this target, but both affect what the rows mean.
