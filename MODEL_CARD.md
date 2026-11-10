# Model card

Written last. What was built, what was found, and where it fails.

## What this is

- Model: <estimator and the pipeline around it>
- Trained on: <split, date range, row count>
- Artefact: `models/<name>.joblib`, rebuilt by `notebooks/03-features.ipynb`

## Intended use

<The decision from PROBLEM.md, and nothing else.>

## Not for

<Uses this model will look fine on and be wrong about.>

## How it scores

| Model | <metric> | vs. previous rung |
|---|---|---|
| Constant | | |
| Hand-written rule | | |
| Simple fitted model | | |
| Chosen model | | |

Held-back set (<July>), scored once: <number>.

## Where it fails

| Slice | n | <metric> | Note |
|---|---|---|---|

<The groups where the average hides a bad result. Name them plainly.>

## What it is sensitive to

<Inputs that move the prediction most, and the ones that should not.>

## Decisions that shape the result

<Target definition, cleaning decisions, the split — each with a pointer to
the notebook that records it.>

## What I would do next

<The one or two things most likely to improve it, and what is not worth trying.>
