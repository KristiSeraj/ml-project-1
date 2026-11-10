# Problem card

Written before any modelling. Kept up to date: if the work contradicts this
file, this file gets fixed in the same commit.

## The question

<One sentence, in the words of whoever asks it.>

## Who decides what, with the answer

<Who acts on a prediction, what they do differently, and when they need it.>

## The target

- Column: `<name>`
- Definition: <the exact rule, including the cutoff and the time window>
- Missing means: <excluded / treated as negative / imputed — and why>
- Excluded rows: <which, and why>

## The metric

- Headline: <metric, and why this one>
- Cost of a false positive: <...>
- Cost of a false negative: <...>
- Worth shipping if: <the threshold, named now, not after seeing the scores>

## The data

- Source: <where it comes from, fetched by `data/get_data.py`>
- Grain: one row per <...>
- Development set: <June>
- Held back, looked at once: <July>

## The baseline ladder

1. Constant / majority class
2. <one hand-written rule>
3. <a simple fitted model on raw columns>

A model earns its place only by beating the rung below it.

## Known constraints

<Latency, interpretability, fairness, data that will not exist at
prediction time, anything that rules an approach out before it is tried.>
