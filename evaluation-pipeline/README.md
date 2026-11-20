# Retrospective forecast matching

This package performs the first stages of retrospective evaluation. It reads
merged historical model submissions, joins every point forecast to the fixed
2025/2026 outcome, calculates individual-model point scores and fits the QRA
ensemble. Probabilistic QRA scoring is added separately.

The matching rule is:

```text
target_end_date = reference_date + 7 × horizon days
```

Thus, horizon 0 is the ISO week ending on `reference_date`, while horizon 3
ends three Sundays later. The final week-20 round therefore uses outcomes
through ISO week 23.

## Run locally

```bash
python -m pip install -e evaluation-pipeline
match-retrospective-forecasts
```

The command discovers historical CSV files under `model-output/`, verifies that
each model has all 33 rounds and each file contains all three locations and
horizons 0–3, and writes:

- `evaluation-output/2025-2026/matched-forecasts.csv`
- `evaluation-output/2025-2026/match-report.json`

No output is written if a forecast is incomplete, invalid or lacks a final
outcome. Before participant forecasts have been merged, `--allow-empty` can be
used to create header-only artifacts.

## Score individual point forecasts

After matching, run:

```bash
score-retrospective-point-forecasts
```

The scoring command writes:

- `point-errors.csv`, with signed, absolute and squared error for every row;
- `point-metrics-by-location-horizon.csv`, with MAE, bias and RMSE separately
  for every model, location and horizon;
- `point-metrics-by-forecast-kind.csv`, comparing horizon-0 nowcasts with
  horizons 1–3 forecasts for every model and location;
- `point-score-report.json`, an audit summary explicitly recording that no
  ranking was created.

Bias is defined as `forecast_value - observed_value`: positive bias means
systematic overprediction and negative bias means systematic underprediction.
The output contains no overall model ranking.

## Build the retrospective QRA

The ensemble is LASSO Quantile Regression Averaging (LQRA): each individual
model forecast is a separate covariate, with an L1 penalty on model
coefficients and an unpenalized intercept. Run:

```bash
build-retrospective-qra
```

For each forecast origin, one coefficient vector per quantile is shared by all
locations and horizons. Each historical QRA forecast is trained only on pooled
outcomes that would already have been published by that forecast's
`reference_date`. A task is published only after its location and horizon have
at least eight such outcomes. The regularization strength is selected
separately for each quantile using temporally ordered expanding-origin pinball
loss.

For the live 2026/2027 season, QRA forecasts through 15 November 2026 are kept
as a frozen snapshot. The pooled specification starts with the forecast origin
22 November 2026, so previously published forecasts are not revised.

The seven quantiles 0.025, 0.1, 0.25, 0.5, 0.75, 0.9 and 0.975 define the
central 50%, 80% and 95% prediction intervals. Negative predictions are
truncated at zero. Quantiles are rearranged into monotone order when necessary,
and every adjustment is recorded.

The command writes:

- `qra-forecasts.csv`, with quantiles and training provenance;
- `qra-coefficients.csv`, with the fitted intercept and model coefficients;
- `qra-skipped-tasks.json`, listing early tasks with fewer than eight outcomes;
- `qra-report.json`, documenting the complete method and first usable round.

## Score the probabilistic QRA

After building the QRA, run:

```bash
score-retrospective-qra
```

The weighted interval score (WIS) uses the median and the central 50%, 80% and
95% intervals. With interval miscoverage levels $\alpha_k$ and three central
intervals, it is calculated as:

```text
WIS = [0.5 * absolute median error
       + sum(alpha_k / 2 * interval_score_k)] / 3.5
```

The empirical coverage for each central interval includes observations exactly
on an interval boundary. WIS is the primary probabilistic score; interval width
is not reported separately. The command writes:

- `qra-scores.csv`, with WIS and three coverage indicators for every QRA task;
- `qra-metrics-by-location-horizon.csv`, with mean WIS and empirical coverage;
- `qra-score-report.json`, documenting the formula and evaluation choices.

The implementation follows [Bracher et al. (2021), *Evaluating epidemic
forecasts in an interval format*](https://doi.org/10.1371/journal.pcbi.1008618).

## Build and compare the probabilistic baseline

The dashboard comparison uses `hub-normalma3`, a deliberately simple
probabilistic reference model. Its centre is the arithmetic mean of the latest
three consecutive weekly observations available in that retrospective round.
For each location and horizon, its normal predictive standard deviation is the
historical rolling-origin RMSE around zero. Only errors whose outcomes were
available by the current round's data cutoff enter the estimate. Quantiles below
zero are truncated at zero.

Build the baseline and optionally export one Hubverse-compatible model-output
file per round:

```bash
build-retrospective-probabilistic-baseline \
  --hubverse-output-root /tmp/hubverse-model-output
```

The Hubverse export has the task columns required by PredTimeChart, including
the derived `target_end_date`, and the same seven quantiles as the QRA. It is a
generated dashboard view and does not change the participant submission format.

Compare the QRA and baseline on the exact intersection of available tasks:

```bash
compare-retrospective-probabilistic-forecasts
```

The principal relative result is:

```text
WIS skill = 1 - mean QRA WIS / mean baseline WIS
```

A positive value means that the QRA has lower mean WIS. The comparison also
reports empirical 50%, 80% and 95% coverage for both models. Interval width is
not reported separately.

The generated artifacts are:

- `normal-ma3-baseline-forecasts.csv` and its method report;
- `normal-ma3-baseline-scores.csv` and location-by-horizon metrics;
- `probabilistic-comparison.csv`, containing task-level paired results;
- `probabilistic-comparison-by-location-horizon.csv`, containing relative WIS,
  WIS skill and coverage;
- `probabilistic-comparison-report.json`, documenting the comparison contract.

## Build the Hubverse dashboard view

The participant hub intentionally keeps its submission format small: individual
models submit means and do not need to calculate `target_end_date`. The Hubverse
PredTimeChart requires quantile models, explicit target dates and enumerated
historical rounds. Generate that derived view with:

```bash
build-dashboard-hub-view --output-root /tmp/dashboard-hub-view
```

The command creates a temporary, self-contained Hubverse hub containing the QRA
and normal-MA3 baseline, frozen demonstration outcomes, dashboard-specific task
configuration, target JSON for every retrospective round and a ready-to-use
`predtimechart-config.yml`. It does not modify the canonical hub or participant
submissions. The separate dashboard repository builds this view afresh before
generating its visualization data.

Method references:

- [Nowotarski and Weron (2015), *Computing electricity spot price prediction
  intervals using quantile regression and forecast averaging*](https://doi.org/10.1007/s00180-014-0523-0).
- [Uniejewski and Weron (2021), *Regularized quantile regression averaging for
  probabilistic electricity price forecasting*](https://doi.org/10.1016/j.eneco.2021.105121).

## Development tests

```bash
python -m unittest discover -s evaluation-pipeline/tests -v
```
