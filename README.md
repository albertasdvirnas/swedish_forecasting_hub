# Swedish Forecasting Hub

Research pilot for weekly influenza forecasts in Sweden, Region Skåne and
Västra Götaland.

The repository follows [hubverse](https://hubverse.io/) directory and file
conventions. The operational components currently include:

- `target-data/time-series.csv` contains frozen weekly observations.
- `target-data-pipeline/` fetches, transforms and validates data from
  Folkhälsodata.
- `retrospective-data/2025-2026/` contains 33 fixed historical round inputs
  and the final outcomes used for evaluation.
- `hub-config/` defines the modeling task and public model metadata schema.
- `model-output/` and `model-metadata/` receive participant submissions.
- `submission-validator/` validates forecasts, metadata and deadlines.
- `submission-tools/` contains Python and R helpers for preparing all 33
  retrospective forecast files as one batch, plus editable starter programs
  that run a participant model and write standard forecast files.
- `demo-models/` contains six reproducible hub-operated baselines whose
  forecasts remain public as a demonstration of evaluation and ensembling.
- `evaluation-pipeline/` matches retrospective forecasts to the fixed final
  outcomes, calculates stratified point scores and builds regularized QRA.
- `evaluation-output/` receives reproducible matched, scored and ensemble data.
- `.github/workflows/` runs development tests, the scheduled data update and
  pull-request validation.

Participant instructions are available in
[`docs/model-submissions.md`](docs/model-submissions.md), with a Swedish quick
guide in [`docs/model-submissions-sv.md`](docs/model-submissions-sv.md).

The retrospective 2025/2026 exercise and its input files are described in
[`retrospective-data/2025-2026/README.md`](retrospective-data/2025-2026/README.md).

The public visualization component will be added separately.

## Weekly data notifications

Each newly merged target week is published as a GitHub Release. To receive a
notification through your GitHub account, select **Watch → Custom → Releases**
on this repository's main page. GitHub delivers the notification according to
your personal web, email and mobile notification settings.
