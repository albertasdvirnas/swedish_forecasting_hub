# Automating a model from an external GitHub repository

This guide describes the recommended weekly workflow for a model whose code is
kept in a separate GitHub repository. The automation runs in the forecast hub,
so its built-in `GITHUB_TOKEN` can create a branch and pull request without a
personal access token.

## 1. Prepare the model repository

The repository should contain a reproducible command that:

1. reads the hub's `target-data/time-series.csv`;
2. uses only observations with `target_end_date <= reference_date - 7 days`;
3. calibrates or refits the model from those observations;
4. writes forecasts for horizons 0–3 in the hub format.

The easiest approach is to adapt the hub's
`submission-tools/model_template.py` or `model_template.R`. Keep dependency
versions in `pyproject.toml`, `requirements.txt`, `renv.lock`, or a comparable
lock file. If fitting is stochastic, set and document a fixed random seed.

Test the command locally from a clone of the hub. For example:

```bash
python path/to/model_template.py 2026-11-08 team-model \
  --output model-output/team-model/2026-11-08-team-model.csv
```

The model metadata file must already exist at
`model-metadata/team-model.yml`.

## 2. Add a hub-side GitHub Actions workflow

Ask the hub operator to copy `.github/workflows/run-gerlee-history.yml` to a
new file and replace:

- the model ID (`gerlee-history`);
- the external `repository: owner/model-repository`;
- the model installation command;
- the command that runs the model.

The workflow is triggered when `target-data/time-series.csv` is merged into
`main`. It performs the following steps:

```text
new target data merged
        ↓
check out hub and model repositories
        ↓
install dependencies and calibrate/run the model
        ↓
validate metadata and forecast CSV
        ↓
push a new branch and open a pull request
```

The workflow should retain `workflow_dispatch` as a manual fallback. Its
permissions must include:

```yaml
permissions:
  contents: write
  pull-requests: write
```

In **Settings → Actions → General**, the hub must use read/write workflow
permissions and allow GitHub Actions to create pull requests. This is a one-time
repository setting controlled by the hub operator.

Use a concurrency group with `cancel-in-progress: false` so one weekly run is
not interrupted by another. Before generating a forecast, check whether the
expected output file already exists; this makes reruns safe and avoids duplicate
pull requests.

## 3. Public and private model repositories

No extra credential is needed to check out a public model repository. For a
private repository, create a fine-grained token or GitHub App credential with
read-only access to that repository, save it as a hub repository secret, and
pass it only to the external-model checkout step. Do not store tokens in code
or forecast files.

Pin the model checkout to a protected branch, tag, or commit. A branch such as
`main` is convenient for weekly updates; a tag or commit gives stricter
reproducibility. The pull request should record the checked-out model commit.

## 4. Weekly operation

After new target data is merged, GitHub Actions downloads both repositories,
calibrates the model, creates the standard CSV, runs the hub validator, and
opens a pull request. The hub operator reviews the validation result and merges
the PR. Merging then refreshes the dashboard automatically.

If calibration or validation fails, no forecast PR is opened. Open the failed
workflow under the repository's **Actions** tab, correct the model repository,
and use **Run workflow** to retry the same `reference_date`.
