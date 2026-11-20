"""Build current-season QRA forecasts from retrospective training data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .qra import (
    ALPHAS,
    COEFFICIENT_COLUMNS,
    MIN_TRAINING_SAMPLES,
    QRA_COLUMNS,
    QUANTILES,
    MIN_CV_TRAINING_SAMPLES,
    _pinball_loss,
    _scaled_fit,
    write_csv,
)
from .score import read_matches

LIVE_START = date.fromisocalendar(2026, 40, 7)
LIVE_END = date.fromisocalendar(2027, 20, 7)
TARGET = "weekly incident influenza cases"
LOCATIONS = {
    "SE": "Sverige",
    "SE-M": "Region Skåne",
    "SE-O": "Västra Götalandsregionen",
}
HORIZONS = (0, 1, 2, 3)
RECENT_CV_ORIGINS = 5
POOLED_QRA_START = date(2026, 11, 22)
FILE_PATTERN = re.compile(
    r"(?P<reference>\d{4}-\d{2}-\d{2})-(?P<model>[A-Za-z0-9_+]+-[A-Za-z0-9_+]+)\.csv"
)


def read_current_points(model_output_root: Path) -> list[dict[str, Any]]:
    """Read valid point forecasts for the live 2026/27 season."""
    result: list[dict[str, Any]] = []
    for path in sorted(model_output_root.glob("*/*.csv")):
        match = FILE_PATTERN.fullmatch(path.name)
        if not match:
            continue
        reference = date.fromisoformat(match.group("reference"))
        if not LIVE_START <= reference <= LIVE_END:
            continue
        model_id = match.group("model")
        if path.parent.name != model_id:
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row.get("output_type") != "mean" or row.get("target") != TARGET:
                continue
            location = row.get("location", "")
            horizon = int(row["horizon"])
            value = float(row["value"])
            if location not in LOCATIONS or horizon not in HORIZONS:
                raise ValueError(f"Invalid task in {path}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid forecast value in {path}")
            result.append(
                {
                    "model_id": model_id,
                    "reference_date": reference.isoformat(),
                    "target_end_date": (reference + timedelta(weeks=horizon)).isoformat(),
                    "target": TARGET,
                    "horizon": horizon,
                    "location": location,
                    "location_name": LOCATIONS[location],
                    "forecast_value": value,
                    "forecast_file": path.relative_to(model_output_root).as_posix(),
                }
            )
    return sorted(
        result,
        key=lambda row: (
            row["reference_date"],
            row["location"],
            row["horizon"],
            row["model_id"],
        ),
    )


def _generate_legacy_current_qra(
    matched_rows: list[dict[str, str]],
    current_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    historical: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in matched_rows:
        key = (row["location"], int(row["horizon"]), row["reference_date"])
        cell = historical.setdefault(
            key,
            {
                "observed": float(row["observed_value"]),
                "target_end_date": row["target_end_date"],
                "outcome_data_version": row["outcome_data_version"],
                "forecasts": {},
            },
        )
        cell["forecasts"][row["model_id"]] = float(row["forecast_value"])

    current: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in current_rows:
        key = (row["location"], int(row["horizon"]), row["reference_date"])
        cell = current.setdefault(
            key,
            {
                "target_end_date": row["target_end_date"],
                "forecasts": {},
            },
        )
        cell["forecasts"][row["model_id"]] = float(row["forecast_value"])

    forecasts: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for (location, horizon, reference), current_cell in sorted(current.items()):
        models = tuple(sorted(current_cell["forecasts"]))
        training = [
            cell
            for (train_location, train_horizon, _), cell in sorted(historical.items())
            if train_location == location
            and train_horizon == horizon
            and all(model in cell["forecasts"] for model in models)
        ]
        if not models or len(training) < MIN_TRAINING_SAMPLES:
            skipped.append(
                {
                    "reference_date": reference,
                    "location": location,
                    "horizon": horizon,
                    "component_models": list(models),
                    "training_sample_count": len(training),
                    "reason": "insufficient_components_or_training_data",
                }
            )
            continue
        x = np.asarray(
            [[cell["forecasts"][model] for model in models] for cell in training],
            dtype=float,
        )
        y = np.asarray([cell["observed"] for cell in training], dtype=float)
        x_new = np.asarray([current_cell["forecasts"][model] for model in models])
        fits = []
        for quantile in QUANTILES:
            first_validation = max(
                MIN_CV_TRAINING_SAMPLES, len(y) - RECENT_CV_ORIGINS
            )
            losses = {alpha: [] for alpha in ALPHAS}
            for validation_index in range(first_validation, len(y)):
                for alpha_candidate in ALPHAS:
                    validation_fit = _scaled_fit(
                        x[:validation_index],
                        y[:validation_index],
                        x[validation_index],
                        quantile=quantile,
                        alpha=alpha_candidate,
                    )
                    losses[alpha_candidate].append(
                        _pinball_loss(
                            float(y[validation_index]),
                            validation_fit.prediction,
                            quantile,
                        )
                    )
            alpha = min(
                ALPHAS,
                key=lambda candidate: (
                    sum(losses[candidate]) / len(losses[candidate]),
                    -candidate,
                ),
            )
            fits.append(
                _scaled_fit(x, y, x_new, quantile=quantile, alpha=alpha)
            )
        nonnegative = [max(0.0, fit.prediction) for fit in fits]
        ordered = sorted(nonnegative)
        crossing_adjusted = any(
            not math.isclose(raw, adjusted, rel_tol=1e-12, abs_tol=1e-12)
            for raw, adjusted in zip(nonnegative, ordered)
        )
        last_training = max(cell["target_end_date"] for cell in training)
        version = max(cell["outcome_data_version"] for cell in training)
        for quantile, fit, value in zip(QUANTILES, fits, ordered):
            forecasts.append(
                {
                    "reference_date": reference,
                    "target_end_date": current_cell["target_end_date"],
                    "target": TARGET,
                    "horizon": horizon,
                    "location": location,
                    "location_name": LOCATIONS[location],
                    "output_type": "quantile",
                    "output_type_id": format(quantile, "g"),
                    "value": format(value, ".12g"),
                    "raw_value": format(fit.prediction, ".12g"),
                    "training_sample_count": len(training),
                    "training_last_target_end_date": last_training,
                    "component_model_count": len(models),
                    "component_models": "|".join(models),
                    "selected_alpha": format(fit.alpha, "g"),
                    "quantile_crossing_adjusted": str(crossing_adjusted).lower(),
                    "outcome_data_version": version,
                }
            )
            for term, coefficient in [
                ("intercept", fit.intercept), *zip(models, fit.coefficients)
            ]:
                coefficients.append(
                    {
                        "reference_date": reference,
                        "horizon": horizon,
                        "location": location,
                        "output_type_id": format(quantile, "g"),
                        "term": term,
                        "coefficient": format(coefficient, ".12g"),
                        "selected_alpha": format(fit.alpha, "g"),
                        "training_sample_count": len(training),
                    }
                )
    return forecasts, coefficients, skipped


def _fit_pooled_quantiles(
    historical: dict[tuple[str, int, str], dict[str, Any]],
    models: tuple[str, ...],
):
    training = [
        (key, cell)
        for key, cell in historical.items()
        if all(model in cell["forecasts"] for model in models)
    ]
    training.sort(
        key=lambda item: (
            item[1]["target_end_date"], item[0][2], item[0][0], item[0][1]
        )
    )
    if len(training) < MIN_TRAINING_SAMPLES:
        return None, training

    losses = {quantile: {alpha: [] for alpha in ALPHAS} for quantile in QUANTILES}
    origins = sorted({reference for (location, horizon, reference), _ in training})
    for origin_text in origins:
        origin = date.fromisoformat(origin_text)
        cutoff = origin - timedelta(days=7)
        previous = [
            item
            for item in training
            if date.fromisoformat(item[1]["target_end_date"]) <= cutoff
        ]
        validation = [item for item in training if item[0][2] == origin_text]
        if len(previous) < MIN_CV_TRAINING_SAMPLES:
            continue
        x_previous = np.asarray(
            [
                [cell["forecasts"][model] for model in models]
                for _, cell in previous
            ],
            dtype=float,
        )
        y_previous = np.asarray(
            [cell["observed"] for _, cell in previous], dtype=float
        )
        first_x = np.asarray(
            [validation[0][1]["forecasts"][model] for model in models], dtype=float
        )
        for quantile in QUANTILES:
            for alpha in ALPHAS:
                fitted = _scaled_fit(
                    x_previous,
                    y_previous,
                    first_x,
                    quantile=quantile,
                    alpha=alpha,
                )
                fitted_coefficients = np.asarray(fitted.coefficients, dtype=float)
                for _, cell in validation:
                    x_new = np.asarray(
                        [cell["forecasts"][model] for model in models], dtype=float
                    )
                    prediction = fitted.intercept + float(
                        np.dot(fitted_coefficients, x_new)
                    )
                    losses[quantile][alpha].append(
                        _pinball_loss(cell["observed"], prediction, quantile)
                    )

    x = np.asarray(
        [[cell["forecasts"][model] for model in models] for _, cell in training],
        dtype=float,
    )
    y = np.asarray([cell["observed"] for _, cell in training], dtype=float)
    first_x = x[0]
    fits = {}
    for quantile in QUANTILES:
        alpha = min(
            ALPHAS,
            key=lambda candidate: (
                sum(losses[quantile][candidate])
                / len(losses[quantile][candidate]),
                -candidate,
            ),
        )
        fits[quantile] = _scaled_fit(
            x, y, first_x, quantile=quantile, alpha=alpha
        )
    return fits, training


def _generate_pooled_current_qra(
    matched_rows: list[dict[str, str]],
    current_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    historical: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in matched_rows:
        key = (row["location"], int(row["horizon"]), row["reference_date"])
        cell = historical.setdefault(
            key,
            {
                "observed": float(row["observed_value"]),
                "target_end_date": row["target_end_date"],
                "outcome_data_version": row["outcome_data_version"],
                "forecasts": {},
            },
        )
        cell["forecasts"][row["model_id"]] = float(row["forecast_value"])

    current: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in current_rows:
        key = (row["location"], int(row["horizon"]), row["reference_date"])
        cell = current.setdefault(
            key,
            {"target_end_date": row["target_end_date"], "forecasts": {}},
        )
        cell["forecasts"][row["model_id"]] = float(row["forecast_value"])

    forecasts: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    references = sorted({reference for _, _, reference in current})
    fit_cache = {}
    for reference in references:
        tasks = [
            (key, cell) for key, cell in current.items() if key[2] == reference
        ]
        component_sets = [set(cell["forecasts"]) for _, cell in tasks]
        models = tuple(sorted(set.intersection(*component_sets))) if tasks else ()
        if not models:
            for (location, horizon, _), _ in tasks:
                skipped.append(
                    {
                        "reference_date": reference,
                        "location": location,
                        "horizon": horizon,
                        "component_models": [],
                        "training_sample_count": 0,
                        "reason": "no_models_available_for_every_current_task",
                    }
                )
            continue
        if models not in fit_cache:
            fit_cache[models] = _fit_pooled_quantiles(historical, models)
        fits, training = fit_cache[models]
        if fits is None:
            for (location, horizon, _), _ in tasks:
                skipped.append(
                    {
                        "reference_date": reference,
                        "location": location,
                        "horizon": horizon,
                        "component_models": list(models),
                        "training_sample_count": len(training),
                        "reason": "insufficient_components_or_training_data",
                    }
                )
            continue

        last_training = max(cell["target_end_date"] for _, cell in training)
        version = max(cell["outcome_data_version"] for _, cell in training)
        for (location, horizon, _), current_cell in sorted(tasks):
            x_new = np.asarray(
                [current_cell["forecasts"][model] for model in models], dtype=float
            )
            raw_predictions = []
            for quantile in QUANTILES:
                fit = fits[quantile]
                raw_predictions.append(
                    fit.intercept
                    + float(np.dot(np.asarray(fit.coefficients), x_new))
                )
            nonnegative = [max(0.0, prediction) for prediction in raw_predictions]
            ordered = sorted(nonnegative)
            crossing_adjusted = any(
                not math.isclose(raw, adjusted, rel_tol=1e-12, abs_tol=1e-12)
                for raw, adjusted in zip(nonnegative, ordered)
            )
            for quantile, value in zip(QUANTILES, ordered):
                fit = fits[quantile]
                raw_value = fit.intercept + float(
                    np.dot(np.asarray(fit.coefficients), x_new)
                )
                forecasts.append(
                    {
                        "reference_date": reference,
                        "target_end_date": current_cell["target_end_date"],
                        "target": TARGET,
                        "horizon": horizon,
                        "location": location,
                        "location_name": LOCATIONS[location],
                        "output_type": "quantile",
                        "output_type_id": format(quantile, "g"),
                        "value": format(value, ".12g"),
                        "raw_value": format(raw_value, ".12g"),
                        "training_sample_count": len(training),
                        "training_last_target_end_date": last_training,
                        "component_model_count": len(models),
                        "component_models": "|".join(models),
                        "selected_alpha": format(fit.alpha, "g"),
                        "quantile_crossing_adjusted": str(crossing_adjusted).lower(),
                        "outcome_data_version": version,
                    }
                )
                for term, coefficient in [
                    ("intercept", fit.intercept), *zip(models, fit.coefficients)
                ]:
                    coefficients.append(
                        {
                            "reference_date": reference,
                            "horizon": horizon,
                            "location": location,
                            "output_type_id": format(quantile, "g"),
                            "term": term,
                            "coefficient": format(coefficient, ".12g"),
                            "selected_alpha": format(fit.alpha, "g"),
                            "training_sample_count": len(training),
                        }
                    )
    return forecasts, coefficients, skipped


def generate_current_qra(
    matched_rows: list[dict[str, str]],
    current_rows: list[dict[str, Any]],
    frozen_forecasts: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    legacy_rows = [
        row
        for row in current_rows
        if date.fromisoformat(row["reference_date"]) < POOLED_QRA_START
    ]
    pooled_rows = [
        row
        for row in current_rows
        if date.fromisoformat(row["reference_date"]) >= POOLED_QRA_START
    ]
    if frozen_forecasts is None:
        legacy = _generate_legacy_current_qra(matched_rows, legacy_rows)
    else:
        invalid_dates = sorted(
            {
                row["reference_date"]
                for row in frozen_forecasts
                if date.fromisoformat(row["reference_date"]) >= POOLED_QRA_START
            }
        )
        if invalid_dates:
            raise ValueError(
                "Frozen current QRA contains dates at or after the cutover: "
                + ", ".join(invalid_dates)
            )
        legacy = (list(frozen_forecasts), [], [])
    pooled = _generate_pooled_current_qra(matched_rows, pooled_rows)
    forecasts = [*legacy[0], *pooled[0]]
    coefficients = [*legacy[1], *pooled[1]]
    return forecasts, coefficients, [*legacy[2], *pooled[2]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matched-input",
        type=Path,
        default=Path("evaluation-output/2025-2026/matched-forecasts.csv"),
    )
    parser.add_argument("--model-output-root", type=Path, default=Path("model-output"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--frozen-before-cutover",
        type=Path,
        default=Path(
            "evaluation-output/2026-2027/current-qra-through-2026-11-15.csv"
        ),
    )
    args = parser.parse_args()
    matched = read_matches(args.matched_input)
    current = read_current_points(args.model_output_root)
    with args.frozen_before_cutover.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        frozen_forecasts = list(csv.DictReader(handle))
    forecasts, coefficients, skipped = generate_current_qra(
        matched, current, frozen_forecasts=frozen_forecasts
    )
    write_csv(args.output_root / "current-qra.csv", QRA_COLUMNS, forecasts)
    write_csv(
        args.output_root / "current-qra-coefficients.csv",
        COEFFICIENT_COLUMNS,
        coefficients,
    )
    (args.output_root / "current-qra-report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "forecast_rows": len(forecasts),
                "task_count": len(forecasts) // len(QUANTILES),
                "skipped": skipped,
                "training_season": "2025/26",
                "component_rule": (
                    "before 2026-11-22: models available for each task; "
                    "from 2026-11-22: models available for every task in the round"
                ),
                "pooling_rule": (
                    "from 2026-11-22 one coefficient vector per quantile is shared "
                    "by all locations and horizons"
                ),
                "pooled_qra_start": POOLED_QRA_START.isoformat(),
                "frozen_forecasts_before_cutover": str(
                    args.frozen_before_cutover
                ),
                "alpha_selection": (
                    "before 2026-11-22: expanding-window pinball loss over the five "
                    "most recent retrospective validation origins; from 2026-11-22: "
                    "expanding-origin pinball loss over the retrospective season"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(forecasts)} current-season QRA quantile rows")


if __name__ == "__main__":
    main()
