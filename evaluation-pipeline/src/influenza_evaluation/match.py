"""Match retrospective point forecasts to the fixed 2025/2026 outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

FORECAST_COLUMNS = (
    "reference_date",
    "target",
    "horizon",
    "location",
    "output_type",
    "output_type_id",
    "value",
)
MATCHED_COLUMNS = (
    "model_id",
    "round_number",
    "reference_date",
    "target_end_date",
    "target",
    "horizon",
    "location",
    "location_name",
    "output_type",
    "output_type_id",
    "forecast_value",
    "observed_value",
    "outcome_status",
    "outcome_source_year_week",
    "outcome_data_version",
    "forecast_file",
)
TARGET = "weekly incident influenza cases"
LOCATIONS = ("SE", "SE-M", "SE-O")
HORIZONS = (0, 1, 2, 3)
FILE_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})-"
    r"(?P<model>[A-Za-z0-9_+]+-[A-Za-z0-9_+]+)\.csv$"
)


def load_manifest(path: Path) -> dict[date, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"round_number", "reference_date"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Manifest is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest contains no rounds")
    result: dict[date, int] = {}
    for row in rows:
        reference_date = date.fromisoformat(row["reference_date"])
        round_number = int(row["round_number"])
        if reference_date.weekday() != 6:
            raise ValueError(f"Manifest date is not a Sunday: {reference_date}")
        if reference_date in result:
            raise ValueError(f"Duplicate manifest date: {reference_date}")
        result[reference_date] = round_number
    if list(result) != sorted(result):
        raise ValueError("Manifest rounds are not chronological")
    if list(result.values()) != list(range(1, len(result) + 1)):
        raise ValueError("Manifest round numbers must be consecutive from 1")
    return result


def _nonnegative_number(value: str, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} is not numeric: {value!r}") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{context} must be finite and at least zero")
    return result


def load_outcomes(path: Path) -> dict[tuple[str, date], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "location",
            "location_name",
            "target_end_date",
            "value",
            "status",
            "source_year_week",
            "data_version",
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Outcomes are missing columns: {missing}")
        result: dict[tuple[str, date], dict[str, str]] = {}
        for row in reader:
            target_end_date = date.fromisoformat(row["target_end_date"])
            key = (row["location"], target_end_date)
            if key in result:
                raise ValueError(f"Duplicate outcome for {key}")
            if row["status"] != "available":
                raise ValueError(f"Outcome is not available for {key}")
            _nonnegative_number(row["value"], f"Outcome for {key}")
            result[key] = row
    if not result:
        raise ValueError("Outcome file contains no observations")
    return result


def discover_historical_files(
    model_output_root: Path,
    reference_dates: set[date],
) -> list[tuple[Path, str, date]]:
    discovered: dict[tuple[str, date], Path] = {}
    if not model_output_root.is_dir():
        return []
    for path in sorted(model_output_root.glob("*/*.csv")):
        match = FILE_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"Invalid forecast filename: {path}")
        reference_date = date.fromisoformat(match.group("date"))
        if reference_date not in reference_dates:
            continue
        model_id = match.group("model")
        if path.parent.name != model_id:
            raise ValueError(f"Forecast directory does not match model_id: {path}")
        key = (model_id, reference_date)
        if key in discovered:
            raise ValueError(f"Duplicate forecast file for {model_id}, {reference_date}")
        discovered[key] = path
    return [
        (path, model_id, reference_date)
        for (model_id, reference_date), path in sorted(
            discovered.items(), key=lambda item: (item[0][1], item[0][0])
        )
    ]


def match_file(
    path: Path,
    *,
    model_id: str,
    reference_date: date,
    round_number: int,
    outcomes: dict[tuple[str, date], dict[str, str]],
    model_output_root: Path,
) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FORECAST_COLUMNS:
            raise ValueError(f"Unexpected forecast columns in {path}")
        forecasts = list(reader)
    seen: set[tuple[str, int]] = set()
    result: list[dict[str, Any]] = []
    for row_number, row in enumerate(forecasts, start=2):
        context = f"{path}, row {row_number}"
        if row["reference_date"] != reference_date.isoformat():
            raise ValueError(f"reference_date does not match filename in {context}")
        if row["target"] != TARGET:
            raise ValueError(f"Unexpected target in {context}")
        if row["output_type"] != "mean":
            raise ValueError(f"output_type must be mean in {context}")
        if row["output_type_id"].strip() not in {"", "NA", "N/A"}:
            raise ValueError(f"output_type_id must be empty or NA in {context}")
        location = row["location"]
        if location not in LOCATIONS:
            raise ValueError(f"Unsupported location in {context}: {location!r}")
        try:
            horizon = int(row["horizon"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid horizon in {context}") from exc
        if str(horizon) != row["horizon"].strip() or horizon not in HORIZONS:
            raise ValueError(f"Unsupported horizon in {context}: {row['horizon']!r}")
        key = (location, horizon)
        if key in seen:
            raise ValueError(f"Duplicate forecast for {key} in {path}")
        seen.add(key)
        forecast_value = _nonnegative_number(row["value"], f"Forecast in {context}")
        target_end_date = reference_date + timedelta(weeks=horizon)
        outcome_key = (location, target_end_date)
        if outcome_key not in outcomes:
            raise ValueError(f"Missing final outcome for {outcome_key} required by {path}")
        outcome = outcomes[outcome_key]
        result.append(
            {
                "model_id": model_id,
                "round_number": round_number,
                "reference_date": reference_date.isoformat(),
                "target_end_date": target_end_date.isoformat(),
                "target": TARGET,
                "horizon": horizon,
                "location": location,
                "location_name": outcome["location_name"],
                "output_type": "mean",
                "output_type_id": "",
                "forecast_value": forecast_value,
                "observed_value": _nonnegative_number(
                    outcome["value"], f"Outcome for {outcome_key}"
                ),
                "outcome_status": outcome["status"],
                "outcome_source_year_week": outcome["source_year_week"],
                "outcome_data_version": outcome["data_version"],
                "forecast_file": path.relative_to(model_output_root).as_posix(),
            }
        )
    expected = {(location, horizon) for location in LOCATIONS for horizon in HORIZONS}
    if seen != expected:
        raise ValueError(f"Historical file is incomplete; missing {sorted(expected - seen)}: {path}")
    return sorted(
        result,
        key=lambda row: (LOCATIONS.index(row["location"]), row["horizon"]),
    )


def match_forecasts(
    *,
    model_output_root: Path,
    manifest_path: Path,
    outcomes_path: Path,
    require_complete_models: bool = True,
    exclude_model_prefixes: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], list[Path]]:
    manifest = load_manifest(manifest_path)
    outcomes = load_outcomes(outcomes_path)
    files = discover_historical_files(model_output_root, set(manifest))
    files = [
        item
        for item in files
        if not any(item[1].startswith(prefix) for prefix in exclude_model_prefixes)
    ]
    if require_complete_models:
        rounds_by_model: dict[str, set[date]] = {}
        for _, model_id, reference_date in files:
            rounds_by_model.setdefault(model_id, set()).add(reference_date)
        for model_id, submitted_dates in rounds_by_model.items():
            missing_dates = sorted(set(manifest) - submitted_dates)
            if missing_dates:
                raise ValueError(
                    f"Historical model {model_id} is missing {len(missing_dates)} "
                    f"round(s), starting with {missing_dates[0]}"
                )
    rows: list[dict[str, Any]] = []
    for path, model_id, reference_date in files:
        rows.extend(
            match_file(
                path,
                model_id=model_id,
                reference_date=reference_date,
                round_number=manifest[reference_date],
                outcomes=outcomes,
                model_output_root=model_output_root,
            )
        )
    return rows, [path for path, _, _ in files]


def write_matches(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCHED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], files: list[Path]) -> None:
    models = sorted({str(row["model_id"]) for row in rows})
    reference_dates = sorted({str(row["reference_date"]) for row in rows})
    report = {
        "status": "complete",
        "model_count": len(models),
        "models": models,
        "forecast_file_count": len(files),
        "matched_row_count": len(rows),
        "first_reference_date": reference_dates[0] if reference_dates else None,
        "last_reference_date": reference_dates[-1] if reference_dates else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-output-root", type=Path, default=Path("model-output"))
    result.add_argument(
        "--manifest",
        type=Path,
        default=Path("retrospective-data/2025-2026/manifest.csv"),
    )
    result.add_argument(
        "--outcomes",
        type=Path,
        default=Path("retrospective-data/2025-2026/final-outcomes.csv"),
    )
    result.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation-output/2025-2026/matched-forecasts.csv"),
    )
    result.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation-output/2025-2026/match-report.json"),
    )
    result.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write empty artifacts instead of failing when no historical files exist",
    )
    result.add_argument(
        "--exclude-model-prefix",
        action="append",
        default=[],
        help="Exclude model IDs beginning with this prefix (repeatable)",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        rows, files = match_forecasts(
            model_output_root=args.model_output_root,
            manifest_path=args.manifest,
            outcomes_path=args.outcomes,
            require_complete_models=True,
            exclude_model_prefixes=tuple(args.exclude_model_prefix),
        )
        if not files and not args.allow_empty:
            raise ValueError("No retrospective forecast files were found")
        write_matches(args.output, rows)
        write_report(args.report, rows, files)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Retrospective forecast matching failed: {exc}") from exc
    print(f"Matched {len(rows)} rows from {len(files)} forecast files")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
