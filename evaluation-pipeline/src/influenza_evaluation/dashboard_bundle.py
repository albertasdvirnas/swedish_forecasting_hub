"""Build compact retrospective and current data files for the dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from .current_qra import read_current_points
from .score import read_matches


CURRENT_POINT_COLUMNS = (
    "model_id",
    "reference_date",
    "target_end_date",
    "target",
    "horizon",
    "location",
    "location_name",
    "forecast_value",
    "forecast_file",
)


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def model_metadata(metadata_root: Path, model_ids: set[str]) -> list[dict[str, Any]]:
    result = []
    for model_id in sorted(model_ids):
        path = metadata_root / f"{model_id}.yml"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
        result.append(
            {
                "model_id": model_id,
                "model_name": data.get("model_name", model_id),
                "team_name": data.get("team_name", ""),
                "method_category": data.get("method_category", ""),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-input", type=Path, required=True)
    parser.add_argument("--retrospective-qra", type=Path, required=True)
    parser.add_argument("--current-qra", type=Path, required=True)
    parser.add_argument("--model-output-root", type=Path, default=Path("model-output"))
    parser.add_argument("--metadata-root", type=Path, default=Path("model-metadata"))
    parser.add_argument(
        "--retrospective-outcomes",
        type=Path,
        default=Path("retrospective-data/2025-2026/final-outcomes.csv"),
    )
    parser.add_argument(
        "--current-outcomes",
        type=Path,
        default=Path("target-data/time-series.csv"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    matched = read_matches(args.matched_input)
    current = read_current_points(args.model_output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.matched_input, args.output_root / "retrospective-point.csv")
    shutil.copyfile(args.retrospective_qra, args.output_root / "retrospective-qra.csv")
    shutil.copyfile(args.current_qra, args.output_root / "current-qra.csv")
    shutil.copyfile(
        args.retrospective_outcomes,
        args.output_root / "retrospective-outcomes.csv",
    )
    shutil.copyfile(args.current_outcomes, args.output_root / "current-outcomes.csv")
    write_csv(args.output_root / "current-point.csv", CURRENT_POINT_COLUMNS, current)

    retrospective_models = {row["model_id"] for row in matched}
    current_models = {row["model_id"] for row in current}
    metadata = model_metadata(
        args.metadata_root, retrospective_models | current_models
    )
    (args.output_root / "models.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        **model,
                        "has_retrospective": model["model_id"] in retrospective_models,
                        "has_current": model["model_id"] in current_models,
                    }
                    for model in metadata
                ]
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    retrospective_dates = sorted({row["reference_date"] for row in matched})
    current_dates = sorted({row["reference_date"] for row in current})
    (args.output_root / "manifest.json").write_text(
        json.dumps(
            {
                "retrospective": {
                    "season": "2025/26",
                    "reference_dates": retrospective_dates,
                    "model_count": len(retrospective_models),
                },
                "current": {
                    "season": "2026/27",
                    "reference_dates": current_dates,
                    "model_count": len(current_models),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Built dashboard bundle for {len(retrospective_models)} retrospective "
        f"and {len(current_models)} current models"
    )


if __name__ == "__main__":
    main()
