from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from influenza_evaluation.match import (
    FORECAST_COLUMNS,
    match_forecasts,
    write_matches,
    write_report,
)


class MatchTests(unittest.TestCase):
    def setUp(self):
        self.repository = Path(__file__).resolve().parents[2]
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_output_root = self.root / "model-output"
        self.model_id = "example-example"

    def tearDown(self):
        self.temporary.cleanup()

    def write_forecast(self, reference_date: str, *, omit_last: bool = False) -> Path:
        path = (
            self.model_output_root
            / self.model_id
            / f"{reference_date}-{self.model_id}.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "reference_date": reference_date,
                "target": "weekly incident influenza cases",
                "horizon": horizon,
                "location": location,
                "output_type": "mean",
                "output_type_id": "",
                "value": 10 + horizon,
            }
            for location in ("SE", "SE-M", "SE-O")
            for horizon in range(4)
        ]
        if omit_last:
            rows.pop()
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FORECAST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def match(
        self,
        *,
        require_complete_models: bool = False,
        exclude_model_prefixes: tuple[str, ...] = (),
    ):
        return match_forecasts(
            model_output_root=self.model_output_root,
            manifest_path=(
                self.repository / "retrospective-data/2025-2026/manifest.csv"
            ),
            outcomes_path=(
                self.repository
                / "retrospective-data/2025-2026/final-outcomes.csv"
            ),
            require_complete_models=require_complete_models,
            exclude_model_prefixes=exclude_model_prefixes,
        )

    def test_matches_horizons_to_the_correct_final_outcomes(self):
        self.write_forecast("2025-10-05")
        self.write_forecast("2026-05-17")
        rows, files = self.match()
        self.assertEqual(len(files), 2)
        self.assertEqual(len(rows), 24)

        first = next(
            row
            for row in rows
            if row["reference_date"] == "2025-10-05"
            and row["location"] == "SE"
            and row["horizon"] == 0
        )
        self.assertEqual(first["target_end_date"], "2025-10-05")
        self.assertEqual(first["observed_value"], 29.0)

        final = next(
            row
            for row in rows
            if row["reference_date"] == "2026-05-17"
            and row["location"] == "SE"
            and row["horizon"] == 3
        )
        self.assertEqual(final["target_end_date"], "2026-06-07")
        self.assertEqual(final["outcome_source_year_week"], "2026W23")
        self.assertEqual(final["observed_value"], 19.0)

    def test_rejects_an_incomplete_historical_file(self):
        self.write_forecast("2025-10-05", omit_last=True)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.match()

    def test_rejects_a_model_missing_historical_rounds(self):
        self.write_forecast("2025-10-05")
        with self.assertRaisesRegex(ValueError, "missing 32 round"):
            self.match(require_complete_models=True)

    def test_ignores_valid_live_rounds(self):
        self.write_forecast("2025-10-05")
        self.write_forecast("2026-10-04")
        rows, files = self.match()
        self.assertEqual(len(files), 1)
        self.assertEqual(len(rows), 12)

    def test_excludes_models_by_prefix(self):
        self.model_id = "hubdemo-example"
        self.write_forecast("2025-10-05")
        self.model_id = "participant-example"
        participant_file = self.write_forecast("2025-10-05")

        rows, files = self.match(exclude_model_prefixes=("hubdemo-",))

        self.assertEqual(files, [participant_file])
        self.assertEqual({row["model_id"] for row in rows}, {"participant-example"})

    def test_writes_machine_readable_outputs(self):
        self.write_forecast("2025-10-05")
        rows, files = self.match()
        csv_path = self.root / "evaluation/matches.csv"
        report_path = self.root / "evaluation/report.json"
        write_matches(csv_path, rows)
        write_report(report_path, rows, files)
        with csv_path.open(encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 12)
        self.assertIn('"matched_row_count": 12', report_path.read_text())


if __name__ == "__main__":
    unittest.main()
