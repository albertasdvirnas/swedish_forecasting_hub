import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

from influenza_target_data.config import OUTPUT_COLUMNS
from influenza_target_data.release import (
    is_new_week,
    latest_week,
    render_notes,
)


def write_target_data(path: Path, weeks: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for week in weeks:
            year = int(week[:4])
            week_number = int(week[-2:])
            writer.writerow(
                {
                    "location": "SE",
                    "location_name": "Sverige",
                    "year": year,
                    "week": week_number,
                    "target_end_date": date.fromisocalendar(
                        year, week_number, 7
                    ).isoformat(),
                    "value": 1,
                    "status": "available",
                    "release_status": "on_time",
                    "data_version": "2026-10-08T12:00:00Z",
                    "source_region_code": "00",
                    "source_year_week": week,
                    "official_release_time": "2026-10-08T12:00:00Z",
                }
            )


class ReleaseTests(unittest.TestCase):
    def test_latest_week_uses_iso_week_order_across_years(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "time-series.csv"
            write_target_data(path, ["2026W52", "2027W01"])
            self.assertEqual(latest_week(path), "2027W01")

    def test_only_a_later_week_creates_a_release(self):
        self.assertTrue(is_new_week("2026W40", "2026W41"))
        self.assertFalse(is_new_week("2026W40", "2026W40"))
        self.assertFalse(is_new_week("2026W40", "2026W39"))

    def test_release_notes_link_to_data_and_submission_guide(self):
        notes = render_notes("owner/hub", "2026W41")
        self.assertIn("ISO week **2026W41**", notes)
        self.assertIn("owner/hub/blob/main/target-data/time-series.csv", notes)
        self.assertIn("owner/hub/blob/main/docs/model-submissions.md", notes)


if __name__ == "__main__":
    unittest.main()
