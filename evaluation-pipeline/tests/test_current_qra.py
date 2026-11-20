import unittest
from datetime import date, timedelta

from influenza_evaluation.current_qra import (
    _generate_legacy_current_qra,
    generate_current_qra,
)


class CurrentQraTests(unittest.TestCase):
    def test_builds_all_quantiles_for_one_available_component(self):
        matched = []
        start = date(2025, 10, 5)
        for index in range(8):
            reference = start + timedelta(weeks=index)
            matched.append(
                {
                    "model_id": "team-model",
                    "reference_date": reference.isoformat(),
                    "target_end_date": reference.isoformat(),
                    "target": "weekly incident influenza cases",
                    "horizon": "0",
                    "location": "SE",
                    "location_name": "Sverige",
                    "forecast_value": str(10 + index),
                    "observed_value": str(12 + index),
                    "outcome_data_version": "2026-08-25T00:00:00Z",
                }
            )
        current = [
            {
                "model_id": "team-model",
                "reference_date": "2026-10-04",
                "target_end_date": "2026-10-04",
                "target": "weekly incident influenza cases",
                "horizon": 0,
                "location": "SE",
                "location_name": "Sverige",
                "forecast_value": 20.0,
                "forecast_file": "team-model/2026-10-04-team-model.csv",
            }
        ]

        forecasts, coefficients, skipped = generate_current_qra(matched, current)
        legacy = _generate_legacy_current_qra(matched, current)

        self.assertEqual(len(forecasts), 7)
        self.assertFalse(skipped)
        self.assertTrue(coefficients)
        self.assertEqual({row["component_model_count"] for row in forecasts}, {1})
        self.assertEqual(
            [float(row["output_type_id"]) for row in forecasts],
            [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975],
        )
        self.assertEqual((forecasts, coefficients, skipped), legacy)
        frozen = [dict(row) for row in forecasts]
        frozen_result = generate_current_qra(
            matched, current, frozen_forecasts=frozen
        )
        self.assertEqual(frozen_result, (frozen, [], []))

    def test_pooled_round_shares_coefficients_across_tasks(self):
        matched = []
        start = date(2025, 10, 5)
        locations = (("SE", "Sverige", 0), ("SE-M", "Region Skåne", 5))
        for index in range(10):
            reference = start + timedelta(weeks=index)
            for location, location_name, location_offset in locations:
                for horizon in (0, 1):
                    baseline = 20 + index + horizon + location_offset
                    for model_id, adjustment in (
                        ("alpha-model", -1),
                        ("beta-model", 2),
                    ):
                        matched.append(
                            {
                                "model_id": model_id,
                                "reference_date": reference.isoformat(),
                                "target_end_date": (
                                    reference + timedelta(weeks=horizon)
                                ).isoformat(),
                                "target": "weekly incident influenza cases",
                                "horizon": str(horizon),
                                "location": location,
                                "location_name": location_name,
                                "forecast_value": str(baseline + adjustment),
                                "observed_value": str(baseline),
                                "outcome_data_version": "fixed-version",
                            }
                        )
        current = []
        reference = date(2026, 11, 22)
        for location, location_name, location_offset in locations:
            for horizon in (0, 1):
                baseline = 40 + horizon + location_offset
                for model_id, adjustment in (("alpha-model", -1), ("beta-model", 2)):
                    current.append(
                        {
                            "model_id": model_id,
                            "reference_date": reference.isoformat(),
                            "target_end_date": (
                                reference + timedelta(weeks=horizon)
                            ).isoformat(),
                            "target": "weekly incident influenza cases",
                            "horizon": horizon,
                            "location": location,
                            "location_name": location_name,
                            "forecast_value": baseline + adjustment,
                            "forecast_file": f"{model_id}/file.csv",
                        }
                    )

        forecasts, coefficients, skipped = generate_current_qra(matched, current)

        self.assertFalse(skipped)
        self.assertEqual(len(forecasts), 4 * 7)
        median_coefficients = [
            row for row in coefficients if row["output_type_id"] == "0.5"
        ]
        for term in ("intercept", "alpha-model", "beta-model"):
            self.assertEqual(
                len(
                    {
                        row["coefficient"]
                        for row in median_coefficients
                        if row["term"] == term
                    }
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
