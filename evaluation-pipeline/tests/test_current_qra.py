import unittest
from datetime import date, timedelta

from influenza_evaluation.current_qra import generate_current_qra


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

        self.assertEqual(len(forecasts), 7)
        self.assertFalse(skipped)
        self.assertTrue(coefficients)
        self.assertEqual({row["component_model_count"] for row in forecasts}, {1})
        self.assertEqual(
            [float(row["output_type_id"]) for row in forecasts],
            [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975],
        )


if __name__ == "__main__":
    unittest.main()
