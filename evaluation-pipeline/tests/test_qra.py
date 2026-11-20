from __future__ import annotations

import unittest
from datetime import date, timedelta

from influenza_evaluation.qra import QUANTILES, build_panel, generate_qra


def synthetic_matches(
    *,
    rounds: int = 15,
    horizons: tuple[int, ...] = (0, 1, 2, 3),
    locations: tuple[str, ...] = ("SE",),
    final_outcome_offset: float = 0,
) -> list[dict[str, str]]:
    rows = []
    start = date.fromisoformat("2025-10-05")
    for index in range(rounds):
        reference_date = start + timedelta(weeks=index)
        for location_index, location in enumerate(locations):
            for horizon in horizons:
                target_end_date = reference_date + timedelta(weeks=horizon)
                baseline = 20 + 2 * index + horizon + 5 * location_index
                observed = baseline
                if index == rounds - 1:
                    observed += final_outcome_offset
                for model_id, adjustment in (("alpha-model", -2), ("beta-model", 3)):
                    rows.append(
                        {
                            "model_id": model_id,
                            "round_number": str(index + 1),
                            "reference_date": reference_date.isoformat(),
                            "target_end_date": target_end_date.isoformat(),
                            "target": "weekly incident influenza cases",
                            "horizon": str(horizon),
                            "location": location,
                            "location_name": (
                                "Sverige" if location == "SE" else "Testregion"
                            ),
                            "output_type": "mean",
                            "forecast_value": str(baseline + adjustment),
                            "observed_value": str(observed),
                            "outcome_data_version": "fixed-version",
                            "forecast_file": f"{model_id}/file.csv",
                        }
                    )
    return rows


class QraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells, cls.models = build_panel(synthetic_matches())
        cls.forecasts, cls.coefficients, cls.skipped = generate_qra(
            cls.cells, cls.models
        )

    def test_starts_after_eight_available_outcomes_for_each_horizon(self):
        self.assertEqual(len(self.forecasts), 154)
        starts = {}
        for row in self.forecasts:
            starts.setdefault(row["horizon"], row["reference_date"])
            self.assertGreaterEqual(row["training_sample_count"], 8)
            cutoff = date.fromisoformat(row["reference_date"]) - timedelta(days=7)
            self.assertLessEqual(
                date.fromisoformat(row["training_last_target_end_date"]), cutoff
            )
        self.assertEqual(starts[0], "2025-11-30")
        self.assertEqual(starts[3], "2025-12-21")
        self.assertEqual(len(self.skipped), 38)

    def test_quantiles_are_nonnegative_and_monotone(self):
        grouped = {}
        for row in self.forecasts:
            key = (row["reference_date"], row["location"], row["horizon"])
            grouped.setdefault(key, []).append(row)
        for rows in grouped.values():
            rows.sort(key=lambda row: float(row["output_type_id"]))
            self.assertEqual(
                tuple(float(row["output_type_id"]) for row in rows), QUANTILES
            )
            values = [float(row["value"]) for row in rows]
            self.assertTrue(all(value >= 0 for value in values))
            self.assertEqual(values, sorted(values))

    def test_preserves_auditable_model_coefficients(self):
        self.assertEqual(
            len(self.coefficients), len(self.forecasts) * (len(self.models) + 1)
        )
        terms = {row["term"] for row in self.coefficients}
        self.assertEqual(terms, {"intercept", "alpha-model", "beta-model"})

    def test_future_outcome_cannot_change_earlier_qra_forecasts(self):
        base_cells, models = build_panel(
            synthetic_matches(rounds=12, horizons=(0,))
        )
        changed_cells, _ = build_panel(
            synthetic_matches(
                rounds=12,
                horizons=(0,),
                final_outcome_offset=10000,
            )
        )
        base, _, _ = generate_qra(base_cells, models)
        changed, _, _ = generate_qra(changed_cells, models)
        self.assertEqual(base, changed)

    def test_coefficients_are_shared_across_locations_and_horizons(self):
        cells, models = build_panel(
            synthetic_matches(locations=("SE", "SE-M"))
        )
        _, coefficients, _ = generate_qra(cells, models)
        selected = [
            row
            for row in coefficients
            if row["reference_date"] == "2026-01-11"
            and row["output_type_id"] == "0.5"
        ]
        for term in ("intercept", *models):
            values = {
                row["coefficient"] for row in selected if row["term"] == term
            }
            self.assertEqual(len(values), 1)
        self.assertEqual(len({row["selected_alpha"] for row in selected}), 1)


if __name__ == "__main__":
    unittest.main()
