import unittest

from core.models import FinalPlan
from evals.run_eval import score_plan


class EvalScoringTests(unittest.TestCase):
    def test_scores_fixed_events_counts_and_workload(self):
        plan = FinalPlan(
            weekly_plan={
                "Monday": "Gym / Study",
                "Tuesday": "Training",
                "Wednesday": "Lacrosse match",
                "Thursday": "Gym / Study",
                "Friday": "Coursework deadline",
                "Saturday": "Rest",
                "Sunday": "Rest",
            },
            stress_note="Estimated workload is high.",
            writer_mode="ollama",
        )
        expectations = {
            "fixed_events": [
                {"day": "Wednesday", "text": "Lacrosse match"}
            ],
            "exact_counts": {"Gym": 2, "Study": 2},
            "workload": "high",
        }

        checks = score_plan(plan, expectations)

        self.assertTrue(all(check.passed for check in checks))

    def test_reports_a_failed_expectation(self):
        plan = FinalPlan(
            weekly_plan={day: "Rest" for day in (
                "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
            )},
            stress_note="Estimated workload is light.",
            writer_mode="fallback",
        )

        checks = score_plan(
            plan,
            {"fixed_events": [{"day": "Friday", "text": "Deadline"}]},
        )

        self.assertFalse(checks[0].passed)


if __name__ == "__main__":
    unittest.main()
