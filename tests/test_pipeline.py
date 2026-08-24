import unittest

from core import AgentManager


DEMO_REQUEST = (
    "I have a lacrosse match on Wednesday, training on Tuesday, "
    "3 evenings to study, a coursework deadline on Friday, "
    "I want to gym 4 times and sleep at least 7 hours."
)


class PipelineTests(unittest.TestCase):
    def test_preserves_fixed_events_and_requested_gym_count(self):
        plan = AgentManager().process(DEMO_REQUEST)
        all_activities = [activity for day in plan.days.values() for activity in day]

        self.assertIn("Lacrosse match", plan.days["Wednesday"])
        self.assertIn("Training", plan.days["Tuesday"])
        self.assertIn("Coursework deadline", plan.days["Friday"])
        self.assertEqual(all_activities.count("Gym session"), 4)
        self.assertEqual(all_activities.count("Study session"), 3)
        self.assertEqual(plan.workload.level, "high")

    def test_empty_request_produces_a_low_pressure_week(self):
        plan = AgentManager().process("")

        self.assertTrue(all(activities == [] for activities in plan.days.values()))
        self.assertEqual(plan.workload.level, "low")
        self.assertIn("Rest / flexible time", plan.render())


if __name__ == "__main__":
    unittest.main()
