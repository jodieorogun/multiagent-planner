import unittest

from agents.planner import PlannerAgent


class PlannerAgentTests(unittest.TestCase):
    def test_extracts_the_demo_request(self):
        request = (
            "I have a lacrosse match on Wednesday, training on Tuesday, "
            "3 evenings to study, a coursework deadline on Friday, "
            "I want to gym 4 times and sleep at least 7 hours."
        )

        parsed = PlannerAgent().run(request)

        self.assertEqual(parsed.gym_sessions, 4)
        self.assertEqual(parsed.minimum_sleep_hours, 7)
        self.assertEqual(
            [(item.activity, item.day) for item in parsed.sports_commitments],
            [("Lacrosse match", "Wednesday"), ("Training", "Tuesday")],
        )
        self.assertEqual(
            [(task.name, task.day, task.sessions) for task in parsed.academic_tasks],
            [
                ("Coursework deadline", "Friday", 1),
                ("Study session", None, 3),
            ],
        )

    def test_defaults_to_no_optional_commitments(self):
        parsed = PlannerAgent().run("Give me a calm week")

        self.assertEqual(parsed.academic_tasks, [])
        self.assertEqual(parsed.sports_commitments, [])
        self.assertEqual(parsed.gym_sessions, 0)
        self.assertEqual(parsed.minimum_sleep_hours, 7)


if __name__ == "__main__":
    unittest.main()
