import unittest

from agents.planner import PlannerAgent
from tests.fakes import ScriptedLLM, planner_response


class PlannerAgentTests(unittest.TestCase):
    def test_converts_llm_json_to_a_typed_brief(self):
        llm = ScriptedLLM(planner_response())
        brief = PlannerAgent(llm).run("plan my week")

        self.assertEqual(brief.gym_sessions, 4)
        self.assertEqual(brief.minimum_sleep_hours, 7)
        self.assertEqual(brief.sports_commitments[0].day, "Wednesday")
        self.assertIn("exact JSON shape", llm.prompts[0])

    def test_reconciles_explicit_facts_missed_by_the_llm(self):
        confused_response = {
            "type": "message",
            "content": {
                "academicTasks": [
                    {"task": "Prepare lacrosse match", "day": None, "frequency": 1}
                ],
                "sportsCommitments": [],
                "workoutGoals": [],
                "constraints": [],
            },
        }
        request = (
            "Lacrosse match on Wednesday, training on Tuesday, 3 evenings to "
            "study, coursework deadline on Friday, gym 4 times and sleep 7 hours"
        )

        brief = PlannerAgent(ScriptedLLM(confused_response)).run(request)

        self.assertEqual(
            [(item.activity, item.day) for item in brief.sports_commitments],
            [("Lacrosse match", "Wednesday"), ("Training", "Tuesday")],
        )
        self.assertEqual(brief.gym_sessions, 4)
        self.assertEqual(brief.minimum_sleep_hours, 7)
        self.assertIn(
            ("Coursework deadline", "Friday"),
            [(task.task, task.day) for task in brief.academic_tasks],
        )


if __name__ == "__main__":
    unittest.main()
