import unittest

from core.manager import AgentManager
from core.models import DAYS
from tests.fakes import ScriptedLLM, planner_response


def writer_response():
    plan = {day: "Rest" for day in DAYS}
    plan.update(
        {
            "Monday": "Gym and Study",
            "Tuesday": "Training",
            "Wednesday": "Lacrosse match and Extra match fuel (+300 calories)",
            "Thursday": "Gym and Study",
            "Friday": "Coursework deadline",
            "Saturday": "Gym",
            "Sunday": "Gym and Study",
        }
    )
    return {
        "type": "message",
        "content": {
            "weeklyPlan": plan,
            "stressNote": "High workload: keep one evening flexible.",
        },
    }


class PipelineTests(unittest.TestCase):
    def test_runs_the_full_agent_pipeline_with_injected_llm(self):
        llm = ScriptedLLM(planner_response(), writer_response())
        manager = AgentManager(llm, workload_predictor=lambda features: 2)

        result = manager.process("plan my week")

        self.assertEqual(result.writer_mode, "ollama")
        self.assertIn("Lacrosse match", result.weekly_plan["Wednesday"])
        self.assertIn("Coursework deadline", result.weekly_plan["Friday"])
        self.assertIn("Canonical draft", llm.prompts[1])

    def test_writer_repairs_an_llm_plan_that_drops_fixed_events(self):
        invalid_writer_plan = writer_response()
        invalid_writer_plan["content"]["weeklyPlan"]["Wednesday"] = "Rest"
        llm = ScriptedLLM(planner_response(), invalid_writer_plan)

        result = AgentManager(llm).process("plan my week")

        self.assertEqual(result.writer_mode, "ollama+validated")
        self.assertIn("Lacrosse match", result.weekly_plan["Wednesday"])
        self.assertIn("Extra match fuel", result.weekly_plan["Wednesday"])

    def test_writer_falls_back_when_its_response_has_the_wrong_shape(self):
        llm = ScriptedLLM(planner_response(), {"unexpected": "response"})

        result = AgentManager(llm).process("plan my week")

        self.assertEqual(result.writer_mode, "fallback")
        self.assertIn("Training", result.weekly_plan["Tuesday"])

    def test_writer_repairs_duplicate_activities(self):
        duplicated_writer_plan = writer_response()
        duplicated_writer_plan["content"]["weeklyPlan"]["Monday"] = (
            "Study, Gym and another Study session"
        )
        llm = ScriptedLLM(planner_response(), duplicated_writer_plan)

        result = AgentManager(llm).process("plan my week")

        self.assertEqual(result.writer_mode, "ollama+validated")
        self.assertEqual(result.weekly_plan["Monday"], "Gym / Study")


if __name__ == "__main__":
    unittest.main()
