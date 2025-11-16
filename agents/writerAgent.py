from agents.base import BaseAgent
from typing import Any, Dict, List
import json

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

OUTPUT_SPEC = """
{
  "type": "message",
  "content": {
    "weeklyPlan": {
      "Monday": "<summary>",
      "Tuesday": "<summary>",
      "Wednesday": "<summary>",
      "Thursday": "<summary>",
      "Friday": "<summary>",
      "Saturday": "<summary>",
      "Sunday": "<summary>"
    },
    "stressNote": "<1 short sentence about workload risk>"
  }
}
"""

class WriterAgent(BaseAgent):
    def run(self, message: Any, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        # extract structured info from previous agents
        planner = next((c.get("message") for c in context if c.get("agent") == "PlannerAgent"), {})
        fitness = next((c.get("message") for c in context if c.get("agent") == "FitnessAgent"), {})
        nutrition = next((c.get("message") for c in context if c.get("agent") == "NutritionAgent"), {})
        stress = next((c.get("toolResult") for c in context if c.get("agent") == "CriticAgent"), None)

        workoutPlan = fitness.get("workoutPlan", {})
        # we won’t over-use these, but keep them in case
        dailyCalories = nutrition.get("dailyCalories", None)
        matchExtra = nutrition.get("matchDayExtra", None)

        # ----- build prompt WITHOUT f-strings using braces -----
        prompt_template = """
You are WriterAgent.

Your task is to MERGE the planner, fitness, nutrition, and stress data into a clean weekly plan.

IMPORTANT RULES (DO NOT BREAK THESE):
- You MUST keep events on their correct days based on the input.
- If PlannerAgent or FitnessAgent gives a day (e.g., "Wednesday"), you MUST put it on Wednesday.
- NEVER move a match or training session to another day.
- If Planner gives frequency but no day, ignore it unless FitnessAgent assigned a day.
- If NutritionAgent gives matchDayExtra, APPLY it only to the day where FitnessAgent marked a match.
- If a day ends up empty, output "Rest".

YOU MUST output EXACT JSON:

__OUTPUT_SPEC__

INPUT DATA (DO NOT INVENT ANYTHING):

Planner:
__PLANNER__

WorkoutPlan:
__WORKOUT__

Nutrition:
__NUTRITION__

StressScore (0=low, 1=moderate, 2=high, 3=burnout):
__STRESS__

Extra logic rules:
- Always prioritise FitnessAgent's days (the workoutPlan keys).
- Insert academic tasks from Planner only on the days they specify.
- Insert sports from Planner only on the days they specify.
- Add "Extra calories today" ONLY if matchDayExtra > 0 AND workoutPlan shows a match on that day.
"""

        prompt = (
            prompt_template
            .replace("__OUTPUT_SPEC__", OUTPUT_SPEC)
            .replace("__PLANNER__", json.dumps(planner, indent=2))
            .replace("__WORKOUT__", json.dumps(workoutPlan, indent=2))
            .replace("__NUTRITION__", json.dumps(nutrition, indent=2))
            .replace("__STRESS__", str(stress))
        )

        return self.llm(prompt)
