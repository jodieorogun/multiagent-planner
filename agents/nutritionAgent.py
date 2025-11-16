from agents.base import BaseAgent
from typing import Any, Dict, List
import json

class NutritionAgent(BaseAgent):
    def __init__(self, name, llm=None):
        super().__init__(name, llm)

    def run(self, message: Any, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        message = { "workoutPlan": { "Mon": "...", ... } }
        """
        workoutPlan = message.get("workoutPlan", {})

        # Identify days where extra fuelling is needed (matches)
        extraFuelDays = []
        for day, activity in workoutPlan.items():
            if activity and "match" in activity.lower():
                extraFuelDays.append(day)

        # Calorie rules:
        dailyCalories = 2000
        extraCaloriesAmount = 300 if extraFuelDays else 0

        # Fixed meals (simple + good for athletes)
        meals = [
            "Breakfast: Greek yogurt + granola",
            "Lunch: Chicken, rice, vegetables",
            "Dinner: High-protein pasta"
        ]

        # STRICT JSON prompt for safety
        prompt = f"""
You are NutritionAgent.

You MUST output STRICT JSON ONLY:

{{
  "type": "message",
  "content": {{
    "dailyCalories": <number>,
    "extraFuelDays": ["Monday", ...],
    "extraCaloriesAmount": <number>,
    "meals": ["...", "...", "..."]
  }}
}}

Rules:
- dailyCalories = 2000.
- extraFuelDays = only days where workoutPlan contains a match.
- extraCaloriesAmount = 300 if there is at least one match, otherwise 0.
- meals must ALWAYS be the 3 simple meals below:
  - Breakfast: Greek yogurt + granola
  - Lunch: Chicken, rice, vegetables
  - Dinner: High-protein pasta

Input workoutPlan:
{json.dumps(workoutPlan, indent=2)}
"""

        return self.llm(prompt)
