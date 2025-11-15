from typing import Any, Dict, List
from agents.base_agent import BaseAgent

class NutritionAgent(BaseAgent):
    def run(self, message: Any, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = f"""
You are NutritionAgent.

You will receive:
- workoutPlan for the week
- constraints (like sleep, energy, weight goals) in the context if needed

Input:
{message}

Design a simple weekly nutrition outline with:
- dailyCalories (average)
- matchDayExtraCalories (if sports match)
- simpleMeals (list of easy, repeatable ideas)
- notes (short guidance)

Return ONLY JSON:
{{
  "type": "message",
  "content": {{
    "dailyCalories": 1900,
    "matchDayExtraCalories": 300,
    "simpleMeals": [
      "Greek yogurt with granola",
      "Microwave rice + chicken",
      "Tuna wrap + salad"
    ],
    "notes": "Short overall notes..."
  }}
}}
"""
        return self.llm(prompt)
