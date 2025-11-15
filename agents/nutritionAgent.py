from agents.base import BaseAgent
from typing import Any, Dict, List


class NutritionAgent(BaseAgent):
    def __init__(self, name, llm=None):
        super().__init__(name, llm)

    def run(self, message: Dict[str, Any], context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        message is the 'content' from FitnessAgent:
        {
          "workoutPlan": { "Mon": "...", ... }
        }
        """
        workoutPlan = message.get("workoutPlan", {})

        # baseline calories
        totalWorkouts = sum(1 for v in workoutPlan.values() if "gym" in v.lower())
        if totalWorkouts >= 4:
            dailyCalories = 2200
        else:
            dailyCalories = 2000

        matchDayExtra = 300 if any("match" in v.lower() for v in workoutPlan.values()) else 0

        meals = [
            "Breakfast: Greek yogurt + granola",
            "Lunch: Rice, chicken, and veg",
            "Dinner: High-protein pasta",
        ]

        return {
            "type": "message",
            "content": {
                "dailyCalories": dailyCalories,
                "matchDayExtra": matchDayExtra,
                "meals": meals,
            }
        }
