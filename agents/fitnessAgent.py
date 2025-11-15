from typing import Any, Dict, List
from agents.base_agent import BaseAgent

class FitnessAgent(BaseAgent):
    def run(self, message: Any, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = f"""
You are FitnessAgent.

User's planning data:
{message}

Use:
- sportsCommitments (e.g., lacrosse match, training)
- workoutGoals (e.g., gym 4x per week)
- constraints (e.g., "no heavy legs before match", "sleep >= 7 hours")

Design a weekly workout plan: Monday to Sunday.

Return ONLY JSON:
{{
  "type": "message",
  "content": {{
    "workoutPlan": {{
      "Monday": "Push",
      "Tuesday": "Pull",
      "Wednesday": "Match day - rest",
      "Thursday": "Legs",
      "Friday": "Push",
      "Saturday": "Active recovery",
      "Sunday": "Rest"
    }}
  }}
}}
Fill in realistic values based on the user data.
"""
        return self.llm(prompt)
