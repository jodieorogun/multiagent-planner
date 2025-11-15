from typing import Any, Dict, List
from agents.base import Agent

class PlannerAgent(Agent):
    def run(self, message: Any, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = f"""
You are PlannerAgent.

User request:
{message}

Your job is to break this into categories:
- academicTasks
- sportsCommitments
- workoutGoals
- otherCommitments
- constraints (like sleep, energy, injuries)

Respond as JSON:
{{
  "type": "message",
  "content": {{
    "academicTasks": [...],
    "sportsCommitments": [...],
    "workoutGoals": [...],
    "otherCommitments": [...],
    "constraints": [...]
  }}
}}
"""
        return self.llm(prompt)
