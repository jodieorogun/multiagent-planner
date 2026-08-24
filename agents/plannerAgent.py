from agents.base import BaseAgent
from typing import Any, Dict, List
import json

PLANNER_SCHEMA = """
{
  "type": "message",
  "content": {
    "academicTasks": [
      {
        "task": "string",
        "day": "string or null",
        "frequency": "number or null"
      }
    ],
    "sportsCommitments": [
      {
        "activity": "string",
        "day": "string or null",
        "frequency": "number or null"
      }
    ],
    "workoutGoals": [
      {
        "activity": "string",
        "frequency": "number"
      }
    ],
    "constraints": [
      {
        "type": "string",
        "value": "string"
      }
    ]
  }
}
"""

class PlannerAgent(BaseAgent):
    def run(self, message: Any, context: List[Dict[str, Any]]) -> Dict[str, Any]:

        # Build prompt WITHOUT any JSON braces inside f-strings
        prompt = (
            "You are PlannerAgent.\n"
            "You MUST output ONLY valid JSON.\n"
            "Use this EXACT JSON schema:\n\n"
            + PLANNER_SCHEMA +
            "\n\nFill the schema using this user request:\n"
            + str(message)
            + "\n"
        )

        # Call the model
        raw = self.llm(prompt)

        # If model already returned dict → done
        if isinstance(raw, dict):
            return raw

        # If returned string, attempt JSON extraction
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except:
                pass

        # If JSON is nested inside {"type":"message","content":"..."}
        if isinstance(raw, dict) and isinstance(raw.get("content"), str):
            try:
                return json.loads(raw["content"])
            except:
                pass

        raise ValueError(f"PlannerAgent returned non-JSON:\n{raw}")
