from agents.base import BaseAgent
from typing import Any, Dict, List
import json


class WriterAgent(BaseAgent):
    def run(self, message: Any, context: List[Dict[str, Any]]) -> Dict[str, Any]:

        # Extract context safely
        planner = next((c.get("message") for c in context if c.get("agent") == "PlannerAgent"), None)
        fitness = next((c.get("message") for c in context if c.get("agent") == "FitnessAgent"), None)
        nutrition = next((c.get("message") for c in context if c.get("agent") == "NutritionAgent"), None)
        critic_tool = next((c.get("toolResult") for c in context if c.get("agent") == "CriticAgent"), None)

        # Build LLM prompt — escape all JSON braces with double {{ }}
        prompt = f"""
You are WriterAgent.
Your job is to combine planning, workouts, nutrition, and stress level into a clean, human-readable weekly plan.

Here is all the structured data:

Planner output:
{json.dumps(planner, indent=2)}

Fitness output:
{json.dumps(fitness, indent=2)}

Nutrition output:
{json.dumps(nutrition, indent=2)}

Critic stress score (0–3):
{critic_tool}

Produce a warm, encouraging weekly plan. Use natural language.
RETURN STRICT JSON:

{{
  "type": "message",
  "content": "<the final written weekly plan>"
}}
"""

        return self.llm(prompt)
