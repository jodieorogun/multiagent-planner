from typing import Any, Dict, List
from agents.base_agent import BaseAgent

class WriterAgent(BaseAgent):
    def run(self, message: Any, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = f"""
You are WriterAgent.

Combine ALL the info in the context into a clear, realistic weekly plan.

context:
{context}

latest message:
{message}

Produce a final plan that includes:
- Day-by-day breakdown (Mon-Sun)
- Workout per day (or rest)
- Key study focus blocks
- Short nutrition notes (calories + simple meals)
- Any adjustments based on workload (e.g., move workouts, add rest)

Return ONLY JSON:
{{
  "type": "message",
  "content": "final weekly plan as a multi-line string"
}}
"""
        return self.llm(prompt)
