from typing import Any, Dict, List
from agents.base_agent import BaseAgent

class CriticAgent(BaseAgent):
    def run(self, message: Any, context: List[Dict[str, Any]]) -> Dict[str, Any]:

        prompt = f"""
You are CriticAgent.

You see the current weekly plan and nutrition data:
message: {message}
context: {context}

1. Estimate approximate numeric features:
   - hoursStudy (per day or per week, convert to a single weekly number or average)
   - hoursSport (per week)
   - hoursWork (per week, assume 0 if unknown)
   - numDeadlines (this week)
   - sleepHours (average per night, try to honour requested >= 7h).

2. Decide whether the workload is light, moderate, high, or burnout risk.

3. If you want an objective estimate, call the workloadPredictor tool with those numeric features.

If you decide to call the tool, respond ONLY as:
{{
  "type": "toolCall",
  "toolName": "workloadPredictor",
  "args": [hoursStudy, hoursSport, hoursWork, numDeadlines, sleepHours]
}}

If you decide you do NOT need the tool, respond ONLY as:
{{
  "type": "message",
  "content": "Your written critique and concrete adjustments to the weekly plan..."
}}
"""
        return self.llm(prompt)
