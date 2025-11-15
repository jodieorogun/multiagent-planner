from agents.base import BaseAgent
import json
from typing import Any, Dict, List

class CriticAgent(BaseAgent):

    def run(self, message: Any, context: List[Dict[str, Any]]) -> Dict[str, Any]:

        # message = nutrition output or fitness output
        # context includes all previous agents

        # --- Extract rough estimates ---
        # We use simple heuristics because the LLM doesn't generate durations.

        hoursStudy = 0
        hoursSport = 0
        hoursWork = 0
        numDeadlines = 0
        sleepHours = 7  # default constraint

        # Look for constraints
        for item in context:
            if item.get("agent") == "PlannerAgent":
                planner = item.get("message", {})
                constraints = planner.get("constraints", [])
                for c in constraints:
                    if "7" in str(c):
                        sleepHours = 7
                    if "8" in str(c):
                        sleepHours = 8

        # Find academic tasks
        for item in context:
            if item.get("agent") == "PlannerAgent":
                academic = item["message"].get("academicTasks", [])
                for a in academic:
                    if "study" in str(a).lower():
                        hoursStudy += 3
                    if "coursework" in str(a).lower():
                        numDeadlines += 1
                        hoursStudy += 4

        # Find sports
        for item in context:
            if item.get("agent") == "PlannerAgent":
                sports = item["message"].get("sportsCommitments", [])
                for s in sports:
                    text = str(s).lower()
                    if "match" in text:
                        hoursSport += 2
                    if "training" in text:
                        if "2" in text:
                            hoursSport += 3
                        else:
                            hoursSport += 1.5

        # Find gym workouts
        for item in context:
            if item.get("agent") == "PlannerAgent":
                wg = item["message"].get("workoutGoals", [])
                for w in wg:
                    if "gym" in str(w).lower():
                        hoursSport += 4  # rough estimate 1h × 4


        # --- Build tool call for PyTorch model ---
        features = [hoursStudy, hoursSport, hoursWork, numDeadlines, sleepHours]

        return {
            "type": "toolCall",
            "toolName": "workloadPredictor",
            "args": features
        }
