from agents.base import BaseAgent
from typing import Any, Dict, List

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class FitnessAgent(BaseAgent):
    def __init__(self, name, llm=None):
        super().__init__(name, llm)

    def run(self, message: Dict[str, Any], context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Expected message from PlannerAgent:
        {
          "academicTasks": [...],
          "sportsCommitments": [...],
          "workoutGoals": [...],
          "constraints": [...]
        }
        """

        academicTasks = message.get("academicTasks", [])
        sportsCommitments = message.get("sportsCommitments", [])
        workoutGoals = message.get("workoutGoals", [])

        workoutPlan = {day: "" for day in WEEKDAYS}

        # -----------------------------
        # 1) Place fixed sports events
        # -----------------------------
        sportsByDay = {}

        for sc in sportsCommitments:
            day = sc.get("day")
            name = sc.get("activity") or sc.get("name") or "Sports"

            if not day:
                continue

            short = day[:3].title()
            if short not in workoutPlan:
                continue

            sportsByDay.setdefault(short, []).append(name)

        for d, events in sportsByDay.items():
            workoutPlan[d] = " / ".join(events)

        # -----------------------------
        # 2) Gym session extraction
        # -----------------------------
        gymSessions = 0
        for wg in workoutGoals:
            if wg.get("activity", "").lower() in ["gym", "gymming", "workout"]:
                freq = wg.get("frequency")
                try:
                    gymSessions = int(freq)
                except:
                    pass

        # -----------------------------
        # 3) Place gym sessions
        # -----------------------------
        if gymSessions > 0:
            preferredOrder = ["Mon", "Tue", "Thu", "Sat", "Sun", "Fri", "Wed"]
            used = 0

            for day in preferredOrder:
                if used >= gymSessions:
                    break

                if workoutPlan[day] == "":
                    workoutPlan[day] = "Gym"
                    used += 1
                else:
                    # if not a match day, combine
                    if "match" not in workoutPlan[day].lower():
                        workoutPlan[day] += " + Gym"
                        used += 1

        return {
            "type": "message",
            "content": {
                "workoutPlan": workoutPlan
            }
        }
