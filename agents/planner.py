import json
import re
from typing import Callable, Dict

from core.models import (
    AcademicTask,
    Constraint,
    DAYS,
    PlanningBrief,
    SportsCommitment,
    WorkoutGoal,
    normalise_day,
)


PLANNER_SCHEMA = {
    "type": "message",
    "content": {
        "academicTasks": [
            {"task": "string", "day": "day or null", "frequency": 1}
        ],
        "sportsCommitments": [
            {"activity": "string", "day": "day or null", "frequency": 1}
        ],
        "workoutGoals": [{"activity": "Gym", "frequency": 0}],
        "constraints": [{"type": "sleep", "value": "7 hours"}],
    },
}


class PlannerAgent:
    def __init__(self, llm: Callable[[str], Dict]):
        self.llm = llm

    def run(self, user_request: str) -> PlanningBrief:
        prompt = (
            "You are PlannerAgent. Extract only facts stated by the user. "
            "Keep fixed events on their stated days and use null when no day is given. "
            "Return this exact JSON shape:\n"
            + json.dumps(PLANNER_SCHEMA, indent=2)
            + "\n\nUser request:\n"
            + user_request
        )
        brief = PlanningBrief.from_llm_response(self.llm(prompt))
        return self._reconcile_explicit_facts(brief, user_request)

    def _reconcile_explicit_facts(
        self, brief: PlanningBrief, user_request: str
    ) -> PlanningBrief:
        """Protect unambiguous facts that a small model may omit or misclassify."""
        academic_tasks = [
            task
            for task in brief.academic_tasks
            if "match" not in task.task.lower() and "training" not in task.task.lower()
        ]
        sports_commitments = list(brief.sports_commitments)
        workout_goals = list(brief.workout_goals)
        constraints = list(brief.constraints)

        for segment in re.split(r"[,.;]", user_request):
            lowered = segment.lower()
            day_match = re.search(
                r"\b(" + "|".join(DAYS) + r")\b", segment, re.IGNORECASE
            )
            day = normalise_day(day_match.group(1)) if day_match else None

            if day and "match" in lowered:
                activity_match = re.search(r"\b([a-z]+)\s+match\b", lowered)
                sport = activity_match.group(1).title() if activity_match else "Sports"
                explicit_commitment = SportsCommitment(
                    activity=f"{sport} match", day=day
                )
                matching_indexes = [
                    index
                    for index, item in enumerate(sports_commitments)
                    if item.day == day and sport.lower() in item.activity.lower()
                ]
                if matching_indexes:
                    sports_commitments[matching_indexes[0]] = explicit_commitment
                elif not any(
                    item.day == day and "match" in item.activity.lower()
                    for item in sports_commitments
                ):
                    sports_commitments.append(explicit_commitment)

            if day and "training" in lowered and not any(
                item.day == day and "training" in item.activity.lower()
                for item in sports_commitments
            ):
                sports_commitments.append(
                    SportsCommitment(activity="Training", day=day)
                )

            if day and "deadline" in lowered and not any(
                task.day == day and "deadline" in task.task.lower()
                for task in academic_tasks
            ):
                name = "Coursework deadline" if "coursework" in lowered else "Deadline"
                academic_tasks.append(
                    AcademicTask(task=name, day=day, duration_hours=4.0)
                )

        study_match = re.search(
            r"\b(\d+)\s+(?:evenings?|sessions?)\b.{0,35}\bstud(?:y|ying)\b",
            user_request,
            re.IGNORECASE,
        )
        if study_match:
            frequency = min(int(study_match.group(1)), 7)
            academic_tasks = [
                task for task in academic_tasks if "study" not in task.task.lower()
            ]
            academic_tasks.append(AcademicTask(task="Study", frequency=frequency))

        gym_match = re.search(
            r"(?:\bgym\s+(\d+)\s+times?\b|\b(\d+)\s+(?:gym|workout)\s+sessions?\b)",
            user_request,
            re.IGNORECASE,
        )
        if gym_match:
            frequency = min(int(gym_match.group(1) or gym_match.group(2)), 7)
            workout_goals = [
                goal
                for goal in workout_goals
                if goal.activity.lower() not in {"gym", "workout", "gymming"}
            ]
            workout_goals.append(WorkoutGoal(activity="Gym", frequency=frequency))

        sleep_match = re.search(
            r"(?:\bsleep\b.{0,30}\b(\d+)\s+hours?\b|"
            r"\b(\d+)\s+hours?\b.{0,30}\bsleep\b)",
            user_request,
            re.IGNORECASE,
        )
        if sleep_match:
            hours = sleep_match.group(1) or sleep_match.group(2)
            constraints = [
                item for item in constraints if "sleep" not in item.type.lower()
            ]
            constraints.append(Constraint(type="sleep", value=f"{hours} hours"))

        return PlanningBrief(
            academic_tasks=academic_tasks,
            sports_commitments=sports_commitments,
            workout_goals=workout_goals,
            constraints=constraints,
        )
