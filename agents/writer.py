import json
from dataclasses import asdict
from typing import Callable, Dict

from core.llm import LLMError
from core.models import (
    DAYS,
    FinalPlan,
    NutritionPlan,
    PlanningBrief,
    SchemaError,
    WorkloadAssessment,
)


class WriterAgent:
    STUDY_DAY_ORDER = (
        "Monday",
        "Thursday",
        "Sunday",
        "Saturday",
        "Tuesday",
        "Wednesday",
        "Friday",
    )

    def __init__(self, llm: Callable[[str], Dict]):
        self.llm = llm

    def run(
        self,
        brief: PlanningBrief,
        workout_plan,
        nutrition: NutritionPlan,
        workload: WorkloadAssessment,
    ) -> FinalPlan:
        draft = self._build_draft(brief, workout_plan, nutrition)
        prompt = self._build_prompt(draft, workload)

        try:
            plan = FinalPlan.from_llm_response(self.llm(prompt))
        except (LLMError, SchemaError):
            return self._fallback_plan(draft, workload, "fallback")

        protected_plan = {}
        repaired = False
        for day in DAYS:
            summary = plan.weekly_plan[day]
            required = draft[day]
            if required and not all(
                item.lower() in summary.lower() for item in required
            ):
                summary = " / ".join(required)
                repaired = True
            protected_plan[day] = summary or "Rest"

        return FinalPlan(
            weekly_plan=protected_plan,
            stress_note=plan.stress_note or workload.note,
            writer_mode="ollama+validated" if repaired else "ollama",
        )

    def _build_draft(self, brief, workout_plan, nutrition):
        draft = {day: list(workout_plan[day]) for day in DAYS}

        for task in brief.academic_tasks:
            if task.day:
                draft[task.day].append(task.task)
                continue

            available_days = [
                day
                for day in self.STUDY_DAY_ORDER
                if not any("match" in item.lower() for item in draft[day])
            ]
            for day in available_days[: task.frequency]:
                draft[day].append(task.task)

        for day in nutrition.extra_fuel_days:
            draft[day].append(
                f"Extra match fuel (+{nutrition.extra_calories_amount} calories)"
            )
        return draft

    @staticmethod
    def _build_prompt(draft, workload):
        output_schema = {
            "type": "message",
            "content": {
                "weeklyPlan": {day: "short summary" for day in DAYS},
                "stressNote": "one short sentence",
            },
        }
        return (
            "You are WriterAgent. Turn the canonical draft into concise daily summaries. "
            "Every listed activity must remain on its current day. Do not invent or move "
            "commitments. Use Rest for empty days. Return this JSON shape:\n"
            + json.dumps(output_schema, indent=2)
            + "\n\nCanonical draft:\n"
            + json.dumps(draft, indent=2)
            + "\n\nWorkload assessment:\n"
            + json.dumps(asdict(workload), indent=2)
        )

    @staticmethod
    def _fallback_plan(draft, workload, mode):
        return FinalPlan(
            weekly_plan={
                day: " / ".join(activities) if activities else "Rest"
                for day, activities in draft.items()
            },
            stress_note=workload.note,
            writer_mode=mode,
        )
