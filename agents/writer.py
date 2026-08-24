import json
from dataclasses import asdict
from typing import Callable, Dict

from core.llm import LLMError
from core.models import (
    DAYS,
    FinalPlan,
    NutritionPlan,
    PlanningBrief,
    ScheduleDraft,
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
        schedule: ScheduleDraft,
        nutrition: NutritionPlan,
        workload: WorkloadAssessment,
    ) -> FinalPlan:
        draft = self._build_draft(brief, schedule, nutrition)
        prompt = self._build_prompt(draft, workload, schedule.conflicts)

        try:
            plan = FinalPlan.from_llm_response(self.llm(prompt))
        except (LLMError, SchemaError):
            return self._fallback_plan(draft, workload, schedule.conflicts, "fallback")

        protected_plan = {}
        repaired = False
        for day in DAYS:
            summary = plan.weekly_plan[day]
            required = draft[day]
            summary_lower = summary.lower()
            has_wrong_count = any(
                summary_lower.count(item.lower()) != required.count(item)
                for item in set(required)
            )
            if required and has_wrong_count:
                summary = " / ".join(required)
                repaired = True
            elif not required and not summary_lower.startswith("rest"):
                summary = "Rest"
                repaired = True
            protected_plan[day] = summary or "Rest"

        stress_note = plan.stress_note
        if workload.label.lower() not in stress_note.lower():
            stress_note = workload.note
            repaired = True

        return FinalPlan(
            weekly_plan=protected_plan,
            stress_note=stress_note,
            writer_mode="ollama+validated" if repaired else "ollama",
            workload=workload,
            conflicts=schedule.conflicts,
        )

    def _build_draft(self, brief, schedule, nutrition):
        draft = {
            day: [event.name for event in schedule.days[day]] for day in DAYS
        }

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

        for day, note in nutrition.daily_notes.items():
            draft[day].append(note)
        return draft

    @staticmethod
    def _build_prompt(draft, workload, conflicts):
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
            + "\n\nScheduling conflicts to mention if relevant:\n"
            + json.dumps(conflicts, indent=2)
        )

    @staticmethod
    def _fallback_plan(draft, workload, conflicts, mode):
        return FinalPlan(
            weekly_plan={
                day: " / ".join(activities) if activities else "Rest"
                for day, activities in draft.items()
            },
            stress_note=workload.note,
            writer_mode=mode,
            workload=workload,
            conflicts=conflicts,
        )
