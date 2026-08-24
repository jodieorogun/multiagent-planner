from typing import Callable, List

from core.models import DAYS, PlanningBrief, ScheduleDraft, WorkloadAssessment, WorkloadFeatures
from tools.workload_model import heuristic_predict


WORKLOAD_LABELS = ("light", "moderate", "high", "burnout risk")


class CriticAgent:
    def __init__(self, predictor: Callable[[List[float]], int] = heuristic_predict):
        self.predictor = predictor

    def run(self, brief: PlanningBrief, schedule: ScheduleDraft = None) -> WorkloadAssessment:
        hours_study = 0.0
        deadlines = 0
        for task in brief.academic_tasks:
            lowered = task.task.lower()
            if "study" in lowered:
                hours_study += task.duration_hours * task.frequency
            if "coursework" in lowered or "deadline" in lowered:
                hours_study += max(task.duration_hours, 2.0)
                deadlines += 1

        daily_load = {day: 0.0 for day in DAYS}
        hours_sport = 0.0
        if schedule:
            for day, events in schedule.days.items():
                daily_load[day] += sum(event.duration_hours for event in events)
                hours_sport += sum(
                    event.duration_hours
                    for event in events
                    if event.name.lower() != "study"
                )

        study_days = (
            "Monday",
            "Thursday",
            "Sunday",
            "Saturday",
            "Tuesday",
            "Wednesday",
            "Friday",
        )
        available_study_days = [
            day
            for day in study_days
            if not schedule
            or not any("match" in event.name.lower() for event in schedule.days[day])
        ]
        next_study_day = 0
        for task in brief.academic_tasks:
            if task.day:
                daily_load[task.day] += task.duration_hours * task.frequency
                continue
            for _ in range(task.frequency):
                day = available_study_days[next_study_day % len(available_study_days)]
                daily_load[day] += task.duration_hours
                next_study_day += 1

        features = WorkloadFeatures(
            hours_study=hours_study,
            hours_sport=hours_sport,
            hours_work=0.0,
            num_deadlines=float(deadlines),
            sleep_hours=brief.minimum_sleep_hours,
        )
        score = int(self.predictor(features.as_list()))
        score = max(0, min(score, len(WORKLOAD_LABELS) - 1))
        label = WORKLOAD_LABELS[score]
        note = (
            f"Estimated workload is {label}. "
            f"Peak day: {max(daily_load, key=daily_load.get)}. "
            "Protect the requested sleep window."
        )
        return WorkloadAssessment(
            score=score,
            label=label,
            note=note,
            features=features,
            daily_load=daily_load,
            peak_day=max(daily_load, key=daily_load.get),
        )
