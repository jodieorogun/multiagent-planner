from core.models import DAYS, ParsedRequest, WeeklyPlan, WorkloadAssessment


class WriterAgent:
    """Merge specialist outputs into one seven-day plan."""

    STUDY_DAY_ORDER = (
        "Monday",
        "Thursday",
        "Sunday",
        "Saturday",
        "Tuesday",
        "Wednesday",
        "Friday",
    )

    def run(
        self,
        request: ParsedRequest,
        fitness_schedule,
        nutrition_notes,
        workload: WorkloadAssessment,
    ) -> WeeklyPlan:
        days = {day: list(fitness_schedule[day]) for day in DAYS}

        for task in request.academic_tasks:
            if task.day:
                days[task.day].append(task.name)
                continue

            available_days = [
                day
                for day in self.STUDY_DAY_ORDER
                if not any("match" in item.lower() for item in days[day])
            ]
            for day in available_days[: task.sessions]:
                days[day].append(task.name)

        for day, note in nutrition_notes.items():
            days[day].append(note)

        return WeeklyPlan(days=days, workload=workload)
