from core.models import DAYS, PlanningBrief, ScheduleDraft, ScheduledEvent


class FitnessAgent:
    GYM_DAY_ORDER = (
        "Monday",
        "Thursday",
        "Saturday",
        "Sunday",
        "Tuesday",
        "Friday",
        "Wednesday",
    )

    def run(self, brief: PlanningBrief):
        schedule = {day: [] for day in DAYS}
        conflicts = []

        for commitment in brief.sports_commitments:
            if commitment.day:
                schedule[commitment.day].append(
                    ScheduledEvent(
                        name=commitment.activity,
                        duration_hours=commitment.duration_hours,
                        priority=commitment.priority,
                        source="user",
                    )
                )

        gym_sessions = min(brief.gym_sessions, 7)
        sessions_added = 0
        for day in self.GYM_DAY_ORDER:
            if sessions_added >= gym_sessions:
                break
            if any("match" in event.name.lower() for event in schedule[day]):
                continue
            gym_goal = next(
                (
                    goal
                    for goal in brief.workout_goals
                    if goal.activity.lower() in {"gym", "workout", "gymming"}
                ),
                None,
            )
            schedule[day].append(
                ScheduledEvent(
                    name="Gym",
                    duration_hours=gym_goal.duration_hours if gym_goal else 1.0,
                    priority="planned",
                    source="workout goal",
                )
            )
            sessions_added += 1

        for day, events in schedule.items():
            total_hours = sum(event.duration_hours for event in events)
            if total_hours > 8:
                conflicts.append(
                    f"{day} contains {total_hours:g} scheduled hours; consider moving an optional event."
                )

        return ScheduleDraft(days=schedule, conflicts=conflicts)
