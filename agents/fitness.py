from core.models import DAYS, PlanningBrief


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

        for commitment in brief.sports_commitments:
            if commitment.day:
                schedule[commitment.day].append(commitment.activity)

        gym_sessions = min(brief.gym_sessions, 7)
        sessions_added = 0
        for day in self.GYM_DAY_ORDER:
            if sessions_added >= gym_sessions:
                break
            if any("match" in activity.lower() for activity in schedule[day]):
                continue
            schedule[day].append("Gym")
            sessions_added += 1

        return schedule
